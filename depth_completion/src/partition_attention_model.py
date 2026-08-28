import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class PartitionPyramid(nn.Module):
    '''
    Builds learned RGB partitions at 1, 1/2, 1/4 and 1/8 resolution.

    In parallel mode every convolution reads the RGB image. In sequential mode
    each convolution reads the preceding partition. The sequential kernels are
    1, 2, 4 and 8 pixels wide; padding keeps every strided output at exactly
    half of its input resolution.
    '''

    SCALES = (1, 2, 4, 8)

    def __init__(self, n_channels=64, mode='parallel'):
        super(PartitionPyramid, self).__init__()

        if mode not in ('parallel', 'sequential'):
            raise ValueError(
                'Unsupported partition pyramid mode: {}'.format(mode))

        self.mode = mode
        n_group = min(8, n_channels)
        while n_channels % n_group != 0:
            n_group = n_group - 1

        if mode == 'parallel':
            convolutions = [
                nn.Conv2d(
                    3,
                    n_channels,
                    kernel_size=scale,
                    stride=scale,
                    padding=0)
                for scale in self.SCALES
            ]
        else:
            # With inputs divisible by eight these settings produce exact
            # 1, 1/2, 1/4 and 1/8 shapes while increasing kernel width.
            kernel_sizes = (1, 2, 4, 8)
            strides = (1, 2, 2, 2)
            paddings = (0, 0, 1, 3)
            convolutions = []
            for index, (kernel_size, stride, padding) in enumerate(zip(
                    kernel_sizes, strides, paddings)):
                n_input = 3 if index == 0 else n_channels
                convolutions.append(nn.Conv2d(
                    n_input,
                    n_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding))

        self.convolutions = nn.ModuleList(convolutions)
        self.normalizations = nn.ModuleList([
            nn.GroupNorm(n_group, n_channels)
            for _ in self.SCALES
        ])
        self.activation = nn.GELU()

    def forward(self, image):
        outputs = []
        x = image

        for convolution, normalization in zip(
                self.convolutions, self.normalizations):
            convolution_input = image if self.mode == 'parallel' else x
            x = self.activation(normalization(convolution(convolution_input)))
            outputs.append(x)

        return outputs


class PartitionAttentionBlock(nn.Module):
    '''Local fine-window attention followed by hierarchical cross-attention.'''

    def __init__(self,
                 n_channels,
                 n_head,
                 n_scale=4,
                 max_local_tokens=32,
                 max_context_tokens=16,
                 dropout=0.0):
        super(PartitionAttentionBlock, self).__init__()

        if n_channels % n_head != 0:
            raise ValueError(
                'n_channels ({}) must be divisible by n_head ({})'.format(
                    n_channels, n_head))

        self.max_local_tokens = max_local_tokens
        self.max_context_tokens = max_context_tokens

        self.local_norm = nn.LayerNorm(n_channels)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)

        self.query_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)
        self.scale_embedding = nn.Parameter(torch.zeros(
            n_scale, 1, n_channels))

        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_channels, n_channels),
            nn.Dropout(dropout))

        nn.init.normal_(self.scale_embedding, std=0.02)

    @staticmethod
    def _to_windows(x, n_grid):
        n_batch, n_channel, n_height, n_width = x.shape

        if n_height % n_grid != 0 or n_width % n_grid != 0:
            raise ValueError(
                'Feature shape {}x{} cannot be divided into a {}x{} grid'.format(
                    n_height, n_width, n_grid, n_grid))

        window_height = n_height // n_grid
        window_width = n_width // n_grid
        windows = x.reshape(
            n_batch,
            n_channel,
            n_grid,
            window_height,
            n_grid,
            window_width)
        windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = windows.reshape(
            n_batch * n_grid * n_grid,
            window_height * window_width,
            n_channel)

        metadata = (
            n_batch,
            n_channel,
            n_height,
            n_width,
            n_grid,
            window_height,
            window_width)

        return windows, metadata

    @staticmethod
    def _from_windows(windows, metadata):
        n_batch, n_channel, n_height, n_width, n_grid, \
            window_height, window_width = metadata
        x = windows.reshape(
            n_batch,
            n_grid,
            n_grid,
            window_height,
            window_width,
            n_channel)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

        return x.reshape(n_batch, n_channel, n_height, n_width)

    @staticmethod
    def _pool_windows(windows, window_height, window_width, max_tokens):
        if window_height * window_width <= max_tokens:
            return windows

        target_height = max(1, min(
            window_height,
            int(math.sqrt(
                max_tokens * window_height / float(window_width)))))
        target_width = max(1, min(
            window_width,
            max_tokens // target_height))

        n_window, _, n_channel = windows.shape
        windows = windows.transpose(1, 2).reshape(
            n_window, n_channel, window_height, window_width)
        windows = functional.adaptive_avg_pool2d(
            windows,
            output_size=(target_height, target_width))

        return windows.flatten(2).transpose(1, 2)

    def _local_forward(self, fine_windows, metadata):
        window_height, window_width = metadata[-2:]
        normalized_windows = self.local_norm(fine_windows)
        local_context = self._pool_windows(
            normalized_windows,
            window_height,
            window_width,
            self.max_local_tokens)
        attended, _ = self.local_attention(
            query=normalized_windows,
            key=local_context,
            value=local_context,
            need_weights=False)

        return fine_windows + attended

    def _hierarchical_context(self, partitions, fine_grid):
        context_by_scale = []

        for scale_index, (scale, partition) in enumerate(zip(
                PartitionPyramid.SCALES, partitions)):
            context_grid = fine_grid // scale
            context_windows, metadata = self._to_windows(
                partition, context_grid)
            window_height, window_width = metadata[-2:]
            context_windows = self._pool_windows(
                context_windows,
                window_height,
                window_width,
                self.max_context_tokens)

            n_batch = partition.shape[0]
            n_token = context_windows.shape[1]
            n_channel = context_windows.shape[2]
            context_windows = context_windows.reshape(
                n_batch,
                context_grid,
                context_grid,
                n_token,
                n_channel)

            # A coarse window is shared by each fine window whose image-space
            # support lies inside it. At scale eight, the one coarse window is
            # therefore shared by all 64 fine windows.
            context_windows = context_windows.repeat_interleave(
                scale, dim=1).repeat_interleave(scale, dim=2)
            context_windows = context_windows.reshape(
                n_batch * fine_grid * fine_grid,
                n_token,
                n_channel)
            context_windows = context_windows + \
                self.scale_embedding[scale_index]
            context_by_scale.append(context_windows)

        return torch.cat(context_by_scale, dim=1)

    def forward(self, fine_feature, partitions, fine_grid=8):
        fine_windows, metadata = self._to_windows(
            fine_feature, fine_grid)
        fine_windows = self._local_forward(fine_windows, metadata)

        context = self._hierarchical_context(partitions, fine_grid)
        attended, _ = self.cross_attention(
            query=self.query_norm(fine_windows),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False)
        fine_windows = fine_windows + attended
        fine_windows = fine_windows + self.feedforward(
            self.feedforward_norm(fine_windows))

        return self._from_windows(fine_windows, metadata)


class PartitionAttentionDepthModel(nn.Module):
    '''
    RGB-only dense depth model with local and hierarchical partition attention.

    Dense queries stay at image resolution. Keys and values are adaptively
    pooled within their spatial windows to bound attention memory while keeping
    the correspondence between fine and coarse image regions exact.
    '''

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 partition_mode='parallel',
                 n_channels=64,
                 n_head=4,
                 n_attention_block=1,
                 max_local_tokens=32,
                 max_context_tokens=16):
        super(PartitionAttentionDepthModel, self).__init__()

        if min_predict_depth <= 0.0:
            raise ValueError('min_predict_depth must be positive')
        if max_predict_depth <= min_predict_depth:
            raise ValueError(
                'max_predict_depth must be greater than min_predict_depth')

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.partition_mode = partition_mode
        self.fine_grid = 8

        self.pyramid = PartitionPyramid(
            n_channels=n_channels,
            mode=partition_mode)
        self.position_convolutions = nn.ModuleList([
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                padding=1,
                groups=n_channels)
            for _ in PartitionPyramid.SCALES
        ])
        self.attention_blocks = nn.ModuleList([
            PartitionAttentionBlock(
                n_channels=n_channels,
                n_head=n_head,
                n_scale=len(PartitionPyramid.SCALES),
                max_local_tokens=max_local_tokens,
                max_context_tokens=max_context_tokens)
            for _ in range(n_attention_block)
        ])
        self.depth_head = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_channels // 2, 1, kernel_size=1))

        self.register_buffer(
            'image_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'image_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _pad_image(self, image):
        n_height, n_width = image.shape[-2:]
        n_pad_height = (self.fine_grid - n_height % self.fine_grid) % \
            self.fine_grid
        n_pad_width = (self.fine_grid - n_width % self.fine_grid) % \
            self.fine_grid

        if n_pad_height > 0 or n_pad_width > 0:
            image = functional.pad(
                image,
                pad=(0, n_pad_width, 0, n_pad_height),
                mode='replicate')

        return image

    def extract_partitions(self, image):
        '''Returns position-aware partitions, padding image size if required.'''
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('image must have shape N x 3 x H x W')

        image = self._pad_image(image)
        normalized_image = (image - self.image_mean) / self.image_std
        partitions = self.pyramid(normalized_image)

        return [
            partition + position_convolution(partition)
            for partition, position_convolution in zip(
                partitions, self.position_convolutions)
        ]

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('image must have shape N x 3 x H x W')

        original_shape = image.shape[-2:]
        partitions = self.extract_partitions(image)
        fine_feature = partitions[0]

        for attention_block in self.attention_blocks:
            fine_feature = attention_block(
                fine_feature=fine_feature,
                partitions=partitions,
                fine_grid=self.fine_grid)

        raw_depth = self.depth_head(fine_feature)
        normalized_log_depth = torch.sigmoid(raw_depth)

        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        output_depth = torch.exp(
            log_min_depth +
            normalized_log_depth * (log_max_depth - log_min_depth))

        return output_depth[..., :original_shape[0], :original_shape[1]]
