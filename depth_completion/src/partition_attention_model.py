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


class LocalPartitionAttentionBlock(nn.Module):
    '''Local fine-window attention with full-resolution boundary refinement.'''

    def __init__(self,
                 n_channels,
                 n_head,
                 max_local_tokens=32,
                 dropout=0.0):
        super(LocalPartitionAttentionBlock, self).__init__()

        self.max_local_tokens = max_local_tokens
        self.local_norm = nn.LayerNorm(n_channels)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.refinement_gate = nn.Parameter(torch.tensor(0.1))
        self.boundary_refinement = nn.Sequential(
            nn.GroupNorm(1, n_channels),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=7,
                padding=3,
                groups=n_channels),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

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

    def forward(self, fine_feature, fine_grid=8):
        fine_windows, metadata = self._to_windows(
            fine_feature, fine_grid)
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
        attended = self._from_windows(attended, metadata)
        fine_feature = fine_feature + \
            torch.tanh(self.attention_gate) * attended

        # This convolution is evaluated after windows are reassembled, so it
        # explicitly mixes features on opposite sides of every grid boundary.
        return fine_feature + torch.tanh(self.refinement_gate) * \
            self.boundary_refinement(fine_feature)


class FineToCoarseAttentionBlock(nn.Module):
    '''Updates each coarse token from its corresponding 2 x 2 finer region.'''

    def __init__(self, n_channels, n_head, dropout=0.0):
        super(FineToCoarseAttentionBlock, self).__init__()

        self.query_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_channels, n_channels),
            nn.Dropout(dropout))

    def forward(self, fine_feature, coarse_feature):
        n_batch, n_channel, fine_height, fine_width = fine_feature.shape
        coarse_height, coarse_width = coarse_feature.shape[-2:]

        if fine_height != 2 * coarse_height or \
                fine_width != 2 * coarse_width:
            raise ValueError(
                'Fine and coarse features must differ by exactly 2x')

        # Each coarse query reads the four finer tokens with the same image
        # support. Repeating this at every level forms the bottom-up path.
        fine_context = functional.unfold(
            fine_feature,
            kernel_size=2,
            stride=2)
        fine_context = fine_context.reshape(
            n_batch, n_channel, 4, coarse_height * coarse_width)
        fine_context = fine_context.permute(0, 3, 2, 1).reshape(
            n_batch * coarse_height * coarse_width, 4, n_channel)
        coarse_queries = coarse_feature.flatten(2).transpose(1, 2).reshape(
            n_batch * coarse_height * coarse_width, 1, n_channel)

        attended, _ = self.attention(
            query=self.query_norm(coarse_queries),
            key=self.context_norm(fine_context),
            value=self.context_norm(fine_context),
            need_weights=False)
        coarse_queries = coarse_queries + \
            torch.tanh(self.attention_gate) * attended
        coarse_queries = coarse_queries + \
            torch.tanh(self.feedforward_gate) * self.feedforward(
                self.feedforward_norm(coarse_queries))

        return coarse_queries.reshape(
            n_batch,
            coarse_height * coarse_width,
            n_channel).transpose(1, 2).reshape(
                n_batch, n_channel, coarse_height, coarse_width)


class GlobalPartitionAttentionBlock(nn.Module):
    '''Global attention over the complete 1/8-resolution feature map.'''

    def __init__(self,
                 n_channels,
                 n_head,
                 max_global_tokens=256,
                 dropout=0.0):
        super(GlobalPartitionAttentionBlock, self).__init__()

        self.max_global_tokens = max_global_tokens
        self.query_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_channels, n_channels),
            nn.Dropout(dropout))

    def _global_context(self, feature):
        n_height, n_width = feature.shape[-2:]
        if n_height * n_width <= self.max_global_tokens:
            return feature.flatten(2).transpose(1, 2)

        target_height = max(1, min(
            n_height,
            int(math.sqrt(
                self.max_global_tokens * n_height / float(n_width)))))
        target_width = max(1, min(
            n_width,
            self.max_global_tokens // target_height))
        context = functional.adaptive_avg_pool2d(
            feature,
            output_size=(target_height, target_width))

        return context.flatten(2).transpose(1, 2)

    def forward(self, feature):
        n_batch, n_channel, n_height, n_width = feature.shape
        queries = feature.flatten(2).transpose(1, 2)
        context = self._global_context(feature)
        attended, _ = self.attention(
            query=self.query_norm(queries),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False)
        queries = queries + torch.tanh(self.attention_gate) * attended
        queries = queries + torch.tanh(self.feedforward_gate) * \
            self.feedforward(self.feedforward_norm(queries))

        return queries.transpose(1, 2).reshape(
            n_batch, n_channel, n_height, n_width)


class CoarseToFineAttentionBlock(nn.Module):
    '''Top-down attention using overlapping 3 x 3 coarse neighborhoods.'''

    def __init__(self,
                 n_channels,
                 n_head,
                 max_groups_per_chunk=4096,
                 dropout=0.0):
        super(CoarseToFineAttentionBlock, self).__init__()

        self.max_groups_per_chunk = max_groups_per_chunk
        self.query_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.refinement_gate = nn.Parameter(torch.tensor(0.1))
        self.context_refinement = nn.Sequential(
            nn.GroupNorm(1, n_channels),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                padding=1,
                groups=n_channels),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def forward(self, fine_feature, coarse_feature):
        n_batch, n_channel, fine_height, fine_width = fine_feature.shape
        coarse_height, coarse_width = coarse_feature.shape[-2:]

        if fine_height != 2 * coarse_height or \
                fine_width != 2 * coarse_width:
            raise ValueError(
                'Fine and coarse features must differ by exactly 2x')

        # Four fine queries share a coarse center, but each reads an overlapping
        # 3 x 3 neighborhood. Adjacent groups therefore share most context
        # instead of receiving a blockwise-constant replicated feature.
        fine_queries = fine_feature.reshape(
            n_batch,
            n_channel,
            coarse_height,
            2,
            coarse_width,
            2)
        fine_queries = fine_queries.permute(0, 2, 4, 3, 5, 1).reshape(
            n_batch, coarse_height * coarse_width, 4, n_channel)

        # Construct and attend to neighborhoods in chunks. This avoids
        # materializing a full H/2 x W/2 x 9 x C unfolded tensor at the
        # highest-resolution top-down stage.
        padded_coarse = functional.pad(
            coarse_feature,
            pad=(1, 1, 1, 1),
            mode='replicate').permute(0, 2, 3, 1)
        offset_y = torch.tensor(
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            device=coarse_feature.device)
        offset_x = torch.tensor(
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            device=coarse_feature.device)
        n_group = coarse_height * coarse_width
        attended_chunks = []

        for start in range(0, n_group, self.max_groups_per_chunk):
            end = min(start + self.max_groups_per_chunk, n_group)
            position = torch.arange(start, end, device=coarse_feature.device)
            center_y = position // coarse_width + 1
            center_x = position % coarse_width + 1
            neighbor_y = center_y.unsqueeze(1) + offset_y.unsqueeze(0)
            neighbor_x = center_x.unsqueeze(1) + offset_x.unsqueeze(0)
            coarse_context = padded_coarse[
                :, neighbor_y, neighbor_x, :]

            query_chunk = fine_queries[:, start:end, ...]
            n_chunk = end - start
            query_chunk = query_chunk.reshape(
                n_batch * n_chunk, 4, n_channel)
            coarse_context = coarse_context.reshape(
                n_batch * n_chunk, 9, n_channel)
            normalized_context = self.context_norm(coarse_context)
            attended, _ = self.attention(
                query=self.query_norm(query_chunk),
                key=normalized_context,
                value=normalized_context,
                need_weights=False)
            attended_chunks.append(attended.reshape(
                n_batch, n_chunk, 4, n_channel))

        attended = torch.cat(attended_chunks, dim=1).reshape(
            n_batch,
            coarse_height,
            coarse_width,
            2,
            2,
            n_channel)
        attended = attended.permute(0, 5, 1, 3, 2, 4).reshape(
            n_batch, n_channel, fine_height, fine_width)

        smooth_context = functional.interpolate(
            coarse_feature,
            size=(fine_height, fine_width),
            mode='bilinear',
            align_corners=False)
        context_update = self.context_refinement(
            attended + smooth_context)

        # The original-resolution feature is always the identity path. Coarse
        # context is an explicitly gated residual, never a replacement.
        return fine_feature + torch.tanh(self.attention_gate) * attended + \
            torch.tanh(self.refinement_gate) * context_update


class PartitionAttentionDepthModel(nn.Module):
    '''
    RGB-only dense depth model with local and hierarchical partition attention.

    A full-resolution identity path retains spatial detail. A bidirectional
    attention pyramid carries information down to a globally communicating
    1/8 map and back through overlapping top-down neighborhoods. Global keys
    and values are pooled to bound memory without compressing the output path.
    '''

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 partition_mode='parallel',
                 n_channels=64,
                 n_head=4,
                 n_attention_block=1,
                 max_local_tokens=32,
                 max_global_tokens=256):
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
        self.local_attention_blocks = nn.ModuleList([
            LocalPartitionAttentionBlock(
                n_channels=n_channels,
                n_head=n_head,
                max_local_tokens=max_local_tokens)
            for _ in range(n_attention_block)
        ])
        self.fine_to_coarse_blocks = nn.ModuleList([
            FineToCoarseAttentionBlock(
                n_channels=n_channels,
                n_head=n_head)
            for _ in range(len(PartitionPyramid.SCALES) - 1)
        ])
        self.global_attention = GlobalPartitionAttentionBlock(
            n_channels=n_channels,
            n_head=n_head,
            max_global_tokens=max_global_tokens)
        self.coarse_to_fine_blocks = nn.ModuleList([
            CoarseToFineAttentionBlock(
                n_channels=n_channels,
                n_head=n_head)
            for _ in range(len(PartitionPyramid.SCALES) - 1)
        ])
        self.detail_refinement = nn.Sequential(
            nn.GroupNorm(1, n_channels),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=5,
                padding=2,
                groups=n_channels),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))
        self.detail_refinement_gate = nn.Parameter(torch.tensor(0.1))
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

        # Retain the original-resolution feature as an identity detail path.
        fine_feature = partitions[0]
        for local_attention_block in self.local_attention_blocks:
            fine_feature = local_attention_block(
                fine_feature=fine_feature,
                fine_grid=self.fine_grid)

        # Bottom-up communication: every coarser feature is updated from the
        # already-updated finer representation below it.
        hierarchy = [fine_feature]
        for fine_to_coarse_block, coarse_partition in zip(
                self.fine_to_coarse_blocks, partitions[1:]):
            hierarchy.append(fine_to_coarse_block(
                fine_feature=hierarchy[-1],
                coarse_feature=coarse_partition))

        # The entire 1/8 map communicates globally before information returns
        # through the top-down path.
        hierarchy[-1] = self.global_attention(hierarchy[-1])

        # Top-down communication uses overlapping neighborhoods plus bilinear
        # context, avoiding the previous blockwise repeat_interleave mapping.
        coarse_feature = hierarchy[-1]
        for coarse_to_fine_block, level in zip(
                self.coarse_to_fine_blocks,
                range(len(hierarchy) - 2, -1, -1)):
            coarse_feature = coarse_to_fine_block(
                fine_feature=hierarchy[level],
                coarse_feature=coarse_feature)

        fine_feature = coarse_feature + \
            torch.tanh(self.detail_refinement_gate) * \
            self.detail_refinement(coarse_feature)

        raw_depth = self.depth_head(fine_feature)
        normalized_log_depth = torch.sigmoid(raw_depth)

        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        output_depth = torch.exp(
            log_min_depth +
            normalized_log_depth * (log_max_depth - log_min_depth))

        return output_depth[..., :original_shape[0], :original_shape[1]]
