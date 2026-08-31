import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


EPSILON = 1e-8


def normalization_groups(n_channels):
    n_group = min(8, n_channels)
    while n_channels % n_group != 0:
        n_group = n_group - 1
    return n_group


def pooled_shape(n_height, n_width, max_tokens):
    if n_height * n_width <= max_tokens:
        return n_height, n_width
    target_height = max(1, min(
        n_height,
        int(math.sqrt(max_tokens * n_height / float(n_width)))))
    target_width = max(1, min(
        n_width,
        max_tokens // target_height))
    return target_height, target_width


class PartitionPyramid(nn.Module):
    '''Builds overlapping RGB features at 1, 1/2, 1/4 and 1/8 scale.'''

    SCALES = (1, 2, 4, 8)

    def __init__(self, n_channels=32, mode='parallel'):
        super(PartitionPyramid, self).__init__()

        if mode not in ('parallel', 'sequential'):
            raise ValueError(
                'Unsupported partition pyramid mode: {}'.format(mode))

        self.mode = mode
        n_group = normalization_groups(n_channels)

        if mode == 'parallel':
            # Overlap before striding avoids the hard non-overlapping patch
            # edges produced by kernel_size == stride.
            kernel_sizes = (1, 3, 7, 15)
            convolutions = [
                nn.Conv2d(
                    3,
                    n_channels,
                    kernel_size=kernel_size,
                    stride=scale,
                    padding=scale - 1)
                for scale, kernel_size in zip(
                    self.SCALES, kernel_sizes)
            ]
        else:
            kernel_sizes = (1, 3, 5, 7)
            strides = (1, 2, 2, 2)
            paddings = (0, 1, 2, 3)
            convolutions = []
            for index, (kernel_size, stride, padding) in enumerate(zip(
                    kernel_sizes, strides, paddings)):
                convolutions.append(nn.Conv2d(
                    3 if index == 0 else n_channels,
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
            x = self.activation(normalization(convolution(
                image if self.mode == 'parallel' else x)))
            outputs.append(x)
        return outputs


class SpatialResidualBlock(nn.Module):
    '''Grid-free spatial refinement that never changes feature resolution.'''

    def __init__(self, n_channels, kernel_size=5):
        super(SpatialResidualBlock, self).__init__()

        self.gate = nn.Parameter(torch.tensor(0.1))
        self.block = nn.Sequential(
            nn.GroupNorm(normalization_groups(n_channels), n_channels),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=n_channels),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def forward(self, feature):
        return feature + torch.tanh(self.gate) * self.block(feature)


class FastFineToCoarseBlock(nn.Module):
    '''Smooth convolutional bottom-up communication between adjacent scales.'''

    def __init__(self, n_channels):
        super(FastFineToCoarseBlock, self).__init__()

        n_group = normalization_groups(n_channels)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.downsample = nn.Sequential(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=n_channels,
                bias=False),
            nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
            nn.GroupNorm(n_group, n_channels),
            nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * n_channels, n_channels, kernel_size=1),
            nn.GroupNorm(n_group, n_channels),
            nn.GELU(),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                padding=1,
                groups=n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def forward(self, fine_feature, coarse_feature):
        downsampled = self.downsample(fine_feature)
        if downsampled.shape[-2:] != coarse_feature.shape[-2:]:
            downsampled = functional.interpolate(
                downsampled,
                size=coarse_feature.shape[-2:],
                mode='bilinear',
                align_corners=False)
        update = self.fusion(torch.cat([
            coarse_feature,
            downsampled
        ], dim=1))
        return coarse_feature + torch.tanh(self.gate) * update


class GlobalPartitionAttentionBlock(nn.Module):
    '''Global communication at 1/8 scale with bounded context tokens.'''

    def __init__(self, n_channels, n_head, max_global_tokens=128):
        super(GlobalPartitionAttentionBlock, self).__init__()

        self.max_global_tokens = max_global_tokens
        self.query_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            batch_first=True)
        self.attention_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_gate = nn.Parameter(torch.tensor(0.1))
        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 2 * n_channels),
            nn.GELU(),
            nn.Linear(2 * n_channels, n_channels))

    def forward(self, feature):
        n_batch, n_channel, n_height, n_width = feature.shape
        queries = feature.flatten(2).transpose(1, 2)
        context_height, context_width = pooled_shape(
            n_height, n_width, self.max_global_tokens)
        context = functional.adaptive_avg_pool2d(
            feature,
            output_size=(context_height, context_width))
        context = context.flatten(2).transpose(1, 2)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            query=self.query_norm(queries),
            key=normalized_context,
            value=normalized_context,
            need_weights=False)
        queries = queries + torch.tanh(self.attention_gate) * attended
        queries = queries + torch.tanh(self.feedforward_gate) * \
            self.feedforward(self.feedforward_norm(queries))
        return queries.transpose(1, 2).reshape(
            n_batch, n_channel, n_height, n_width)


class SparseDepthTokenEncoder(nn.Module):
    '''Encodes every valid sparse measurement as one metric token.'''

    def __init__(self, n_channels, min_predict_depth, max_predict_depth):
        super(SparseDepthTokenEncoder, self).__init__()

        self.log_min_depth = math.log(min_predict_depth)
        self.log_depth_range = \
            math.log(max_predict_depth) - self.log_min_depth
        self.image_projection = nn.Linear(n_channels, n_channels)
        self.depth_embedding = nn.Sequential(
            nn.Linear(1, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels))
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(6, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels))
        self.output_norm = nn.LayerNorm(n_channels)

    @staticmethod
    def coordinate_features(x, y):
        return torch.stack([
            x,
            y,
            torch.sin(math.pi * x),
            torch.cos(math.pi * x),
            torch.sin(math.pi * y),
            torch.cos(math.pi * y)
        ], dim=-1)

    def forward(self, image_feature, sparse_depth, validity_map):
        n_batch, n_channel, n_height, n_width = image_feature.shape
        valid = torch.logical_and(
            validity_map > 0.0,
            torch.logical_and(
                torch.isfinite(sparse_depth),
                sparse_depth > 0.0))
        n_valid = valid.flatten(1).sum(dim=1)
        max_token = max(1, int(torch.max(n_valid).item()))

        tokens = image_feature.new_zeros((
            n_batch, max_token, n_channel))
        padding_mask = torch.ones(
            (n_batch, max_token),
            dtype=torch.bool,
            device=image_feature.device)
        image_tokens = image_feature.flatten(2).transpose(1, 2)
        sparse_values = sparse_depth.flatten(1)

        for batch_index in range(n_batch):
            indices = torch.nonzero(
                valid[batch_index].flatten(),
                as_tuple=False).flatten()
            n_token = indices.numel()
            if n_token == 0:
                padding_mask[batch_index, 0] = False
                continue

            y = indices // n_width
            x = indices % n_width
            normalized_x = 2.0 * x.to(image_feature.dtype) / \
                max(1, n_width - 1) - 1.0
            normalized_y = 2.0 * y.to(image_feature.dtype) / \
                max(1, n_height - 1) - 1.0
            depth = sparse_values[batch_index, indices]
            normalized_log_depth = (
                torch.log(torch.clamp(depth, min=EPSILON)) -
                self.log_min_depth) / self.log_depth_range
            token = self.image_projection(
                image_tokens[batch_index, indices])
            token = token + self.depth_embedding(
                normalized_log_depth.unsqueeze(-1))
            token = token + self.coordinate_embedding(
                self.coordinate_features(normalized_x, normalized_y))
            tokens[batch_index, :n_token] = self.output_norm(token)
            padding_mask[batch_index, :n_token] = False

        return tokens, padding_mask, n_valid > 0


class SparseMetricAttentionBlock(nn.Module):
    '''Lets pooled image context attend to all sparse metric tokens.'''

    def __init__(self, n_channels, n_head, max_metric_queries=128):
        super(SparseMetricAttentionBlock, self).__init__()

        self.max_metric_queries = max_metric_queries
        self.query_norm = nn.LayerNorm(n_channels)
        self.token_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            batch_first=True)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(6, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels))
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.refinement = nn.Sequential(
            nn.GroupNorm(normalization_groups(n_channels), n_channels),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                padding=1,
                groups=n_channels),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def _query_coordinates(self, n_height, n_width, feature):
        y = torch.linspace(
            -1.0, 1.0, n_height,
            dtype=feature.dtype,
            device=feature.device)
        x = torch.linspace(
            -1.0, 1.0, n_width,
            dtype=feature.dtype,
            device=feature.device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        coordinates = SparseDepthTokenEncoder.coordinate_features(
            x.flatten(), y.flatten())
        return self.coordinate_embedding(coordinates).unsqueeze(0)

    def forward(self, feature, sparse_tokens, padding_mask, has_valid):
        n_batch, _, n_height, n_width = feature.shape
        query_height, query_width = pooled_shape(
            n_height, n_width, self.max_metric_queries)
        metric_feature = functional.adaptive_avg_pool2d(
            feature,
            output_size=(query_height, query_width))
        queries = metric_feature.flatten(2).transpose(1, 2)
        queries = queries + self._query_coordinates(
            query_height, query_width, feature)
        normalized_tokens = self.token_norm(sparse_tokens)
        attended, _ = self.attention(
            query=self.query_norm(queries),
            key=normalized_tokens,
            value=normalized_tokens,
            key_padding_mask=padding_mask,
            need_weights=False)
        attended = attended.transpose(1, 2).reshape(
            n_batch, -1, query_height, query_width)
        attended = functional.interpolate(
            attended,
            size=(n_height, n_width),
            mode='bilinear',
            align_corners=False)
        update = self.refinement(attended)
        update = update * has_valid.to(feature.dtype).view(
            n_batch, 1, 1, 1)
        return feature + torch.tanh(self.gate) * update


class FastCoarseToFineBlock(nn.Module):
    '''Grid-free top-down fusion using bilinear interpolation and convolution.'''

    def __init__(self, n_channels):
        super(FastCoarseToFineBlock, self).__init__()

        n_group = normalization_groups(n_channels)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * n_channels, n_channels, kernel_size=1),
            nn.GroupNorm(n_group, n_channels),
            nn.GELU(),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=5,
                padding=2,
                groups=n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=1))

    def forward(self, fine_feature, coarse_feature):
        coarse_feature = functional.interpolate(
            coarse_feature,
            size=fine_feature.shape[-2:],
            mode='bilinear',
            align_corners=False)
        update = self.fusion(torch.cat([
            fine_feature,
            coarse_feature
        ], dim=1))
        return fine_feature + torch.tanh(self.gate) * update


class PartitionAttentionDepthModel(nn.Module):
    '''Fast dense depth model with global RGB context and sparse metric tokens.'''

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 partition_mode='parallel',
                 n_channels=32,
                 n_head=4,
                 max_global_tokens=128,
                 max_metric_queries=128):
        super(PartitionAttentionDepthModel, self).__init__()

        if min_predict_depth <= 0.0:
            raise ValueError('min_predict_depth must be positive')
        if max_predict_depth <= min_predict_depth:
            raise ValueError(
                'max_predict_depth must be greater than min_predict_depth')
        if n_channels % n_head != 0:
            raise ValueError('n_channels must be divisible by n_head')

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.partition_mode = partition_mode
        self.n_scale = 8

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
        self.detail_refinement = SpatialResidualBlock(
            n_channels=n_channels,
            kernel_size=7)
        self.fine_to_coarse_blocks = nn.ModuleList([
            FastFineToCoarseBlock(n_channels=n_channels)
            for _ in range(len(PartitionPyramid.SCALES) - 1)
        ])
        self.sparse_token_encoder = SparseDepthTokenEncoder(
            n_channels=n_channels,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth)
        self.sparse_metric_attention = SparseMetricAttentionBlock(
            n_channels=n_channels,
            n_head=n_head,
            max_metric_queries=max_metric_queries)
        self.global_attention = GlobalPartitionAttentionBlock(
            n_channels=n_channels,
            n_head=n_head,
            max_global_tokens=max_global_tokens)
        self.coarse_to_fine_blocks = nn.ModuleList([
            FastCoarseToFineBlock(n_channels=n_channels)
            for _ in range(len(PartitionPyramid.SCALES) - 1)
        ])
        self.output_refinement = SpatialResidualBlock(
            n_channels=n_channels,
            kernel_size=7)
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

    def _pad(self, tensor, mode='replicate'):
        n_height, n_width = tensor.shape[-2:]
        n_pad_height = (self.n_scale - n_height % self.n_scale) % \
            self.n_scale
        n_pad_width = (self.n_scale - n_width % self.n_scale) % \
            self.n_scale
        if n_pad_height > 0 or n_pad_width > 0:
            tensor = functional.pad(
                tensor,
                pad=(0, n_pad_width, 0, n_pad_height),
                mode=mode)
        return tensor

    def extract_partitions(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('image must have shape N x 3 x H x W')

        image = self._pad(image)
        normalized_image = (image - self.image_mean) / self.image_std
        partitions = self.pyramid(normalized_image)
        return [
            partition + position_convolution(partition)
            for partition, position_convolution in zip(
                partitions, self.position_convolutions)
        ]

    def _prepare_sparse_depth(self, image, sparse_depth, validity_map):
        if sparse_depth is None:
            sparse_depth = image.new_zeros((
                image.shape[0], 1, image.shape[-2], image.shape[-1]))
        if sparse_depth.ndim != 4 or sparse_depth.shape[1] != 1 or \
                sparse_depth.shape[0] != image.shape[0] or \
                sparse_depth.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                'sparse_depth must have shape N x 1 x H x W')
        if validity_map is None:
            validity_map = (sparse_depth > 0.0).to(sparse_depth.dtype)
        if validity_map.shape != sparse_depth.shape:
            raise ValueError('validity_map must match sparse_depth shape')

        sparse_depth = self._pad(sparse_depth, mode='constant')
        validity_map = self._pad(validity_map, mode='constant')
        validity_map = torch.logical_and(
            validity_map > 0.0,
            torch.logical_and(
                torch.isfinite(sparse_depth),
                sparse_depth > 0.0)).to(sparse_depth.dtype)
        sparse_depth = torch.where(
            validity_map > 0.0,
            torch.clamp(
                sparse_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.zeros_like(sparse_depth))
        return sparse_depth, validity_map

    def _apply_metric_constraints(
            self, prior_depth, sparse_depth, validity_map):
        n_valid = torch.sum(validity_map, dim=[1, 2, 3], keepdim=True)
        log_scale = torch.sum(
            validity_map * (
                torch.log(torch.clamp(sparse_depth, min=EPSILON)) -
                torch.log(torch.clamp(prior_depth, min=EPSILON))),
            dim=[1, 2, 3],
            keepdim=True) / (n_valid + EPSILON)
        log_scale = torch.where(
            n_valid > 0.0,
            log_scale,
            torch.zeros_like(log_scale))
        output_depth = torch.clamp(
            prior_depth * torch.exp(log_scale),
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return validity_map * sparse_depth + \
            (1.0 - validity_map) * output_depth

    def forward(self, image, sparse_depth=None, validity_map=None):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('image must have shape N x 3 x H x W')

        original_shape = image.shape[-2:]
        sparse_depth, validity_map = self._prepare_sparse_depth(
            image=image,
            sparse_depth=sparse_depth,
            validity_map=validity_map)
        partitions = self.extract_partitions(image)

        # This full-resolution identity branch is never spatially compressed.
        fine_feature = self.detail_refinement(partitions[0])
        hierarchy = [fine_feature]
        for block, coarse_partition in zip(
                self.fine_to_coarse_blocks, partitions[1:]):
            hierarchy.append(block(
                fine_feature=hierarchy[-1],
                coarse_feature=coarse_partition))

        sparse_tokens, padding_mask, has_valid = self.sparse_token_encoder(
            image_feature=fine_feature,
            sparse_depth=sparse_depth,
            validity_map=validity_map)
        hierarchy[-1] = self.sparse_metric_attention(
            feature=hierarchy[-1],
            sparse_tokens=sparse_tokens,
            padding_mask=padding_mask,
            has_valid=has_valid)
        hierarchy[-1] = self.global_attention(hierarchy[-1])

        coarse_feature = hierarchy[-1]
        for block, level in zip(
                self.coarse_to_fine_blocks,
                range(len(hierarchy) - 2, -1, -1)):
            coarse_feature = block(
                fine_feature=hierarchy[level],
                coarse_feature=coarse_feature)
        fine_feature = self.output_refinement(coarse_feature)

        raw_depth = self.depth_head(fine_feature)
        normalized_log_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        prior_depth = torch.exp(
            log_min_depth +
            normalized_log_depth * (log_max_depth - log_min_depth))
        output_depth = self._apply_metric_constraints(
            prior_depth=prior_depth,
            sparse_depth=sparse_depth,
            validity_map=validity_map)
        return output_depth[..., :original_shape[0], :original_shape[1]]
