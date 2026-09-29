import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(
    'external_src', 'depth_completion', 'kbnet', 'src'))
from net_utils import Conv2d


class AttentionUpdate(nn.Module):
    '''Update query tokens from context tokens with multi-head attention.'''

    def __init__(self, n_channels, n_head):
        super(AttentionUpdate, self).__init__()

        self.n_head = n_head
        self.head_channels = n_channels // n_head

        self.query_projection = nn.Linear(n_channels, n_channels)
        self.key_projection = nn.Linear(n_channels, n_channels)
        self.value_projection = nn.Linear(n_channels, n_channels)
        self.output_projection = nn.Linear(n_channels, n_channels)

        self.attention_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Linear(4 * n_channels, n_channels))
        self.feedforward_norm = nn.LayerNorm(n_channels)

    def forward(self, query, context):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

        residual = query
        query = self.attention_norm(query)
        context = self.context_norm(context)

        projected_query = self.query_projection(query).reshape(
            n_batch,
            n_query,
            self.n_head,
            self.head_channels).permute(0, 2, 1, 3)
        projected_key = self.key_projection(context).reshape(
            n_batch,
            n_context,
            self.n_head,
            self.head_channels).permute(0, 2, 1, 3)
        projected_value = self.value_projection(context).reshape(
            n_batch,
            n_context,
            self.n_head,
            self.head_channels).permute(0, 2, 1, 3)

        similarity = torch.matmul(
            projected_query,
            projected_key.transpose(-2, -1))
        similarity = similarity / math.sqrt(self.head_channels)
        attention_weights = torch.softmax(similarity, dim=-1)

        attended = torch.matmul(attention_weights, projected_value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            n_batch,
            n_query,
            self.n_head * self.head_channels)
        attended = self.output_projection(attended)

        attended = residual + attended
        feedforward = self.feedforward(
            self.feedforward_norm(attended))
        return attended + feedforward


def add_position_encoding(feature):
    '''Add fixed global 2D Fourier positions to a spatial feature map.'''
    _, n_channel, n_height, n_width = feature.shape
    n_frequency = n_channel // 4

    y = (torch.arange(
        n_height,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_height
    x = (torch.arange(
        n_width,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_width

    frequencies = 2.0 ** torch.arange(
        n_frequency,
        dtype=feature.dtype,
        device=feature.device)
    y_angle = 2.0 * math.pi * y[:, None] * frequencies[None, :]
    x_angle = 2.0 * math.pi * x[:, None] * frequencies[None, :]

    position = torch.cat([
        torch.sin(y_angle)[:, None, :].expand(
            n_height, n_width, n_frequency),
        torch.cos(y_angle)[:, None, :].expand(
            n_height, n_width, n_frequency),
        torch.sin(x_angle)[None, :, :].expand(
            n_height, n_width, n_frequency),
        torch.cos(x_angle)[None, :, :].expand(
            n_height, n_width, n_frequency)
    ], dim=2)
    position = position.permute(2, 0, 1)[None, :, :, :]
    return feature + position


def image_to_patch_vectors(image, patch_size):
    '''Flatten every non-overlapping patch without discarding input pixels.'''
    n_batch, n_channel, n_height, n_width = image.shape
    n_patch_height = n_height // patch_size
    n_patch_width = n_width // patch_size

    patches = image.reshape(
        n_batch,
        n_channel,
        n_patch_height,
        patch_size,
        n_patch_width,
        patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(
        n_batch,
        n_patch_height,
        n_patch_width,
        patch_size * patch_size * n_channel)


def tokens_to_partitions(tokens, n_partition):
    '''Group the global patch-token grid into a fixed partition grid.'''
    n_batch, n_height, n_width, n_channel = tokens.shape
    partition_height = n_height // n_partition
    partition_width = n_width // n_partition

    partitions = tokens.reshape(
        n_batch,
        n_partition,
        partition_height,
        n_partition,
        partition_width,
        n_channel)
    partitions = partitions.permute(0, 1, 3, 2, 4, 5)
    return partitions.reshape(
        n_batch * n_partition * n_partition,
        partition_height * partition_width,
        n_channel)


def partitions_to_tokens(partitions, n_batch, n_partition, partition_side):
    '''Restore partition tokens to one continuous global token grid.'''
    n_channel = partitions.shape[-1]
    tokens = partitions.reshape(
        n_batch,
        n_partition,
        n_partition,
        partition_side,
        partition_side,
        n_channel)
    tokens = tokens.permute(0, 1, 3, 2, 4, 5)
    return tokens.reshape(
        n_batch,
        n_partition * partition_side,
        n_partition * partition_side,
        n_channel)


class PartitionAttentionPatchTokenDepthModel(nn.Module):
    '''
    Patch-token ablation with attention-only hierarchical processing. Patch
    tokens communicate locally inside fixed partitions, travel fine-to-coarse,
    communicate globally through one bottom token per partition, and travel
    coarse-to-fine. Full-resolution RGB features finally query nearby updated
    patch tokens before a three-convolution depth head.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 patch_size=16,
                 partition_size=128,
                 n_partition=4):
        super(PartitionAttentionPatchTokenDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.partition_size = partition_size
        self.n_partition = n_partition

        token_side = partition_size // patch_size
        self.level_token_sides = []
        while token_side >= 1:
            self.level_token_sides.append(token_side)
            token_side = token_side // 2
        self.n_level = len(self.level_token_sides)

        # The patch projection is the DPT-style initial tokenization. Every
        # 16x16 RGB patch is flattened and projected to one C-dimensional token.
        self.patch_projection = nn.Linear(
            patch_size * patch_size * 3,
            n_channels)

        # Preserve full-resolution RGB detail for the final pixel queries. This
        # is the only convolution before the attention hierarchy.
        self.pixel_embedding = Conv2d(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=3,
            stride=1)

        # Local self-attention is applied at every non-bottom level inside each
        # partition independently.
        self.local_attention_blocks = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Each learned spatial query summarizes one aligned 2x2 group of fine
        # tokens into one coarse token.
        self.pooling_queries = nn.ParameterList([
            nn.Parameter(torch.randn(
                1,
                side * side,
                n_channels) * 0.02)
            for side in self.level_token_sides[1:]
        ])
        self.fine_to_coarse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # The bottom contains one token for each of the 4x4 partitions, so this
        # attention operation provides whole-image communication.
        self.bottom_global_attention = AttentionUpdate(
            n_channels,
            n_head)

        # Retained fine tokens query all coarse tokens from the same partition
        # while information travels back toward the finest patch-token level.
        self.coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Full-resolution pixels query a 3x3 neighborhood in the final global
        # patch-token grid. This crosses partition boundaries without requiring
        # full-resolution pixel-to-pixel self-attention.
        self.pixel_from_patch_attention = AttentionUpdate(
            n_channels,
            n_head)

        n_patch_side = n_partition * self.level_token_sides[0]
        patch_indices = torch.arange(
            n_patch_side * n_patch_side).reshape(
                n_patch_side,
                n_patch_side)
        neighbors = []
        for offset_y in [-1, 0, 1]:
            for offset_x in [-1, 0, 1]:
                y = torch.arange(n_patch_side) + offset_y
                x = torch.arange(n_patch_side) + offset_x
                y = torch.clamp(y, min=0, max=n_patch_side - 1)
                x = torch.clamp(x, min=0, max=n_patch_side - 1)
                neighbors.append(patch_indices[y[:, None], x[None, :]])
        self.register_buffer(
            'neighbor_indices',
            torch.stack(neighbors, dim=-1).reshape(-1, 9))

        # The requested three spatial convolutions decode the attended
        # full-resolution feature. The first two use Conv2d's activation and
        # the last produces the unconstrained raw depth value.
        self.depth_output = nn.Sequential(
            Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1),
            Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1),
            Conv2d(
                n_channels,
                1,
                kernel_size=3,
                stride=1,
                activation_func=None))

    def attention_pool(self, fine, coarse_query, attention_block, coarse_side):
        '''Summarize every aligned 2x2 fine-token group with attention.'''
        n_partition_batch, _, n_channel = fine.shape
        fine_side = 2 * coarse_side

        context = fine.reshape(
            n_partition_batch,
            coarse_side,
            2,
            coarse_side,
            2,
            n_channel)
        context = context.permute(0, 1, 3, 2, 4, 5).reshape(
            n_partition_batch * coarse_side * coarse_side,
            4,
            n_channel)

        query = coarse_query.expand(
            n_partition_batch,
            -1,
            -1).reshape(
                n_partition_batch * coarse_side * coarse_side,
                1,
                n_channel)
        coarse = attention_block(query, context)
        return coarse.reshape(
            n_partition_batch,
            coarse_side * coarse_side,
            n_channel)

    def pixel_queries(self, feature):
        '''Group full-resolution features by their corresponding image patch.'''
        n_batch, n_channel, n_height, n_width = feature.shape
        n_patch_height = n_height // self.patch_size
        n_patch_width = n_width // self.patch_size

        queries = feature.reshape(
            n_batch,
            n_channel,
            n_patch_height,
            self.patch_size,
            n_patch_width,
            self.patch_size)
        queries = queries.permute(0, 2, 4, 3, 5, 1)
        return queries.reshape(
            n_batch,
            n_patch_height * n_patch_width,
            self.patch_size * self.patch_size,
            n_channel)

    def queries_to_feature(self, queries, n_height, n_width):
        '''Restore patch-grouped pixel queries to a full-resolution feature.'''
        n_batch, _, _, n_channel = queries.shape
        n_patch_height = n_height // self.patch_size
        n_patch_width = n_width // self.patch_size

        feature = queries.reshape(
            n_batch,
            n_patch_height,
            n_patch_width,
            self.patch_size,
            self.patch_size,
            n_channel)
        feature = feature.permute(0, 5, 1, 3, 2, 4)
        return feature.reshape(
            n_batch,
            n_channel,
            n_height,
            n_width)

    def forward(self, image):
        n_batch, _, n_height, n_width = image.shape

        # Build one token from every non-overlapping RGB patch, then add global
        # position before imposing the 4x4 partition grid.
        patch_vectors = image_to_patch_vectors(
            image,
            self.patch_size)
        patch_tokens = self.patch_projection(patch_vectors)
        patch_feature = patch_tokens.permute(0, 3, 1, 2)
        patch_feature = add_position_encoding(patch_feature)
        patch_tokens = patch_feature.permute(0, 2, 3, 1)
        partitions = tokens_to_partitions(
            patch_tokens,
            self.n_partition)

        # Fine-to-coarse traversal. The fine representation from every level
        # is retained so it can later query the globally informed coarse path.
        levels = []
        fine = partitions
        for level in range(self.n_level - 1):
            fine = self.local_attention_blocks[level](fine, fine)
            levels.append(fine)
            fine = self.attention_pool(
                fine=fine,
                coarse_query=self.pooling_queries[level],
                attention_block=self.fine_to_coarse_attention[level],
                coarse_side=self.level_token_sides[level + 1])
        levels.append(fine)

        # The 16 bottom partition tokens attend globally to one another.
        bottom = levels[-1].reshape(
            n_batch,
            self.n_partition * self.n_partition,
            self.n_channels)
        bottom = self.bottom_global_attention(bottom, bottom)
        levels[-1] = bottom.reshape(
            n_batch * self.n_partition * self.n_partition,
            1,
            self.n_channels)

        # Coarse-to-fine traversal updates each retained fine level from all
        # spatial coarse tokens belonging to its corresponding partition.
        for level in range(self.n_level - 2, -1, -1):
            levels[level] = self.coarse_to_fine_attention[level](
                levels[level],
                levels[level + 1])

        # Restore one continuous 32x32 patch grid. Every pixel query receives
        # context from its parent patch and the eight neighboring patch tokens.
        patch_tokens = partitions_to_tokens(
            levels[0],
            n_batch=n_batch,
            n_partition=self.n_partition,
            partition_side=self.level_token_sides[0])
        flat_patch_tokens = patch_tokens.reshape(
            n_batch,
            -1,
            self.n_channels)
        patch_context = flat_patch_tokens[:, self.neighbor_indices, :]

        pixel_feature = self.pixel_embedding(image)
        pixel_feature = add_position_encoding(pixel_feature)
        pixel_queries = self.pixel_queries(pixel_feature)

        n_patch = pixel_queries.shape[1]
        pixel_queries = pixel_queries.reshape(
            n_batch * n_patch,
            self.patch_size * self.patch_size,
            self.n_channels)
        patch_context = patch_context.reshape(
            n_batch * n_patch,
            9,
            self.n_channels)
        pixel_queries = self.pixel_from_patch_attention(
            pixel_queries,
            patch_context)
        pixel_queries = pixel_queries.reshape(
            n_batch,
            n_patch,
            self.patch_size * self.patch_size,
            self.n_channels)

        feature = self.queries_to_feature(
            pixel_queries,
            n_height=n_height,
            n_width=n_width)
        raw_depth = self.depth_output(feature)

        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        return torch.exp(
            log_min_depth +
            normalized_depth * (log_max_depth - log_min_depth))
