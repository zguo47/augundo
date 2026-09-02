import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class AttentionUpdate(nn.Module):
    '''
    Attention between query and context sequences with the same channel count.
    '''

    def __init__(self, n_channels, n_head):
        super(AttentionUpdate, self).__init__()

        self.n_head = n_head
        self.head_channels = n_channels // n_head

        # Q, K and V remain in the common C-channel space used by every level.
        self.query_projection = nn.Linear(n_channels, n_channels)
        self.key_projection = nn.Linear(n_channels, n_channels)
        self.value_projection = nn.Linear(n_channels, n_channels)

        # The attention result also has C channels, so it can be added directly
        # to the C-channel query sequence.
        self.output_projection = nn.Linear(n_channels, n_channels)

    def forward(self, query, context):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

        # Q: N x L_q x C_q -> N x heads x L_q x D_head
        projected_query = self.query_projection(query)
        projected_query = projected_query.reshape(
            n_batch, n_query, self.n_head, self.head_channels)
        projected_query = projected_query.permute(0, 2, 1, 3)

        # K and V: N x L_k x C_k -> N x heads x L_k x D_head
        projected_key = self.key_projection(context)
        projected_key = projected_key.reshape(
            n_batch, n_context, self.n_head, self.head_channels)
        projected_key = projected_key.permute(0, 2, 1, 3)

        projected_value = self.value_projection(context)
        projected_value = projected_value.reshape(
            n_batch, n_context, self.n_head, self.head_channels)
        projected_value = projected_value.permute(0, 2, 1, 3)

        # Scaled dot-product attention performs these exact operations:
        #   weights = softmax(Q K^T / sqrt(D_head))
        #   attended = weights V
        # Thus each query receives a weighted sum of every context value. The
        # fused primitive does not change or approximate the attention rule.
        attended = functional.scaled_dot_product_attention(
            projected_query,
            projected_key,
            projected_value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            n_batch, n_query, self.n_head * self.head_channels)
        attended = self.output_projection(attended)

        # The residual retains the query and adds only the information obtained
        # from the context tokens.
        return query + attended


class ConvolutionPyramid(nn.Module):
    '''Creates five parallel spatial levels with no padding or overlap.'''

    SCALES = (1, 2, 4, 8, 16)

    def __init__(self, input_channels, n_channels):
        super(ConvolutionPyramid, self).__init__()

        # Every level is produced directly from the branch input. Kernel size
        # equals stride and padding is zero. Consequently convolution windows
        # do not overlap, and the results are H, H/2, H/4, H/8 and H/16 for
        # input dimensions divisible by sixteen.
        self.convolutions = nn.ModuleList([
            nn.Conv2d(
                input_channels,
                n_channels,
                kernel_size=scale,
                stride=scale)
            for scale in self.SCALES
        ])

    def forward(self, branch_input):
        return [
            convolution(branch_input)
            for convolution in self.convolutions
        ]


def feature_to_partitions(feature, n_grid):
    '''Rearranges a feature map into n_grid x n_grid token sequences.'''

    n_batch, n_channel, n_height, n_width = feature.shape
    partition_height = n_height // n_grid
    partition_width = n_width // n_grid

    # Spatial positions inside one partition become the token dimension P.
    # The result keeps the two grid coordinates explicit for cross-level maps.
    partitions = feature.reshape(
        n_batch,
        n_channel,
        n_grid,
        partition_height,
        n_grid,
        partition_width)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1)
    return partitions.reshape(
        n_batch,
        n_grid,
        n_grid,
        partition_height * partition_width,
        n_channel)


def partitions_to_feature(partitions, n_height, n_width):
    '''Reverses feature_to_partitions without interpolation or filtering.'''

    n_batch, n_grid, _, _, n_channel = partitions.shape
    partition_height = n_height // n_grid
    partition_width = n_width // n_grid
    feature = partitions.reshape(
        n_batch,
        n_grid,
        n_grid,
        partition_height,
        partition_width,
        n_channel)
    feature = feature.permute(0, 5, 1, 3, 2, 4)
    return feature.reshape(
        n_batch, n_channel, n_height, n_width)


class PartitionAttentionDepthModel(nn.Module):
    '''
    Minimal two-branch partition-attention baseline.

    Both RGB and sparse depth form the same five-level spatial hierarchy. Each
    branch first exchanges information locally, the two modalities then
    exchange information at every level, and adjacent levels communicate in a
    fine-to-coarse pass followed by a coarse-to-fine pass.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4):
        super(PartitionAttentionDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.n_channels = n_channels

        # A partition has the spatial dimensions of the smallest feature map.
        # The five levels therefore contain 16x16, 8x8, 4x4, 2x2 and 1x1
        # non-overlapping grid partitions.
        self.level_grids = (16, 8, 4, 2, 1)

        # RGB and sparse depth use independent parallel convolutions. Every
        # output level in both branches has exactly C channels.
        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)
        self.sparse_pyramid = ConvolutionPyramid(
            input_channels=1,
            n_channels=n_channels)

        # Step 1: self-attention is applied independently inside every
        # partition of every level, for both input branches.
        self.rgb_local_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in self.level_grids
        ])
        self.sparse_local_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in self.level_grids
        ])

        # Sparse-to-RGB and RGB-to-sparse attention make the exchange
        # bidirectional. Corresponding partitions at a level have equal token
        # counts and equal channel counts, but use separate learned weights.
        self.rgb_from_sparse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in self.level_grids
        ])
        self.sparse_from_rgb_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in self.level_grids
        ])

        # Step 2: a coarse partition queries the four fine partitions occupying
        # the same image region. Both query and context have C channels.
        self.rgb_fine_to_coarse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(4)
        ])
        self.sparse_fine_to_coarse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(4)
        ])

        # Step 3: each fine partition queries its updated parent partition. The
        # result remains in the shared C-channel space.
        self.rgb_coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(4)
        ])
        self.sparse_coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(4)
        ])

        # One linear map converts each final full-resolution RGB token into one
        # depth value. This is not a spatial convolution and does not mix pixels.
        self.depth_output = nn.Linear(n_channels, 1)

    def _local_attention(self, partitions, attention_blocks):
        outputs = []
        for level_partitions, attention_block in zip(
                partitions, attention_blocks):
            n_batch, n_grid, _, n_token, n_channel = \
                level_partitions.shape
            tokens = level_partitions.reshape(
                n_batch * n_grid * n_grid,
                n_token,
                n_channel)
            tokens = attention_block(tokens, tokens)
            outputs.append(tokens.reshape(
                n_batch,
                n_grid,
                n_grid,
                n_token,
                n_channel))
        return outputs

    def _cross_modal_level(self, rgb, sparse, level):
        n_batch, n_grid, _, n_token, n_channel = rgb.shape
        rgb_tokens = rgb.reshape(
            n_batch * n_grid * n_grid,
            n_token,
            n_channel)
        sparse_tokens = sparse.reshape(
            n_batch * n_grid * n_grid,
            n_token,
            n_channel)

        # RGB queries attend to sparse-depth tokens in the same partition.
        # Sparse-depth queries independently attend to the RGB tokens in that
        # partition, so information exchange is bidirectional.
        updated_rgb = self.rgb_from_sparse_attention[level](
            rgb_tokens, sparse_tokens)
        updated_sparse = self.sparse_from_rgb_attention[level](
            sparse_tokens, rgb_tokens)
        return updated_rgb.reshape(rgb.shape), \
            updated_sparse.reshape(sparse.shape)

    def _fine_context_for_coarse(self, fine_partitions):
        n_batch, fine_grid, _, n_token, n_channel = fine_partitions.shape
        coarse_grid = fine_grid // 2

        # Reorder the 2x2 children of every coarse partition next to one
        # another, then concatenate their token dimensions from 4P to one 4P
        # context sequence.
        context = fine_partitions.reshape(
            n_batch,
            coarse_grid,
            2,
            coarse_grid,
            2,
            n_token,
            n_channel)
        context = context.permute(0, 1, 3, 2, 4, 5, 6)
        return context.reshape(
            n_batch * coarse_grid * coarse_grid,
            4 * n_token,
            n_channel)

    def _fine_to_coarse(self, partitions, attention_blocks):
        outputs = list(partitions)

        for level, attention_block in enumerate(attention_blocks):
            fine_context = self._fine_context_for_coarse(outputs[level])
            coarse = outputs[level + 1]
            n_batch, n_grid, _, n_token, n_channel = coarse.shape
            coarse_queries = coarse.reshape(
                n_batch * n_grid * n_grid,
                n_token,
                n_channel)
            updated_coarse = attention_block(
                coarse_queries,
                fine_context)
            outputs[level + 1] = updated_coarse.reshape(coarse.shape)

        return outputs

    def _parent_context_for_fine(self, coarse_partitions):
        n_batch, coarse_grid, _, n_token, n_channel = \
            coarse_partitions.shape

        # Copy each parent reference to its four child positions. Each child
        # subsequently uses the same updated parent partition as its context.
        context = coarse_partitions.reshape(
            n_batch,
            coarse_grid,
            1,
            coarse_grid,
            1,
            n_token,
            n_channel)
        context = context.expand(
            n_batch,
            coarse_grid,
            2,
            coarse_grid,
            2,
            n_token,
            n_channel)
        return context.reshape(
            n_batch * (2 * coarse_grid) * (2 * coarse_grid),
            n_token,
            n_channel)

    def _coarse_to_fine_level(
            self, fine, coarse, attention_block):
        n_batch, n_grid, _, n_token, n_channel = fine.shape
        fine_queries = fine.reshape(
            n_batch * n_grid * n_grid,
            n_token,
            n_channel)
        parent_context = self._parent_context_for_fine(coarse)
        updated_fine = attention_block(
            fine_queries,
            parent_context)
        return updated_fine.reshape(fine.shape)

    def forward(self, image, sparse_depth, validity_map=None):
        # Initial convolutions create the five RGB and sparse-depth levels.
        rgb_features = self.rgb_pyramid(image)
        sparse_features = self.sparse_pyramid(sparse_depth)

        # A fixed token-space partition size H/16 x W/16 gives level grids of
        # 16x16, 8x8, 4x4, 2x2 and 1x1. The reshape forms a strict grid: no
        # partition contains a token belonging to another partition.
        rgb_partitions = [
            feature_to_partitions(feature, n_grid)
            for feature, n_grid in zip(rgb_features, self.level_grids)
        ]
        sparse_partitions = [
            feature_to_partitions(feature, n_grid)
            for feature, n_grid in zip(sparse_features, self.level_grids)
        ]

        # Step 1: attention only among tokens inside the same partition.
        rgb_partitions = self._local_attention(
            rgb_partitions, self.rgb_local_attention)
        sparse_partitions = self._local_attention(
            sparse_partitions, self.sparse_local_attention)

        # Step 2: fine-to-coarse attention. A coarse partition attends to its
        # 2x2 group of fine children, joining neighboring fine partitions.
        rgb_partitions = self._fine_to_coarse(
            rgb_partitions, self.rgb_fine_to_coarse_attention)
        sparse_partitions = self._fine_to_coarse(
            sparse_partitions, self.sparse_fine_to_coarse_attention)

        # The bottom partitions cover the whole image. They exchange RGB and
        # sparse-depth information before their global context travels upward.
        rgb_partitions[4], sparse_partitions[4] = \
            self._cross_modal_level(
                rgb_partitions[4], sparse_partitions[4], level=4)

        # Step 3: proceed from bottom to top. At each adjacent pair, every fine
        # partition first attends to the coarse partition covering the same
        # image region. Corresponding RGB and sparse partitions then exchange
        # their updated information through attention.
        for level in range(3, -1, -1):
            rgb_partitions[level] = self._coarse_to_fine_level(
                fine=rgb_partitions[level],
                coarse=rgb_partitions[level + 1],
                attention_block=self.rgb_coarse_to_fine_attention[level])
            sparse_partitions[level] = self._coarse_to_fine_level(
                fine=sparse_partitions[level],
                coarse=sparse_partitions[level + 1],
                attention_block=self.sparse_coarse_to_fine_attention[level])
            rgb_partitions[level], sparse_partitions[level] = \
                self._cross_modal_level(
                    rgb_partitions[level],
                    sparse_partitions[level],
                    level=level)

        # The sparse branch has already entered the RGB branch through the
        # bidirectional cross-modal attention. Depth is read from the final
        # full-resolution RGB tokens with a per-token linear projection.
        full_rgb = rgb_partitions[0]
        raw_depth = self.depth_output(full_rgb)
        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        depth_partitions = torch.exp(
            log_min_depth +
            normalized_depth * (log_max_depth - log_min_depth))

        return partitions_to_feature(
            depth_partitions,
            n_height=image.shape[-2],
            n_width=image.shape[-1])
