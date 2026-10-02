import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(
    'external_src', 'depth_completion', 'kbnet', 'src'))
from net_utils import Conv2d

PARTITION_SIZES = [
    (128, 128),
    (8, 8),
    (4, 4),
    (2, 2),
    (1, 1)
]

class AttentionUpdate(nn.Module):
    '''Update with attention '''
    def __init__(self, n_channels, n_head):
        super(AttentionUpdate, self).__init__()

        self.n_head = n_head
        self.head_channels = n_channels // n_head
        self.query_chunk_size = 32

        # Q, K and V
        self.query_projection = nn.Linear(n_channels, n_channels)
        self.key_projection = nn.Linear(n_channels, n_channels)
        self.value_projection = nn.Linear(n_channels, n_channels)

        # Attention result 
        self.output_projection = nn.Linear(n_channels, n_channels)

        # Layer normalization and feedforward after multi-head attention.
        self.attention_norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Linear(4 * n_channels, n_channels))
        self.feedforward_norm = nn.LayerNorm(n_channels)

    def forward(self, query, context, attention_mask=None):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

        # normalize query and context
        residual = query
        query = self.attention_norm(query)
        context = self.context_norm(context)

        # Prepare Q, K, and V for multi-head attention
        # Q: N x L_q x C_q -> N x heads x L_q x D_head
        projected_query = self.query_projection(query)
        projected_query = projected_query.reshape(
            n_batch, 
            n_query, 
            self.n_head, 
            self.head_channels)
        projected_query = projected_query.permute(0, 2, 1, 3)

        # K and V: N x L_k x C_k -> N x heads x L_k x D_head
        projected_key = self.key_projection(context)
        projected_key = projected_key.reshape(
            n_batch, 
            n_context, 
            self.n_head, 
            self.head_channels)
        projected_key = projected_key.permute(0, 2, 1, 3)

        projected_value = self.value_projection(context)
        projected_value = projected_value.reshape(
            n_batch, 
            n_context, 
            self.n_head, 
            self.head_channels)
        projected_value = projected_value.permute(0, 2, 1, 3)

        # attended = softmax(Q K^T / sqrt(D_head)) V
        similarity = torch.matmul(
            projected_query,
            projected_key.transpose(-2, -1))

        similarity = similarity / math.sqrt(self.head_channels)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(
                n_batch // attention_mask.shape[0],
                1,
                1)
            similarity = similarity + attention_mask[:, None, :, :]

        attention_weights = torch.softmax(similarity, dim=-1)

        attended = torch.matmul(
            attention_weights,
            projected_value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            n_batch, 
            n_query, 
            self.n_head * self.head_channels)

        attended = self.output_projection(attended)

        # Residual connection after attention
        x = residual + attended

        # Layer norm before ffn
        feedforward = self.feedforward(self.feedforward_norm(x))

        # Residual connection after feedforward
        x = x + feedforward

        return x


class ConvolutionPyramid(nn.Module):
    '''Creates five sequential spatial levels from RGB.'''

    def __init__(self, input_channels, n_channels):
        super(ConvolutionPyramid, self).__init__()

        # Three 3x3 convolutions first produce the full-resolution level R_0.
        self.full_resolution_convolutions = nn.ModuleList([
            nn.Sequential(
                Conv2d(
                    input_channels if layer == 0 else n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1)
                )
            for layer in range(3)
        ])

        # Every following 3x3 convolution acts on the preceding level. The
        # first block uses stride 16, then the remaining blocks use stride 2.
        self.downsample_convolutions = nn.ModuleList([
            nn.Sequential(
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1),
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,),
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=16 if level == 0 else 2)
                )
            for level in range(4)
        ])

    def forward(self, image):
        feature = image
        for convolution in self.full_resolution_convolutions:
            feature = convolution(feature)

        features = [feature]
        for convolution in self.downsample_convolutions:
            feature = convolution(feature)
            features.append(feature)

        return features


def feature_to_partitions(feature, partition_height, partition_width):
    '''Divide a feature map into a grid of partitions.'''

    # Partition height and width is always FIXED.

    # The result is B x grid_h x grid_w x T x C, where T is partition height x
    # partition width. The grid dimensions are the same across all levels,
    # while T decreases.
    n_batch, n_channel, n_height, n_width = feature.shape

    # For example: if image size is 512 x 512, then
    # n_partition_height = 4, n_partition_width = 4
    n_partition_height = n_height // partition_height
    n_partition_width = n_width // partition_width

    partitions = feature.reshape(
        n_batch,
        n_channel,
        n_partition_height,
        partition_height,
        n_partition_width,
        partition_width)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1)
    return partitions.reshape(
        n_batch,
        n_partition_height,
        n_partition_width,
        partition_height * partition_width,
        n_channel)


def partitions_to_feature(partitions, n_height, n_width):
    '''Reverses partitions to feature'''

    n_batch, n_partition_height, n_partition_width, _, n_channel = partitions.shape
    partition_height = n_height // n_partition_height
    partition_width = n_width // n_partition_width
    feature = partitions.reshape(
        n_batch,
        n_partition_height,
        n_partition_width,
        partition_height,
        partition_width,
        n_channel)
    feature = feature.permute(0, 5, 1, 3, 2, 4)
    return feature.reshape(
        n_batch, 
        n_channel, 
        n_height, 
        n_width)


def add_position_encoding(feature):
    '''Add 2D position information to every feature token.'''
    _, n_channel, n_height, n_width = feature.shape
    n_frequency = n_channel // 4

    # Normalize so positions from different resolutions use the same coordinate system.
    y = (torch.arange(
        n_height,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_height
    x = (torch.arange(
        n_width,
        dtype=feature.dtype,
        device=feature.device) + 0.5) / n_width

    # Multiple Fourier frequencies
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


class PartitionAttentionDepthModel(nn.Module):
    '''
    RGB forms a sequential five-level pyramid. Every level uses the same
    partition grid. Information is exchanged globally at the bottom
    level, and then from the bottom level back to the full-resolution level.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 n_self_attention=2):
        super(PartitionAttentionDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.n_channels = n_channels
        self.n_level = 5
        self.n_iteration = 2
        self.n_self_attention = n_self_attention

        # The RGB convolutions operate sequentially. R_0 is full resolution,
        # R_1 is downsampled by 16, and each remaining level is downsampled by 2.
        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        # Step 1: every partition below the full-resolution level performs
        # self attention independently.
        # self.rgb_local_attention = nn.ModuleList([
        #     AttentionUpdate(n_channels, n_head)
        #     for _ in range(self.n_level - 1)
        # ])

        # Step 2: all tokens from all level below first partitions attend to one
        # another, providing global communication across the grid.
        self.rgb_full_attention = nn.ModuleList([
            nn.ModuleList([
                AttentionUpdate(n_channels, n_head)
                for _ in range(n_self_attention)
            ])
            for _ in range(self.n_level - 1)
        ])

        # Step 3: every upper-level partition queries the corresponding
        # partition at the adjacent lower-resolution level.
        self.rgb_coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # A second R_1 to R_0 attention block uses a half-partition-shifted
        # grid. Its output is blended with the regular-grid output.
        self.rgb_shifted_coarse_to_fine_attention = AttentionUpdate(
            n_channels,
            n_head)

        # Step 4: every lower-level partition queries the corresponding
        # partition at the adjacent upper-resolution level.
        self.rgb_fine_to_coarse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # After propagating back to the full-resolution level, restore the
        # partitions to one spatial feature map and pass it through three
        # convolutions with activation.
        self.convolutiontofullres = nn.ModuleList([
            nn.Sequential(
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
                Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)))
            for _ in range(self.n_iteration)
        ])

        # Three spatial convolutions decode the final full-resolution feature.
        # The first two use Conv2d's activation and the last produces raw depth.
        self.depth_output = nn.Sequential(
            Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels,
                1,
                kernel_size=3,
                stride=1,
                activation_func=None))

    def local_attention(self, partitions, attention_blocks):
        '''Local attention within each partition.'''
        outputs = []
        for level_partitions, attention_block in zip(partitions, attention_blocks):
            n_batch, n_partition_height, n_partition_width, n_token, n_channel = level_partitions.shape
            tokens = level_partitions.reshape(
                n_batch * n_partition_height * n_partition_width,
                n_token,
                n_channel)

            tokens = attention_block(tokens, tokens)
            outputs.append(tokens.reshape(
                n_batch,
                n_partition_height,
                n_partition_width,
                n_token,
                n_channel))
        return outputs

    def bottom_attention(self, partitions, attention_block):
        '''Full self-attention across every partition in one level.'''
        n_batch, n_partition_height, n_partition_width, n_token, n_channel = partitions.shape

        # Every bottom token attend to every token from every other bottom partition.
        tokens = partitions.reshape(
            n_batch,
            n_partition_height * n_partition_width * n_token,
            n_channel)
        tokens = attention_block(tokens, tokens)
        return tokens.reshape(partitions.shape)

    def repeated_bottom_attention(self, partitions, attention_blocks):
        '''Apply multiple consecutive full self-attention blocks.'''
        for attention_block in attention_blocks:
            partitions = self.bottom_attention(
                partitions,
                attention_block)
        return partitions

    def coarse_to_fine_level(
            self,
            fine,
            coarse,
            attention_block,
            attention_mask=None):
        '''Update each fine partition from its corresponding coarse partition.'''
        n_batch, n_partition_height, n_partition_width, n_fine_token, n_channel = fine.shape
        n_coarse_token = coarse.shape[3]

        fine_queries = fine.reshape(
            n_batch * n_partition_height * n_partition_width,
            n_fine_token,
            n_channel)
        coarse_context = coarse.reshape(
            n_batch * n_partition_height * n_partition_width,
            n_coarse_token,
            n_channel)
        updated_fine = attention_block(
            fine_queries,
            coarse_context,
            attention_mask=attention_mask)
        return updated_fine.reshape(fine.shape)

    def swin_coarse_to_fine_level(
            self,
            fine,
            coarse,
            attention_block,
            shifted_attention_block):
        '''Blend regular and shifted partition attention from R_1 to R_0.'''

        # Both branches start from the same R_0 and R_1 features. The regular
        # branch uses the original corresponding 4 x 4 partition grid.
        regular_fine = self.coarse_to_fine_level(
            fine,
            coarse,
            attention_block)

        n_fine_height = fine.shape[1] * PARTITION_SIZES[0][0]
        n_fine_width = fine.shape[2] * PARTITION_SIZES[0][1]
        n_coarse_height = coarse.shape[1] * PARTITION_SIZES[1][0]
        n_coarse_width = coarse.shape[2] * PARTITION_SIZES[1][1]

        fine_feature = partitions_to_feature(
            fine,
            n_height=n_fine_height,
            n_width=n_fine_width)
        coarse_feature = partitions_to_feature(
            coarse,
            n_height=n_coarse_height,
            n_width=n_coarse_width)

        # The second branch shifts both levels by half of their own partition
        # size, so its windows cross the boundaries of the regular grid.
        fine_shift = (
            PARTITION_SIZES[0][0] // 2,
            PARTITION_SIZES[0][1] // 2)
        coarse_shift = (
            PARTITION_SIZES[1][0] // 2,
            PARTITION_SIZES[1][1] // 2)
        shifted_fine_feature = torch.roll(
            fine_feature,
            shifts=(-fine_shift[0], -fine_shift[1]),
            dims=(-2, -1))
        shifted_coarse_feature = torch.roll(
            coarse_feature,
            shifts=(-coarse_shift[0], -coarse_shift[1]),
            dims=(-2, -1))
        shifted_fine = feature_to_partitions(
            shifted_fine_feature,
            partition_height=PARTITION_SIZES[0][0],
            partition_width=PARTITION_SIZES[0][1])
        shifted_coarse = feature_to_partitions(
            shifted_coarse_feature,
            partition_height=PARTITION_SIZES[1][0],
            partition_width=PARTITION_SIZES[1][1])

        # Build a cross-resolution Swin mask. A fine query and coarse key with
        # different labels are adjacent only because torch.roll wrapped them
        # across an outer image boundary.
        fine_region = torch.zeros(
            1,
            1,
            n_fine_height,
            n_fine_width,
            dtype=fine_feature.dtype,
            device=fine_feature.device)
        coarse_region = torch.zeros(
            1,
            1,
            n_coarse_height,
            n_coarse_width,
            dtype=coarse_feature.dtype,
            device=coarse_feature.device)
        fine_height_slices = (
            slice(0, -PARTITION_SIZES[0][0]),
            slice(-PARTITION_SIZES[0][0], -fine_shift[0]),
            slice(-fine_shift[0], None))
        fine_width_slices = (
            slice(0, -PARTITION_SIZES[0][1]),
            slice(-PARTITION_SIZES[0][1], -fine_shift[1]),
            slice(-fine_shift[1], None))
        coarse_height_slices = (
            slice(0, -PARTITION_SIZES[1][0]),
            slice(-PARTITION_SIZES[1][0], -coarse_shift[0]),
            slice(-coarse_shift[0], None))
        coarse_width_slices = (
            slice(0, -PARTITION_SIZES[1][1]),
            slice(-PARTITION_SIZES[1][1], -coarse_shift[1]),
            slice(-coarse_shift[1], None))
        region_index = 0
        for fine_height_slice, coarse_height_slice in zip(
                fine_height_slices,
                coarse_height_slices):
            for fine_width_slice, coarse_width_slice in zip(
                    fine_width_slices,
                    coarse_width_slices):
                fine_region[
                    :, :, fine_height_slice, fine_width_slice] = region_index
                coarse_region[
                    :, :, coarse_height_slice, coarse_width_slice] = region_index
                region_index = region_index + 1

        fine_region_tokens = feature_to_partitions(
            fine_region,
            partition_height=PARTITION_SIZES[0][0],
            partition_width=PARTITION_SIZES[0][1]).reshape(
                fine.shape[1] * fine.shape[2],
                fine.shape[3])
        coarse_region_tokens = feature_to_partitions(
            coarse_region,
            partition_height=PARTITION_SIZES[1][0],
            partition_width=PARTITION_SIZES[1][1]).reshape(
                coarse.shape[1] * coarse.shape[2],
                coarse.shape[3])
        attention_mask = \
            fine_region_tokens[:, :, None] - \
            coarse_region_tokens[:, None, :]
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            -100.0)
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0,
            0.0)

        shifted_fine = self.coarse_to_fine_level(
            shifted_fine,
            shifted_coarse,
            shifted_attention_block,
            attention_mask=attention_mask)

        regular_fine_feature = partitions_to_feature(
            regular_fine,
            n_height=n_fine_height,
            n_width=n_fine_width)
        shifted_fine_feature = partitions_to_feature(
            shifted_fine,
            n_height=n_fine_height,
            n_width=n_fine_width)
        shifted_fine_feature = torch.roll(
            shifted_fine_feature,
            shifts=fine_shift,
            dims=(-2, -1))

        # A cosine confidence is largest at a window center and smallest at
        # its boundary. The shifted confidence has the complementary layout.
        y = (torch.arange(
            n_fine_height,
            dtype=fine_feature.dtype,
            device=fine_feature.device) + 0.5) % PARTITION_SIZES[0][0]
        x = (torch.arange(
            n_fine_width,
            dtype=fine_feature.dtype,
            device=fine_feature.device) + 0.5) % PARTITION_SIZES[0][1]
        y_confidence = torch.sin(
            math.pi * y / PARTITION_SIZES[0][0]) ** 2
        x_confidence = torch.sin(
            math.pi * x / PARTITION_SIZES[0][1]) ** 2
        regular_confidence = \
            (y_confidence[:, None] * x_confidence[None, :])[None, None, :, :]
        shifted_confidence = torch.roll(
            regular_confidence,
            shifts=fine_shift,
            dims=(-2, -1))
        confidence_sum = regular_confidence + shifted_confidence
        zero_confidence = confidence_sum == 0
        confidence_sum = confidence_sum.masked_fill(
            zero_confidence,
            1.0)
        regular_weight = regular_confidence / confidence_sum
        regular_weight = regular_weight.masked_fill(
            zero_confidence,
            0.5)
        shifted_weight = 1.0 - regular_weight

        fine_feature = \
            regular_weight * regular_fine_feature + \
            shifted_weight * shifted_fine_feature
        return feature_to_partitions(
            fine_feature,
            partition_height=PARTITION_SIZES[0][0],
            partition_width=PARTITION_SIZES[0][1])

    def fine_to_coarse_level(self, coarse, fine, attention_block):
        '''Update each coarse partition from its corresponding fine partition.'''
        n_batch, n_partition_height, n_partition_width, n_coarse_token, n_channel = coarse.shape
        n_fine_token = fine.shape[3]

        coarse_queries = coarse.reshape(
            n_batch * n_partition_height * n_partition_width,
            n_coarse_token,
            n_channel)
        fine_context = fine.reshape(
            n_batch * n_partition_height * n_partition_width,
            n_fine_token,
            n_channel)
        updated_coarse = attention_block(
            coarse_queries,
            fine_context)
        return updated_coarse.reshape(coarse.shape)

    def forward(self, image):

        # The RGB pyramid is sequential.
        rgb_features = self.rgb_pyramid(image)
        rgb_features = [
            add_position_encoding(feature)
            for feature in rgb_features
        ]

        rgb_partitions = [
            feature_to_partitions(
                feature,
                partition_height,
                partition_width)
            for feature, (partition_height, partition_width) in 
                zip(rgb_features, PARTITION_SIZES)
        ]

        # Step 1: local self-attention for every level except R_0.
        # rgb_partitions[1:] = self.local_attention(
        #     rgb_partitions[1:],
        #     self.rgb_local_attention)

        # Full self attention at the bottom level before traveling upward.
        rgb_partitions[-1] = self.repeated_bottom_attention(
            rgb_partitions[-1],
            self.rgb_full_attention[-1])

        for iteration in range(self.n_iteration):

            # Bottom-up travel.
            for level in range(self.n_level - 2, -1, -1):

                # Coarse to fine exchange
                if level == 0:
                    rgb_partitions[level] = self.swin_coarse_to_fine_level(
                        fine=rgb_partitions[level],
                        coarse=rgb_partitions[level + 1],
                        attention_block=self.rgb_coarse_to_fine_attention[level],
                        shifted_attention_block=self.rgb_shifted_coarse_to_fine_attention)
                else:
                    rgb_partitions[level] = self.coarse_to_fine_level(
                        fine=rgb_partitions[level],
                        coarse=rgb_partitions[level + 1],
                        attention_block=self.rgb_coarse_to_fine_attention[level])
                
                if level == 0:
                    full_resolution_feature = partitions_to_feature(
                        rgb_partitions[level],
                        n_height=image.shape[-2],
                        n_width=image.shape[-1])
                    full_resolution_feature = self.convolutiontofullres[iteration](
                        full_resolution_feature)
                    rgb_partitions[level] = feature_to_partitions(
                        full_resolution_feature,
                        partition_height=PARTITION_SIZES[level][0],
                        partition_width=PARTITION_SIZES[level][1])

                # Full self attention for every level except the top level.
                if level > 0:
                    rgb_partitions[level] = self.repeated_bottom_attention(
                        rgb_partitions[level],
                        self.rgb_full_attention[level - 1])

            if iteration < self.n_iteration - 1:

                # Top-down travel.
                for level in range(1, self.n_level):

                    # Fine to coarse exchange
                    rgb_partitions[level] = self.fine_to_coarse_level(
                        coarse=rgb_partitions[level],
                        fine=rgb_partitions[level - 1],
                        attention_block=self.rgb_fine_to_coarse_attention[level - 1])

                    # Full self attention after each partition exchange.
                    rgb_partitions[level] = self.repeated_bottom_attention(
                        rgb_partitions[level],
                        self.rgb_full_attention[level - 1])

        # Restore the final full-resolution partitions and decode spatially.
        full_rgb = partitions_to_feature(
            rgb_partitions[0],
            n_height=image.shape[-2],
            n_width=image.shape[-1])
        raw_depth = self.depth_output(full_rgb)
        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        return torch.exp(
            log_min_depth +
            normalized_depth * (log_max_depth - log_min_depth))
