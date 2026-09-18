import math

import torch
import torch.nn as nn

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

    def forward(self, query, context):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

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

        attention_weights = torch.softmax(similarity, dim=-1)

        attended = torch.matmul(
            attention_weights,
            projected_value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            n_batch, 
            n_query, 
            self.n_head * self.head_channels)
        attended = self.output_projection(attended)

        return query + attended


class ConvolutionPyramid(nn.Module):
    '''Creates five sequential spatial levels from RGB.'''

    def __init__(self, input_channels, n_channels):
        super(ConvolutionPyramid, self).__init__()

        # Three 3x3 convolutions first produce the full-resolution level R_0.
        self.full_resolution_convolutions = nn.ModuleList([
            nn.Conv2d(
                input_channels if layer == 0 else n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            for layer in range(3)
        ])

        # Every following 3x3 convolution acts on the preceding level. Its
        # stride of 2 halves the height and width, creating R_1 through R_4.
        self.downsample_convolutions = nn.ModuleList([
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=2,
                padding=1)
            for _ in range(4)
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


def feature_to_partitions(feature, n_grid_height, n_grid_width=None):
    '''Divide a feature map into a grid of partitions.'''

    if n_grid_width is None:
        n_grid_width = n_grid_height

    # The result is B x grid_h x grid_w x T x C. The same grid dimensions
    # are used at every level, while T decreases with spatial resolution.
    n_batch, n_channel, n_height, n_width = feature.shape
    partition_height = n_height // n_grid_height
    partition_width = n_width // n_grid_width

    partitions = feature.reshape(
        n_batch,
        n_channel,
        n_grid_height,
        partition_height,
        n_grid_width,
        partition_width)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1)
    return partitions.reshape(
        n_batch,
        n_grid_height,
        n_grid_width,
        partition_height * partition_width,
        n_channel)


def partitions_to_feature(partitions, n_height, n_width):
    '''Reverses partitions to feature'''

    n_batch, n_grid_height, n_grid_width, _, n_channel = partitions.shape
    partition_height = n_height // n_grid_height
    partition_width = n_width // n_grid_width
    feature = partitions.reshape(
        n_batch,
        n_grid_height,
        n_grid_width,
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
    partition grid. Information is exchanged locally, globally at the bottom
    level, and then from the bottom level back to the full-resolution level.
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
        self.n_level = 5

        # The RGB convolutions operate sequentially. R_0 is full resolution,
        # and each subsequent level has half the preceding spatial resolution.
        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        # Step 1: every partition at every resolution has an independent
        # local self-attention operation.
        self.rgb_local_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level)
        ])

        # Step 2: all tokens from all bottom-level partitions attend to one
        # another, providing global communication across the partition grid.
        self.rgb_bottom_attention = AttentionUpdate(n_channels, n_head)

        # Step 3: every upper-level partition queries the corresponding
        # partition at the adjacent lower-resolution level.
        self.rgb_coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Each final full-resolution RGB token is decoded to one depth value.
        self.depth_output = nn.Linear(n_channels, 1)

    def local_attention(self, partitions, attention_blocks):
        '''Local attention within each partition.'''
        outputs = []
        for level_partitions, attention_block in zip(partitions, attention_blocks):
            n_batch, n_grid_height, n_grid_width, n_token, n_channel = \
                level_partitions.shape
            tokens = level_partitions.reshape(
                n_batch * n_grid_height * n_grid_width,
                n_token,
                n_channel)

            # The grid dimensions are folded into the batch dimension, so all
            # partitions are processed independently and simultaneously.
            tokens = attention_block(tokens, tokens)
            outputs.append(tokens.reshape(
                n_batch,
                n_grid_height,
                n_grid_width,
                n_token,
                n_channel))
        return outputs

    def bottom_attention(self, partitions, attention_block):
        '''Full self-attention across every bottom-level partition.'''
        n_batch, n_grid_height, n_grid_width, n_token, n_channel = \
            partitions.shape

        # Unlike local attention, the grid coordinates are folded into the
        # sequence dimension. Every bottom token can therefore attend to every
        # token from every other bottom partition.
        tokens = partitions.reshape(
            n_batch,
            n_grid_height * n_grid_width * n_token,
            n_channel)
        tokens = attention_block(tokens, tokens)
        return tokens.reshape(partitions.shape)

    def coarse_to_fine_level(self, fine, coarse, attention_block):
        '''Update each fine partition from its corresponding coarse partition.'''
        n_batch, n_grid_height, n_grid_width, n_fine_token, n_channel = \
            fine.shape
        n_coarse_token = coarse.shape[3]

        # The grid is identical at both levels. Folding the same grid location
        # into the batch dimension pairs each fine partition only with its
        # spatially corresponding coarse partition.
        fine_queries = fine.reshape(
            n_batch * n_grid_height * n_grid_width,
            n_fine_token,
            n_channel)
        coarse_context = coarse.reshape(
            n_batch * n_grid_height * n_grid_width,
            n_coarse_token,
            n_channel)
        updated_fine = attention_block(
            fine_queries,
            coarse_context)
        return updated_fine.reshape(fine.shape)

    def forward(self, image, sparse_depth=None, validity_map=None):
        del sparse_depth, validity_map

        # The RGB pyramid is sequential: R_l is produced directly from R_l-1.
        rgb_features = self.rgb_pyramid(image)
        rgb_features = [
            add_position_encoding(feature)
            for feature in rgb_features
        ]

        # The bottom feature dimensions define one fixed partition grid for
        # every level. For a 448x640 image this is a 28x40 grid at all levels,
        # giving partition sizes 16x16, 8x8, 4x4, 2x2 and 1x1.
        n_grid_height = rgb_features[-1].shape[-2]
        n_grid_width = rgb_features[-1].shape[-1]
        rgb_partitions = [
            feature_to_partitions(
                feature,
                n_grid_height,
                n_grid_width)
            for feature in rgb_features
        ]

        # Step 1: local self-attention is independent within every partition
        # and is applied at all five resolutions.
        rgb_partitions = self.local_attention(
            rgb_partitions,
            self.rgb_local_attention)

        # Step 2: the bottom level performs full attention across its entire
        # partition grid before the information travels upward.
        rgb_partitions[-1] = self.bottom_attention(
            rgb_partitions[-1],
            self.rgb_bottom_attention)

        # Step 3: update R_3 from R_4, then R_2 from R_3, continuing until the
        # full-resolution R_0 has received information from all lower levels.
        for level in range(self.n_level - 2, -1, -1):
            rgb_partitions[level] = self.coarse_to_fine_level(
                fine=rgb_partitions[level],
                coarse=rgb_partitions[level + 1],
                attention_block=self.rgb_coarse_to_fine_attention[level])

        # Depth is read from the final full-resolution RGB tokens.
        full_rgb = rgb_partitions[0]
        raw_depth = self.depth_output(full_rgb)
        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        depth_partitions = torch.exp(log_min_depth + normalized_depth * (log_max_depth - log_min_depth))

        return partitions_to_feature(
            depth_partitions,
            n_height=image.shape[-2],
            n_width=image.shape[-1])
