import math

import torch
import torch.nn as nn


class AttentionUpdate(nn.Module):
    '''Update with attention.'''
    def __init__(self, n_channels, n_head):
        super(AttentionUpdate, self).__init__()

        self.n_head = n_head
        self.head_channels = n_channels // n_head

        # Q, K and V
        self.query_projection = nn.Linear(n_channels, n_channels)
        self.key_projection = nn.Linear(n_channels, n_channels)
        self.value_projection = nn.Linear(n_channels, n_channels)

        # Attention result
        self.output_projection = nn.Linear(n_channels, n_channels)

    def forward(self, query, context):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

        # Q: N x L_q x C -> N x heads x L_q x D_head
        projected_query = self.query_projection(query)
        projected_query = projected_query.reshape(
            n_batch,
            n_query,
            self.n_head,
            self.head_channels)
        projected_query = projected_query.permute(0, 2, 1, 3)

        # K and V: N x L_k x C -> N x heads x L_k x D_head
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
    '''Creates four sequential RGB levels for idea two.'''

    def __init__(self, input_channels, n_channels):
        super(ConvolutionPyramid, self).__init__()

        # Three 1x1 convolutions preserve the input resolution and create R_0.
        self.full_resolution_convolutions = nn.ModuleList([
            nn.Conv2d(
                input_channels if layer == 0 else n_channels,
                n_channels,
                kernel_size=1,
                stride=1)
            for layer in range(3)
        ])

        # R_1 and R_2 each use two 3x3 convolutions and one 7x7 convolution.
        # The first 3x3 convolution halves the preceding spatial resolution.
        self.level_1_convolutions = nn.ModuleList([
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=7,
                stride=1,
                padding=3)
        ])
        self.level_2_convolutions = nn.ModuleList([
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=7,
                stride=1,
                padding=3)
        ])

        # The bottom block operates on R_2 sequentially. For a 448x640 input,
        # its final 19x5 convolution with stride 3x5 maps 112x160 to 32x32.
        # With 16x16 partitions, the bottom level therefore has a 2x2 grid.
        self.bottom_convolutions = nn.ModuleList([
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=(19, 5),
                stride=(3, 5))
        ])

    def apply_convolutions(self, feature, convolutions):
        for convolution in convolutions:
            feature = convolution(feature)
        return feature

    def forward(self, image):
        level_0 = self.apply_convolutions(
            image,
            self.full_resolution_convolutions)
        level_1 = self.apply_convolutions(
            level_0,
            self.level_1_convolutions)
        level_2 = self.apply_convolutions(
            level_1,
            self.level_2_convolutions)
        level_3 = self.apply_convolutions(
            level_2,
            self.bottom_convolutions)

        return [level_0, level_1, level_2, level_3]


def feature_to_partitions(feature, partition_size):
    '''Divide a feature map into non-overlapping k x k partitions.'''
    n_batch, n_channel, n_height, n_width = feature.shape
    n_grid_height = n_height // partition_size
    n_grid_width = n_width // partition_size

    partitions = feature.reshape(
        n_batch,
        n_channel,
        n_grid_height,
        partition_size,
        n_grid_width,
        partition_size)
    partitions = partitions.permute(0, 2, 4, 3, 5, 1)
    return partitions.reshape(
        n_batch,
        n_grid_height,
        n_grid_width,
        partition_size * partition_size,
        n_channel)


def partitions_to_feature(partitions, n_height, n_width):
    '''Reverse partitions to a feature map.'''
    n_batch, n_grid_height, n_grid_width, n_token, n_channel = \
        partitions.shape
    partition_size = int(math.sqrt(n_token))

    feature = partitions.reshape(
        n_batch,
        n_grid_height,
        n_grid_width,
        partition_size,
        partition_size,
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


class PartitionAttentionDepthModel(nn.Module):
    '''
    RGB forms a four-level pyramid. Every level performs local attention, 
    the bottom level performs full attention, and every lower level updates 
    every higher-resolution level.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 partition_size=16):
        super(PartitionAttentionDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.n_channels = n_channels
        self.n_level = 4
        self.partition_size = partition_size

        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        # Every partition at every resolution performs self-attention
        # independently from the other partitions at that level.
        self.rgb_local_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level)
        ])

        # All tokens from all four bottom partitions perform full self attention.
        self.rgb_bottom_attention = AttentionUpdate(n_channels, n_head)

        # A separate attention block is used for each directed lower-to-upper
        # connection: 3->2, 3->1, 3->0, 2->1, 2->0 and 1->0.
        self.rgb_lower_to_upper_attention = nn.ModuleDict({
            '{}_to_{}'.format(source_level, target_level):
                AttentionUpdate(n_channels, n_head)
            for source_level in range(1, self.n_level)
            for target_level in range(source_level)
        })

        self.depth_output = nn.Linear(n_channels, 1)

    def local_attention(self, partitions, attention_blocks):
        '''Local self-attention within every partition at every level.'''
        outputs = []
        for level_partitions, attention_block in zip(
                partitions,
                attention_blocks):
            n_batch, n_grid_height, n_grid_width, n_token, n_channel = \
                level_partitions.shape
            tokens = level_partitions.reshape(
                n_batch * n_grid_height * n_grid_width,
                n_token,
                n_channel)
            tokens = attention_block(tokens, tokens)
            outputs.append(tokens.reshape(level_partitions.shape))

        return outputs

    def full_attention(self, partitions, attention_block):
        '''Full self-attention across all partitions in one level.'''
        n_batch, n_grid_height, n_grid_width, n_token, n_channel = \
            partitions.shape
        tokens = partitions.reshape(
            n_batch,
            n_grid_height * n_grid_width * n_token,
            n_channel)
        tokens = attention_block(tokens, tokens)
        return tokens.reshape(partitions.shape)

    def corresponding_context(self, source, target_grid_height, target_grid_width):
        '''Route each target partition to the source partition covering its center.'''
        n_batch, source_grid_height, source_grid_width, n_token, n_channel = \
            source.shape

        target_y = torch.arange(
            target_grid_height,
            device=source.device)
        target_x = torch.arange(
            target_grid_width,
            device=source.device)
        source_y = torch.floor(
            (target_y + 0.5) * source_grid_height / target_grid_height).long()
        source_x = torch.floor(
            (target_x + 0.5) * source_grid_width / target_grid_width).long()

        context = source[:, source_y[:, None], source_x[None, :], :, :]
        return context.reshape(
            n_batch * target_grid_height * target_grid_width,
            n_token,
            n_channel)

    def lower_to_upper_level(self, upper, lower, attention_block):
        '''Update an upper level from spatially corresponding lower partitions.'''
        n_batch, upper_grid_height, upper_grid_width, n_token, n_channel = \
            upper.shape
        upper_queries = upper.reshape(
            n_batch * upper_grid_height * upper_grid_width,
            n_token,
            n_channel)
        lower_context = self.corresponding_context(
            lower,
            upper_grid_height,
            upper_grid_width)
        updated_upper = attention_block(
            upper_queries,
            lower_context)
        return updated_upper.reshape(upper.shape)

    def forward(self, image):
        # Create R_0, R_1, R_2 and R_3 sequentially from RGB.
        rgb_features = self.rgb_pyramid(image)
        rgb_features = [
            add_position_encoding(feature)
            for feature in rgb_features
        ]

        # Every level is divided into non-overlapping 16x16 partitions.
        rgb_partitions = [
            feature_to_partitions(feature, self.partition_size)
            for feature in rgb_features
        ]

        # First perform local self-attention independently at every level.
        rgb_partitions = self.local_attention(
            rgb_partitions,
            self.rgb_local_attention)

        # The 2x2 bottom grid then performs full attention across all four
        # partitions and all tokens contained in those partitions.
        rgb_partitions[-1] = self.full_attention(
            rgb_partitions[-1],
            self.rgb_bottom_attention)

        # Each lower level updates every level above it. Because sources are
        # processed from bottom to top, an updated intermediate level passes
        # both its own information and information received from lower levels.
        for source_level in range(self.n_level - 1, 0, -1):
            for target_level in range(source_level - 1, -1, -1):
                attention_name = '{}_to_{}'.format(
                    source_level,
                    target_level)
                rgb_partitions[target_level] = self.lower_to_upper_level(
                    upper=rgb_partitions[target_level],
                    lower=rgb_partitions[source_level],
                    attention_block=self.rgb_lower_to_upper_attention[
                        attention_name])

        # Decode every updated full-resolution token directly to metric depth.
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
