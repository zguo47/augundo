import math

import torch
import torch.nn as nn

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

    def forward(self, query, context):
        n_batch, n_query, _ = query.shape
        n_context = context.shape[1]

        # normalize query and context
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
        x = query + attended

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
                nn.Conv2d(
                    input_channels if layer == 0 else n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
            for layer in range(3)
        ])

        # Every following 3x3 convolution acts on the preceding level. The
        # first block uses stride 16, then the remaining blocks use stride 2.
        self.downsample_convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=16 if level == 0 else 2,
                    padding=1),
                nn.LeakyReLU(inplace=True)
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

    # TODO: Adjust for different input image sizes so that partition height and width is always FIXED.


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
        self.n_iteration = 5

        # The RGB convolutions operate sequentially. R_0 is full resolution,
        # R_1 is downsampled by 16, and each remaining level is downsampled by 2.
        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        # Step 1: every partition below the full-resolution level performs
        # self attention independently.
        self.rgb_local_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Step 2: all tokens from all level below first partitions attend to one
        # another, providing global communication across the grid.
        self.rgb_full_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Step 3: every upper-level partition queries the corresponding
        # partition at the adjacent lower-resolution level.
        self.rgb_coarse_to_fine_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Step 4: every lower-level partition queries the corresponding
        # partition at the adjacent upper-resolution level.
        self.rgb_fine_to_coarse_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Before propagating to the full-res level, pass the features through a feedforward network.
        self.feedforwardtofullres = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_channels, 4 * n_channels),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4 * n_channels, n_channels))
            for _ in range(self.n_iteration)
        ])

        # Each final full-resolution RGB token is decoded to one depth value.
        self.depth_output = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 * n_channels, 1))
        # self.depth_output = nn.Linear(n_channels, 1)

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

    def coarse_to_fine_level(self, fine, coarse, attention_block):
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
            coarse_context)
        return updated_fine.reshape(fine.shape)

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
        rgb_partitions[-1] = self.bottom_attention(
            rgb_partitions[-1],
            self.rgb_full_attention[-1])

        for iteration in range(self.n_iteration):

            # Bottom-up travel.
            for level in range(self.n_level - 2, -1, -1):

                # Coarse to fine exchange
                rgb_partitions[level] = self.coarse_to_fine_level(
                    fine=rgb_partitions[level],
                    coarse=rgb_partitions[level + 1],
                    attention_block=self.rgb_coarse_to_fine_attention[level])
                
                if level == 0:
                    rgb_partitions[level] = self.feedforwardtofullres[iteration](rgb_partitions[level])

                # Full self attention for every level except the top level.
                if level > 0:
                    rgb_partitions[level] = self.bottom_attention(
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
                    rgb_partitions[level] = self.bottom_attention(
                        rgb_partitions[level],
                        self.rgb_full_attention[level - 1])

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