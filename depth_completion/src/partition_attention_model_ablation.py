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
    '''Creates eight sequential spatial levels from RGB.'''

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
                nn.LeakyReLU(inplace=True))
            for layer in range(3)
        ])

        # Every following block acts on the preceding level and reduces its
        # height and width by 2.
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
                    stride=2,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
            for _ in range(7)
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
    RGB forms an eight-level pyramid with progressive 2x downsampling. The four
    coarse levels perform full self-attention independently. A U-Net-style
    decoder upsamples from the bottom, concatenates the feature at every upper
    level, and returns to full resolution.
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
        self.n_level = 8
        self.first_attention_level = 4

        # R_0 is full resolution and every later level is downsampled by 2.
        # For a 512x512 input, the levels are 512, 256, 128, 64, 32, 16, 8 and 4.
        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        # The 32x32, 16x16, 8x8 and 4x4 levels each perform independent full
        # self-attention. Full attention is not applied to the added high-
        # resolution levels because its memory grows quadratically in the
        # number of spatial tokens.
        self.rgb_full_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.first_attention_level, self.n_level)
        ])

        # Every decoder stage doubles the resolution until returning to R_0.
        self.up_convolutions = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    n_channels,
                    n_channels,
                    kernel_size=2,
                    stride=2),
                nn.LeakyReLU(inplace=True))
            for _ in range(self.n_level - 2, -1, -1)
        ])

        # Fuse to restore to C channels
        self.fusion_convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    2 * n_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True))
            for _ in range(self.n_level - 1)
        ])

        # Every final full-resolution token passes through the same
        # feedforward network to produce one raw depth value.
        self.depth_output = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 * n_channels, 1))

    def full_attention(self, feature, attention_block):
        '''Full self-attention across every spatial token in one level.'''
        n_batch, n_channel, n_height, n_width = feature.shape
        tokens = feature.permute(0, 2, 3, 1).reshape(
            n_batch,
            n_height * n_width,
            n_channel)
        tokens = attention_block(tokens, tokens)
        return tokens.reshape(
            n_batch,
            n_height,
            n_width,
            n_channel).permute(0, 3, 1, 2)

    def forward(self, image):
        # Create R_0 through R_7 sequentially from RGB using progressive 2x
        # downsampling.
        rgb_features = self.rgb_pyramid(image)
        rgb_features = [
            add_position_encoding(feature)
            for feature in rgb_features
        ]

        # Full self-attention is independent at each feasible coarse resolution.
        for level, attention_block in zip(
                range(self.first_attention_level, self.n_level),
                self.rgb_full_attention):
            rgb_features[level] = self.full_attention(
                rgb_features[level],
                attention_block)

        # Start at R_7. Each stage upsamples the current decoder feature,
        # concatenates it with the corresponding encoder feature, and fuses
        # the 2C concatenated channels back into C channels.
        feature = rgb_features[-1]
        for level, up_convolution, fusion_convolution in zip(
                range(self.n_level - 2, -1, -1),
                self.up_convolutions,
                self.fusion_convolutions):
            feature = up_convolution(feature)
            feature = torch.cat([
                feature,
                rgb_features[level]
            ], dim=1)
            feature = fusion_convolution(feature)

        # Convert the full-resolution feature map to a token sequence so the
        # final feedforward network operates independently on every pixel.
        full_rgb = feature.permute(0, 2, 3, 1)
        raw_depth = self.depth_output(full_rgb)
        raw_depth = raw_depth.permute(0, 3, 1, 2)

        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        return torch.exp(
            log_min_depth +
            normalized_depth * (log_max_depth - log_min_depth))
