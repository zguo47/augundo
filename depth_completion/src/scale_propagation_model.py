import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as functional

sys.path.insert(0, os.path.join(
    'external_src', 'depth_completion', 'kbnet', 'src'))
from net_utils import Conv2d


def add_position_encoding(feature):
    '''Add normalized two-dimensional Fourier position information.'''
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


class SelfAttentionBlock(nn.Module):
    '''Self-attention and feedforward block.'''

    def __init__(self, n_channels, n_head):
        super(SelfAttentionBlock, self).__init__()

        self.attention_norm = nn.LayerNorm(n_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_channels,
            num_heads=n_head,
            batch_first=True)
        self.feedforward_norm = nn.LayerNorm(n_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Linear(4 * n_channels, n_channels))

    def forward(self, tokens):
        normalized_tokens = self.attention_norm(tokens)
        attended_tokens, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False)
        tokens = tokens + attended_tokens
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class ConvolutionBlock(nn.Module):
    '''Three spatial convolutions; the first optionally downsamples.'''

    def __init__(self, in_channels, out_channels, stride):
        super(ConvolutionBlock, self).__init__()

        self.convolutions = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)))

    def forward(self, feature):
        return self.convolutions(feature)


class SharedEncoder(nn.Module):
    '''Five-resolution encoder shared by concatenated RGB and sparse depth.'''

    def __init__(self, n_channels, n_head):
        super(SharedEncoder, self).__init__()

        self.level0 = ConvolutionBlock(4, n_channels, stride=1)
        self.level1 = ConvolutionBlock(n_channels, n_channels, stride=2)
        self.level2 = ConvolutionBlock(n_channels, n_channels, stride=2)

        # For the last two levels, there is one downsampling conv + 2 attention blocks.
        self.level3_downsample = Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True))
        self.level4_downsample = Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=2,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True))
        self.level3_attention = nn.ModuleList([
            SelfAttentionBlock(n_channels, n_head)
            for _ in range(2)
        ])
        self.level4_attention = nn.ModuleList([
            SelfAttentionBlock(n_channels, n_head)
            for _ in range(2)
        ])

    def attention_level(self, feature, attention_blocks):
        n_batch, n_channel, n_height, n_width = feature.shape
        feature = add_position_encoding(feature)
        tokens = feature.permute(0, 2, 3, 1).reshape(
            n_batch,
            n_height * n_width,
            n_channel)
        for attention_block in attention_blocks:
            tokens = attention_block(tokens)
        return tokens.reshape(
            n_batch,
            n_height,
            n_width,
            n_channel).permute(0, 3, 1, 2)

    def forward(self, image, sparse_depth):
        feature = torch.cat([image, sparse_depth], dim=1)

        level0 = self.level0(feature)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        level3 = self.attention_level(
            self.level3_downsample(level2),
            self.level3_attention)
        level4 = self.attention_level(
            self.level4_downsample(level3),
            self.level4_attention)

        return [level0, level1, level2, level3, level4]


def propagate_depth(depth, validity_map, propagation_logits):
    '''Multiply each local depth neighborhood by learned propagation weights.'''

    n_batch, _, n_height, n_width = depth.shape
    # Convert logits to weights
    propagation_weights = torch.softmax(propagation_logits, dim=1)

    # For every pixel, extract its 3x3 neighborhood patches then reshape to
    # B x 9 x H x W
    depth_patches = functional.unfold(
        functional.pad(depth, (1, 1, 1, 1), mode='replicate'),
        kernel_size=3).reshape(n_batch, 9, n_height, n_width)
    validity_patches = functional.unfold(
        functional.pad(validity_map, (1, 1, 1, 1), mode='replicate'),
        kernel_size=3).reshape(n_batch, 9, n_height, n_width)

    # remove invalid neighbors
    valid_weights = propagation_weights * validity_patches
    # normalize again and average valid depths
    weight_sum = torch.sum(valid_weights, dim=1, keepdim=True)
    propagated_depth = torch.sum(
        valid_weights * depth_patches,
        dim=1,
        keepdim=True) / (weight_sum + 1e-7)
    # Compute output validity
    propagated_validity = (weight_sum > 0.0).to(depth.dtype)
    return propagated_depth * propagated_validity, propagated_validity


def rejection_keep_mask(rejection_logits, temperature, training):
    '''Convert two-channel keep/reject logits into a hard keep mask.'''

    if training:
        rejection_decision = functional.gumbel_softmax(
            rejection_logits,
            tau=temperature,
            hard=True,
            dim=1)
        return rejection_decision[:, 0:1, :, :]

    rejection_decision = torch.argmax(
        rejection_logits,
        dim=1,
        keepdim=True)
    return (rejection_decision == 0).to(rejection_logits.dtype)
    # return torch.ones_like(rejection_logits[:, 0:1, :, :])


class PropagationDecoderBlock(nn.Module):
    '''Upsample and refine the preceding propagation map at one scale.'''

    def __init__(self, n_channels, gumbel_temperature):
        super(PropagationDecoderBlock, self).__init__()

        # Inputs are decoder features, encoder skip features, previous dense depth, 
        # nine upsampled propagation logits, and two rejection logits.
        # C + C + 1 + 9 + 2 = 2C + 12
        self.fusion = ConvolutionBlock(
            2 * n_channels + 12,
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.propagation_update = Conv2d(
            n_channels,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_update = Conv2d(
            n_channels,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.depth_residual = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(
            self,
            feature,
            skip,
            depth,
            propagation_logits,
            rejection_logits):
        # upsample with bilinear interpolation
        feature = functional.interpolate(
            feature,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        depth = functional.interpolate(
            depth,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        propagation_logits = functional.interpolate(
            propagation_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        rejection_logits = functional.interpolate(
            rejection_logits,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        # concat upsampled coarse context, encoder skip features, current depth, 
        # propagation logits, and rejection logits and pass through three conv.
        feature = self.fusion(torch.cat([
            feature,
            skip,
            depth,
            propagation_logits,
            rejection_logits
        ], dim=1))
        # Refine propagation weights. Conv predicts a nine channel correction.
        propagation_logits = propagation_logits + self.propagation_update(feature)
        # Refine keep/reject decisions. Conv predicts a two channel correction.
        rejection_logits = rejection_logits + self.rejection_update(feature)
        keep_mask = rejection_keep_mask(
            rejection_logits=rejection_logits,
            temperature=self.gumbel_temperature,
            training=self.training)

        # Refine depth. Conv predicts a one channel correction.
        depth = depth + self.depth_residual(feature)
        # Propagate the refined depth using the updated propagation logits.
        depth, _ = propagate_depth(
            depth=depth,
            validity_map=keep_mask,
            propagation_logits=propagation_logits)
        return feature, depth, propagation_logits, rejection_logits, keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder, bottom propagation map, and multiscale decoder.'''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature

        # Takes in RGB and sparse depth together and produce 5 feature maps.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # The bottom feature predicts a dense coarse depth. 
        self.coarse_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

        # Nine logits define one learned 3 x 3 propagation map per pixel.
        self.propagation_map = Conv2d(
            n_channels,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)

        # Two logits represent keep and reject for every coarse depth.
        self.rejection_map = Conv2d(
            n_channels,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        # Initialize with a preference for keeping depth values.
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=gumbel_temperature)
            for _ in range(4)
        ])

        # Confidence determines how strongly each original-res sparse depth 
        # constrains the learned depth.
        self.sparse_confidence = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)

        coarse_feature = features[-1]
        # Predict initial dense coarse depth. Ensure it is positive.
        depth = functional.softplus(self.coarse_depth(coarse_feature)) + self.min_predict_depth
        propagation_logits = self.propagation_map(coarse_feature)
        rejection_logits = self.rejection_map(coarse_feature) + self.rejection_bias
        keep_mask = rejection_keep_mask(
            rejection_logits=rejection_logits,
            temperature=self.gumbel_temperature,
            training=self.training)
        print('coarse keep:', keep_mask.mean().item())
        depth, _ = propagate_depth(
            depth=depth,
            validity_map=keep_mask,
            propagation_logits=propagation_logits)

        feature = coarse_feature
        for decoder_block, level in zip(
                self.decoder,
                range(3, -1, -1)):
            feature, depth, propagation_logits, rejection_logits, keep_mask = \
                decoder_block(
                feature=feature,
                skip=features[level],
                depth=depth,
                propagation_logits=propagation_logits,
                rejection_logits=rejection_logits)
            
            print(
                'level:', level,
                'keep:', keep_mask.mean().item(),
                'depth:',
                depth.min().item(),
                depth.max().item(),
                depth.std().item())
        # Apply sparse depth constraints only at their original pixel locations.
        sparse_validity = (sparse_depth > 0.0).to(sparse_depth.dtype)
        # Rejection makes a hard decision; confidence then controls the strength
        # of each sparse measurement that was kept.
        sparse_confidence = torch.sigmoid(self.sparse_confidence(feature)) * sparse_validity * keep_mask
        # depth = sparse_confidence * sparse_depth + (1.0 - sparse_confidence) * depth
        depth = sparse_validity * sparse_depth + (1.0 - sparse_validity) * depth
        depth, propagated_validity = propagate_depth(
            depth=depth,
            validity_map=keep_mask,
            propagation_logits=propagation_logits)

        depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        return depth
