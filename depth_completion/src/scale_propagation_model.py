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


def sparse_depth_at_resolution(sparse_depth, output_size):
    '''Select original sparse measurements on the given resolution level.
    '''

    return functional.interpolate(
        sparse_depth,
        size=output_size,
        mode='nearest')


def propagate_scale(depth, sparse_depth, keep_mask, propagation_logits):
    '''Send scale corrections from accepted sparse sources to their neighbors.

    Arg(s):
        depth : torch.Tensor[float32]
            B x 1 x H x W positive dense depth before this propagation step
        sparse_depth : torch.Tensor[float32]
            B x 1 x H x W measurements sampled at this resolution
        keep_mask : torch.Tensor[float32]
            B x 1 x H x W keep decisions, zero at missing or rejected points
        propagation_logits : torch.Tensor[float32]
            B x 9 x H x W outgoing weights, indexed at the source pixel
    Returns:
        torch.Tensor[float32] : B x 1 x H x W refined dense metric depth

    '''

    n_batch, _, n_height, n_width = depth.shape
    sparse_scale = sparse_depth / depth.clamp_min(1e-3)
    # Each source has independent outgoing gates. Unlike a softmax, sigmoid
    # also allows ALL of a source's outgoing messages to be weak.
    outgoing_weights = torch.sigmoid(propagation_logits) * keep_mask
    outgoing_scale = outgoing_weights * sparse_scale

    # Each column of fold is one SOURCE pixel's 3 x 3 outgoing message patch.
    # Overlapping patches sum at their DESTINATIONS. Padding here describes
    # the fold geometry: messages outside the image are discarded, not wrapped
    # around or replicated into extra boundary measurements.
    incoming_scale = functional.fold(
        outgoing_scale.reshape(n_batch, 9, n_height * n_width),
        output_size=(n_height, n_width),
        kernel_size=3,
        padding=1)
    incoming_weight = functional.fold(
        outgoing_weights.reshape(n_batch, 9, n_height * n_width),
        output_size=(n_height, n_width),
        kernel_size=3,
        padding=1)
    correction = (1.0 + incoming_scale) / (1.0 + incoming_weight)
    return depth * correction


def rejection_keep_mask(rejection_logits, sparse_depth, temperature, training):
    '''Make keep/reject decisions for this level's sparse depth.
    '''

    if training:
        rejection_decision = functional.gumbel_softmax(
            rejection_logits,
            tau=temperature,
            hard=True,
            dim=1)
        keep_mask = rejection_decision[:, 0:1, :, :]
    else:
        rejection_decision = torch.argmax(
            rejection_logits,
            dim=1,
            keepdim=True)
        keep_mask = (rejection_decision == 0).to(sparse_depth.dtype)

    return keep_mask * (sparse_depth > 0.0).to(sparse_depth.dtype)


class PropagationDecoderBlock(nn.Module):
    '''Upsample dense metric depth, then apply sparse scale corrections.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection):
        super(PropagationDecoderBlock, self).__init__()

        # Carry only features and dense metric depth from the previous level.
        # Add this level's encoder skip and nearest-sampled sparse measurements:
        # C + C + 1 + 1 = 2C + 2 input channels.
        self.fusion = ConvolutionBlock(
            2 * n_channels + 2,
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.depth_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)

        # Heads see features, current metric depth, and current sparse depth.
        # Both heads are freshly evaluated at each level; neither is a residual
        # added to an interpolated map from a preceding level.
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

    def forward(self, feature, skip, depth, sparse_depth):
        # Match the actual skip dimensions, including for odd-sized inputs.
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
        sparse_depth = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=skip.shape[-2:])

        feature = self.fusion(torch.cat([
            feature,
            skip,
            depth,
            sparse_depth
        ], dim=1))
        # A positive multiplicative refinement learns detail before applying
        # metric constraints. There is no whole-image renormalization here:
        # this tensor already represents metric depth inherited from below.
        depth = depth * torch.exp(torch.tanh(self.depth_update(feature)))

        propagation_input = torch.cat([feature, depth, sparse_depth], dim=1)
        if self.use_outlier_rejection:
            rejection_logits = \
                self.rejection_map(propagation_input) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                sparse_depth=sparse_depth,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            rejection_logits = torch.zeros_like(sparse_depth).repeat(1, 2, 1, 1)
            keep_mask = (sparse_depth > 0.0).to(sparse_depth.dtype)

        propagation_logits = self.propagation_map(propagation_input)
        depth = propagate_scale(
            depth=depth,
            sparse_depth=sparse_depth,
            keep_mask=keep_mask,
            propagation_logits=propagation_logits)

        # Rejected points never enter the explicit scale messages and there is
        # no subsequent sparse-depth overwrite. Original sparse measurements
        # are reconsidered independently at the next level, as requested.
        return feature, depth, propagation_logits, rejection_logits, keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Five-resolution encoder with hierarchical source-directed scale updates.

    The bottom level predicts a positive depth shape, establishes metric scale
    from its accepted sparse measurements, and sends local scale corrections.
    Each decoder block upsamples the resulting dense metric depth, refines it
    using encoder features, and repeats rejection and source-directed correction.
    Only final full-resolution depth is supervised by the existing wrapper.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0,
                 use_outlier_rejection=True):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection

        # Takes in RGB and sparse depth together and produce 5 feature maps.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # Positive bottom prediction supplies both a normalized scaleless shape
        # and a learned metric fallback if nearest sampling leaves no anchors.
        self.coarse_scaleless_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.propagation_map = Conv2d(
            n_channels + 2,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))
        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=gumbel_temperature,
                use_outlier_rejection=use_outlier_rejection)
            for _ in range(4)
        ])

    def forward(self, image, sparse_depth):
        features = self.encoder(image, sparse_depth)
        coarse_feature = features[-1]
        coarse_sparse_depth = sparse_depth_at_resolution(
            sparse_depth=sparse_depth,
            output_size=coarse_feature.shape[-2:])

        coarse_depth = functional.softplus(
            self.coarse_scaleless_depth(coarse_feature)) + 1e-3
        fallback_scale = torch.mean(coarse_depth, dim=[2, 3], keepdim=True)
        scaleless_depth = coarse_depth / fallback_scale
        propagation_input = torch.cat([
            coarse_feature,
            scaleless_depth,
            coarse_sparse_depth
        ], dim=1)

        # Reject BEFORE computing either the global metric initialization or
        # local messages. Rejected measurements contribute to neither operation.
        if self.use_outlier_rejection:
            rejection_logits = \
                self.rejection_map(propagation_input) + self.rejection_bias
            keep_mask = rejection_keep_mask(
                rejection_logits=rejection_logits,
                sparse_depth=coarse_sparse_depth,
                temperature=self.gumbel_temperature,
                training=self.training)
        else:
            keep_mask = (coarse_sparse_depth > 0.0).to(sparse_depth.dtype)

        # Each accepted measurement proposes a metric scale d_sparse / shape.
        # One additional vote comes from the learned fallback scale. This makes
        # empty/all-rejected coarse levels finite and trainable without ever
        # bringing rejected measurements back into the initialization.
        sparse_scale = coarse_sparse_depth / scaleless_depth.clamp_min(1e-3)
        scale = (
            torch.sum(keep_mask * sparse_scale, dim=[2, 3], keepdim=True) +
            fallback_scale) / (
                torch.sum(keep_mask, dim=[2, 3], keepdim=True) + 1.0)
        depth = scaleless_depth * scale
        propagation_logits = self.propagation_map(propagation_input)
        depth = propagate_scale(
            depth=depth,
            sparse_depth=coarse_sparse_depth,
            keep_mask=keep_mask,
            propagation_logits=propagation_logits)

        feature = coarse_feature
        for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
            # Positional inputs are visible to debug forward hooks on PyTorch
            # 1.10, whose hooks do not receive keyword arguments.
            feature, depth, _, _, _ = decoder_block(
                feature,
                features[level],
                depth,
                sparse_depth)

        # Keep the existing output bounds and single final supervised loss.
        return torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
