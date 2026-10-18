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


def sparse_depth_at_resolution(sparse_depth, output_size, return_indices=False):
    '''Sparse depth at requested resolution.
    '''

    # Pool only the positive-entries to locate a valid measurement.
    has_measurement, measurement_indices = functional.adaptive_max_pool2d(
        (sparse_depth > 0.0).to(sparse_depth.dtype),
        output_size=output_size,
        return_indices=True)
    selected_depth = torch.gather(
        sparse_depth.flatten(2),
        dim=2,
        index=measurement_indices.flatten(2))
    selected_depth = selected_depth.reshape_as(has_measurement) * has_measurement
    # Debug comparisons use the selected original pixel, not pooled ground truth.
    if return_indices:
        return selected_depth, measurement_indices
    return selected_depth


def normalize_rejection_input(feature, depth, sparse_depth):
    n_channel = feature.shape[1]
    feature = functional.layer_norm(
        feature.permute(0, 2, 3, 1),
        normalized_shape=(n_channel,)).permute(0, 3, 1, 2)
    return torch.cat([feature, depth, sparse_depth], dim=1)


def geometry_at_sparse_points(scaleless_depth, measurement_indices, input_size):
    '''Sample this level's geometry at the selected original-image pixels.'''
    n_input_height, n_input_width = input_size
    # The sampler returns flattened indices into the original sparse image,
    # not coordinates of coarse-cell centers. Keep the per-image indices.
    y = (measurement_indices // n_input_width).to(scaleless_depth.dtype)
    x = (measurement_indices % n_input_width).to(scaleless_depth.dtype)
    y = 2.0 * y / max(n_input_height - 1, 1) - 1.0
    x = 2.0 * x / max(n_input_width - 1, 1) - 1.0
    grid = torch.cat([x, y], dim=1).permute(0, 2, 3, 1)

    # This is equivalent to upsampling Z to the input size and gathering the
    # selected pixels, without allocating a full-resolution map at each level.
    # align_corners matches the existing decoder/loss interpolation convention.
    return functional.grid_sample(
        scaleless_depth.detach(),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)


def sparse_depth_to_points(sparse_depth):
    '''Keep every positive measurement with its original batch and pixel index.'''
    locations = torch.nonzero(sparse_depth[:, 0] > 0.0, as_tuple=False)
    batch, y, x = locations.unbind(dim=1)
    return dict(
        batch=batch,
        y=y,
        x=x,
        depth=sparse_depth[batch, 0, y, x],
        input_size=sparse_depth.shape[-2:])


def point_coordinates(points, output_size, dtype):
    '''Map original pixel coordinates continuously into this resolution.'''
    n_height, n_width = output_size
    n_input_height, n_input_width = points['input_size']
    y = points['y'].to(dtype) * (n_height - 1) / max(n_input_height - 1, 1)
    x = points['x'].to(dtype) * (n_width - 1) / max(n_input_width - 1, 1)
    return y, x


def features_at_points(feature, points):
    '''Bilinearly read features at all original points, without rasterizing them.

    The result is P x C, including multiple points in the same coarse cell.
    Four-corner gathers implement align_corners=True and also support empty
    point sets, unequal point counts per image, and one-pixel spatial axes.
    '''
    n_height, n_width = feature.shape[-2:]
    y, x = point_coordinates(points, (n_height, n_width), feature.dtype)
    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp_max(n_height - 1), (x0 + 1).clamp_max(n_width - 1)
    dy, dx = (y - y0)[:, None], (x - x0)[:, None]
    batch = points['batch']
    return \
        (1.0 - dy) * (1.0 - dx) * feature[batch, :, y0, x0] + \
        (1.0 - dy) * dx * feature[batch, :, y0, x1] + \
        dy * (1.0 - dx) * feature[batch, :, y1, x0] + \
        dy * dx * feature[batch, :, y1, x1]


def point_scale_observations(
        feature,
        scaleless_depth,
        scale,
        points,
        rejection_map,
        rejection_bias,
        use_outlier_rejection,
        gumbel_temperature,
        training,
        max_predict_depth,
        epsilon=1e-7):
    '''Form independent scale observations and fresh rejection for every point.'''
    point_geometry = features_at_points(scaleless_depth.detach(), points)[:, 0]
    sparse_scale = points['depth'] / point_geometry.clamp_min(epsilon)
    n_point = sparse_scale.numel()
    rejection_logits = feature.new_zeros(1, 2, 1, n_point)
    if n_point == 0 or not use_outlier_rejection:
        return sparse_scale, torch.ones_like(sparse_scale), rejection_logits

    # A 1 x 1 head acts on each point independently. Neighboring entries in
    # this packed list need not be spatial neighbors and must not be convolved.
    point_feature = features_at_points(feature, points).transpose(0, 1)[None, :, None, :]
    if scale is None:
        # Preserve the bottom head's scaleless context before metric initialization.
        point_depth = point_geometry
    else:
        point_depth = features_at_points(scaleless_depth.detach() * scale, points)[:, 0]
        point_depth = point_depth / max_predict_depth
    point_sparse = points['depth'][None, None, None, :]
    rejection_input = normalize_rejection_input(
        feature=point_feature,
        depth=point_depth[None, None, None, :],
        sparse_depth=point_sparse / max_predict_depth)
    rejection_logits = rejection_map(rejection_input) + rejection_bias
    keep_mask = rejection_keep_mask(
        rejection_logits=rejection_logits,
        sparse_depth=point_sparse,
        temperature=gumbel_temperature,
        training=training).reshape(-1)
    return sparse_scale, keep_mask, rejection_logits


class SparseScaleInjection(nn.Module):
    '''Let nearby decoder pixels receive individual original-point observations.'''

    def __init__(self, n_channels):
        super(SparseScaleInjection, self).__init__()

        n_point_channels = max(1, n_channels // 4)
        self.query_projection = Conv2d(
            n_channels, n_point_channels, kernel_size=1, activation_func=None)
        self.source_projection = Conv2d(
            n_channels, n_point_channels, kernel_size=1, activation_func=None)
        self.n_point_channels = n_point_channels

    def forward(
            self, scale, feature, source_feature, scaleless_depth,
            points, sparse_scale, keep_mask):
        n_batch, _, n_height, n_width = scale.shape
        if sparse_scale.numel() == 0:
            return scale, torch.zeros_like(scale)

        # Full-resolution encoder features identify each original measurement.
        # They remain separate even when their coordinates share a coarse cell.
        source = features_at_points(source_feature, points)
        source = functional.layer_norm(source, (source.shape[1],))
        source = self.source_projection(source.transpose(0, 1)[None, :, None, :])
        source = source.reshape(self.n_point_channels, -1).transpose(0, 1)
        query = functional.layer_norm(
            feature.permute(0, 2, 3, 1), (feature.shape[1],)).permute(0, 3, 1, 2)
        query = self.query_projection(query)

        # Each source considers only nine nearby destination pixels at this
        # level. Work scales with 9P, not P times the number of image pixels.
        y, x = point_coordinates(points, (n_height, n_width), scale.dtype)
        offsets = torch.arange(-1, 2, device=scale.device)
        target_y = y.round().long()[:, None] + offsets.repeat_interleave(3)[None, :]
        target_x = x.round().long()[:, None] + offsets.repeat(3)[None, :]
        inside = (target_y >= 0) & (target_y < n_height) & \
            (target_x >= 0) & (target_x < n_width)
        safe_y = target_y.clamp(0, n_height - 1)
        safe_x = target_x.clamp(0, n_width - 1)
        destination = query[points['batch'][:, None], :, safe_y, safe_x]
        similarity = (destination * source[:, None, :]).sum(dim=2) / math.sqrt(
            self.n_point_channels)

        # Actual subpixel offsets, not just a coarse-cell index, determine
        # spatial support. Feature compatibility lets destinations distinguish
        # sources on different surfaces; rejected sources contribute no depth.
        squared_distance = (target_y.to(scale.dtype) - y[:, None]).square() + \
            (target_x.to(scale.dtype) - x[:, None]).square()
        weights = torch.sigmoid(similarity) * torch.exp(-0.5 * squared_distance)
        weights = weights * inside.to(scale.dtype) * keep_mask[:, None]
        indices = points['batch'][:, None] * (n_height * n_width) + \
            safe_y * n_width + safe_x
        incoming_weight = scale.new_zeros(n_batch * n_height * n_width).scatter_add(
            0, indices.reshape(-1), weights.reshape(-1)).reshape_as(scale)
        incoming_scale = scale.new_zeros(n_batch * n_height * n_width).scatter_add(
            0, indices.reshape(-1), (weights * sparse_scale[:, None]).reshape(-1))
        incoming_scale = incoming_scale.reshape_as(scale)

        # The inherited/learned map is a prior with unit weight, not a raster
        # replacing the original sources. Fine levels query those sources anew.
        scale = (scale + incoming_scale) / (1.0 + incoming_weight)
        return scale, incoming_weight


def propagate_scale(
        scale,
        propagation_logits,
        n_propagation_iterations=1):
    '''Relay the persistent scale map after individual-source injection.

    Arg(s):
        scale : torch.Tensor[float32]
            B x 1 x H x W inherited and learned positive scale map
        propagation_logits : torch.Tensor[float32]
            B x 9 x H x W outgoing weights, indexed at the source pixel
        n_propagation_iterations : int
            number of neighboring exchanges at this resolution
    Returns:
            torch.Tensor[float32] : B x 1 x H x W refined scale map

    '''

    n_batch, _, n_height, n_width = scale.shape
    outgoing_weights = torch.sigmoid(propagation_logits)

    # Each column of fold is one source pixel's 3 x 3 outgoing message patch.
    # Overlapping patches sum at their destinations. 
    incoming_weight = functional.fold(
        outgoing_weights.reshape(n_batch, 9, n_height * n_width),
        output_size=(n_height, n_width),
        kernel_size=3,
        padding=1)
    for _ in range(n_propagation_iterations):
        outgoing_scale = outgoing_weights * scale
        incoming_scale = functional.fold(
            outgoing_scale.reshape(n_batch, 9, n_height * n_width),
            output_size=(n_height, n_width),
            kernel_size=3,
            padding=1)

        # Average the current scale with incoming messages. The learned center
        # connection is retained in addition to this fixed self-retention term.
        scale = (scale + incoming_scale) / (1.0 + incoming_weight)
    return scale


class ScalePropagation(nn.Module):
    '''Expose propagation as a module so optional debug hooks see its inputs.'''

    def __init__(self, n_propagation_iterations):
        super(ScalePropagation, self).__init__()
        self.n_propagation_iterations = n_propagation_iterations

    def forward(
            self, scale, scaleless_depth, points, keep_mask, logits,
            sparse_scale):
        # Point observations remain explicit for debug hooks. Injection already
        # took place individually; this loop only relays the resulting scale.
        return propagate_scale(scale, logits, self.n_propagation_iterations)


def normalize_scaleless_depth(depth):
    '''Fix the geometry's global magnitude; the scale map carries metric units.'''
    return depth / depth.mean(dim=[2, 3], keepdim=True).clamp_min(1e-7)


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
    '''Refine scaleless geometry and the inherited scale map at a finer level.'''

    def __init__(
            self,
            n_channels,
            gumbel_temperature,
            use_outlier_rejection,
            n_propagation_iterations=2,
            max_predict_depth=8.0):
        super(PropagationDecoderBlock, self).__init__()

        # Carry features, geometry, and scale with the encoder skip. Sparse
        # constraints stay in their original point list, not a pooled channel.
        self.fusion = ConvolutionBlock(
            2 * n_channels + 2,
            n_channels,
            stride=1)
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.n_propagation_iterations = n_propagation_iterations
        self.max_predict_depth = max_predict_depth
        self.depth_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        # Start by retaining the inherited scale rather than changing it.
        nn.init.zeros_(self.scale_update.conv.weight)
        self.scale_propagation = ScalePropagation(n_propagation_iterations)
        self.sparse_injection = SparseScaleInjection(n_channels)

        # Dense relay uses features and metric depth. Rejection is evaluated
        # separately on every original measurement with its own sparse value.
        self.propagation_map = Conv2d(
            n_channels + 1,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=1,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))

    def forward(self, feature, skip, scaleless_depth, scale, points, source_feature):

        feature = functional.interpolate(
            feature,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        scaleless_depth = functional.interpolate(
            scaleless_depth,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        scale = functional.interpolate(
            scale,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)

        feature = self.fusion(torch.cat([
            feature,
            skip,
            # Shared features can train under both objectives, but map inputs
            # are context only: SI losses must not train preceding scale heads.
            scaleless_depth.detach(),
            scale.detach()
        ], dim=1))
        # Geometry stays positive and mean-one at every resolution. Only the
        # SI objectives directly train this prediction and its preceding maps.
        scaleless_depth = normalize_scaleless_depth(
            scaleless_depth * torch.exp(torch.tanh(self.depth_update(feature))))
        scale = scale * torch.exp(torch.tanh(self.scale_update(feature)))
        # Metric heads see current metric context without a gradient into Z.
        depth = scaleless_depth.detach() * scale

        sparse_scale, keep_mask, rejection_logits = point_scale_observations(
            feature, scaleless_depth, scale, points,
            self.rejection_map, self.rejection_bias,
            self.use_outlier_rejection, self.gumbel_temperature,
            self.training, self.max_predict_depth)
        propagation_logits = self.propagation_map(torch.cat([feature, depth], dim=1))
        scale, _ = self.sparse_injection(
            scale, feature, source_feature, scaleless_depth,
            points, sparse_scale, keep_mask)
        scale = self.scale_propagation(
            scale, scaleless_depth, points, keep_mask, propagation_logits, sparse_scale)

        # No exact measurement restoration: the next decoder receives learned,
        # propagated scale, and makes its own fresh rejection decisions.
        return feature, scaleless_depth, scale, propagation_logits, rejection_logits, keep_mask


class ScalePropagationDepthModel(nn.Module):
    '''Shared encoder with persistent geometry and metric scale at five levels.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 gumbel_temperature=1.0,
                 use_outlier_rejection=True,
                 n_propagation_iterations=(1, 2, 3, 4, 6)):
        super(ScalePropagationDepthModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.gumbel_temperature = gumbel_temperature
        self.use_outlier_rejection = use_outlier_rejection
        self.uses_sparse_points = True
        # Iteration counts are ordered from the bottom to full resolution.
        # Coarse scales are denser; finer scales receive more exchanges.
        self.n_propagation_iterations = n_propagation_iterations

        # Takes in RGB and sparse depth together and produce 5 feature maps.
        self.encoder = SharedEncoder(
            n_channels=n_channels,
            n_head=n_head)

        # Positive bottom prediction supplies both a normalized scaleless depth
        self.coarse_scaleless_depth = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.scale_update = Conv2d(
            n_channels,
            1,
            kernel_size=3,
            stride=1,
            activation_func=None)
        nn.init.zeros_(self.scale_update.conv.weight)
        self.scale_propagation = ScalePropagation(n_propagation_iterations[0])
        self.sparse_injection = SparseScaleInjection(n_channels)
        self.propagation_map = Conv2d(
            n_channels + 1,
            9,
            kernel_size=3,
            stride=1,
            activation_func=None)
        self.rejection_map = Conv2d(
            n_channels + 2,
            2,
            kernel_size=1,
            stride=1,
            activation_func=None)
        self.rejection_bias = nn.Parameter(
            torch.tensor([2.0, 0.0]).reshape(1, 2, 1, 1))
        self.decoder = nn.ModuleList([
            PropagationDecoderBlock(
                n_channels=n_channels,
                gumbel_temperature=gumbel_temperature,
                use_outlier_rejection=use_outlier_rejection,
                n_propagation_iterations=n_iteration,
                max_predict_depth=max_predict_depth)
            for n_iteration in n_propagation_iterations[1:]
        ])

    def forward(self, image, sparse_depth, return_all_outputs=False):
        features = self.encoder(image, sparse_depth)
        coarse_feature = features[-1]
        # Build once and retain every measurement through all decoder levels.
        points = sparse_depth_to_points(sparse_depth)

        coarse_depth = functional.softplus(
            self.coarse_scaleless_depth(coarse_feature)) + 1e-3
        fallback_scale = torch.mean(coarse_depth, dim=[2, 3], keepdim=True)
        scaleless_depth = normalize_scaleless_depth(coarse_depth)
        metric_geometry = scaleless_depth.detach()
        # Reject each original point before global initialization or injection.
        sparse_scale, keep_mask, _ = point_scale_observations(
            coarse_feature, scaleless_depth, None, points,
            self.rejection_map, self.rejection_bias,
            self.use_outlier_rejection, self.gumbel_temperature,
            self.training, self.max_predict_depth, epsilon=1e-3)
        n_batch = image.shape[0]
        scale_sum = image.new_zeros(n_batch).scatter_add(
            0, points['batch'], keep_mask * sparse_scale)
        keep_sum = image.new_zeros(n_batch).scatter_add(0, points['batch'], keep_mask)
        scale = (scale_sum[:, None, None, None] + fallback_scale.detach()) / (
            keep_sum[:, None, None, None] + 1.0)
        # Initialize a dense constant scale from the existing global estimate,
        # then allow the head and sparse-guided propagation to refine it.
        scale = scale.expand_as(scaleless_depth)
        scale = scale * torch.exp(torch.tanh(self.scale_update(coarse_feature)))
        propagation_logits = self.propagation_map(torch.cat([
            coarse_feature, metric_geometry], dim=1))
        scale, _ = self.sparse_injection(
            scale, coarse_feature, features[0], scaleless_depth,
            points, sparse_scale, keep_mask)
        scale = self.scale_propagation(
            scale, metric_geometry, points, keep_mask, propagation_logits, sparse_scale)

        feature = coarse_feature
        scaleless_depths = [scaleless_depth]
        scale_maps = [scale]
        for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
            feature, scaleless_depth, scale, _, _, _ = decoder_block(
                feature,
                features[level],
                scaleless_depth,
                scale,
                points,
                features[0])
            scaleless_depths.append(scaleless_depth)
            scale_maps.append(scale)

        # Clamp before output.
        depth = torch.clamp(
            scaleless_depth * scale,
            min=self.min_predict_depth,
            max=self.max_predict_depth)
        if not return_all_outputs:
            return depth
        # Full resolution first. Explicit outputs also work with DataParallel;
        # no retained per-forward graph or replica-local cache is needed.
        scaleless_depths = scaleless_depths[::-1]
        scale_maps = scale_maps[::-1]
        return dict(
            depth=depth,
            scaleless_depths=scaleless_depths,
            scale_maps=scale_maps,
            metric_depths_for_loss=[
                geometry.detach() * metric_scale
                for geometry, metric_scale in zip(
                    scaleless_depths, scale_maps)])
