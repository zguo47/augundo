import torch
import torch.nn as nn
import torch.nn.functional as functional

from scale_propagation_model import (
    Conv2d,
    ConvolutionBlock,
    ScalePropagation,
    SharedEncoder,
    normalize_scaleless_depth,
    sparse_depth_at_resolution)


class GeometryDecoderBlock(nn.Module):
    '''Upsample RGB features and fuse the corresponding encoder skip.'''

    def __init__(self, n_channels):
        super(GeometryDecoderBlock, self).__init__()
        self.fusion = ConvolutionBlock(2 * n_channels, n_channels, stride=1)

    def forward(self, feature, skip):
        feature = functional.interpolate(
            feature, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return self.fusion(torch.cat([feature, skip], dim=1))


class RGBGeometryModel(nn.Module):
    '''Predict one positive full-resolution geometry using RGB alone.

    No sparse measurements or scale maps enter this network. Its decoder
    features supply RGB guidance to the separate scale-propagation decoder.
    The five geometries used by propagation are resized copies of this ONE
    final prediction, not independent predictions with different meanings.
    '''

    def __init__(self, n_channels=32, n_head=4, encoder_type='conv'):
        super(RGBGeometryModel, self).__init__()
        self.encoder = SharedEncoder(
            n_channels=n_channels, n_head=n_head, encoder_type=encoder_type,
            use_sparse_depth_features=False)
        self.decoder = nn.ModuleList([
            GeometryDecoderBlock(n_channels) for _ in range(4)
        ])
        self.depth_output = nn.Sequential(
            Conv2d(
                n_channels, n_channels, kernel_size=3, stride=1,
                activation_func=nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels, n_channels, kernel_size=3, stride=1,
                activation_func=nn.LeakyReLU(negative_slope=0.10, inplace=True)),
            Conv2d(
                n_channels, 1, kernel_size=3, stride=1, activation_func=None))

    def forward(self, image):
        encoder_features = self.encoder(image)
        feature = encoder_features[-1]
        features = [feature]
        for decoder_block, skip in zip(self.decoder, encoder_features[-2::-1]):
            feature = decoder_block(feature, skip)
            features.append(feature)
        features = features[::-1]
        geometry = normalize_scaleless_depth(
            functional.softplus(self.depth_output(feature)) + 1e-3)
        geometries = [geometry] + [
            functional.interpolate(
                geometry, size=value.shape[-2:], mode='bilinear', align_corners=True)
            for value in features[1:]]
        # Normalize only once at full resolution. Renormalizing individual
        # levels would change the units of the inherited scale map again.
        return features, geometries


def global_scale_from_sparse(sparse_depth, geometry):
    '''Estimate one metric scale per image from full-resolution measurements.'''
    measured = (sparse_depth > 0.0).to(geometry.dtype)
    ratios = sparse_depth / geometry.detach().clamp_min(1e-7)
    count = measured.sum(dim=[2, 3], keepdim=True)
    scale = (ratios * measured).sum(dim=[2, 3], keepdim=True) / count.clamp_min(1.0)
    # An image with no measurements has no observable metric scale. A neutral
    # scale keeps its output finite without inventing a measurement.
    return torch.where(count > 0.0, scale, torch.ones_like(scale))


class FixedGeometryPropagationDecoderBlock(nn.Module):
    '''Upsample the scale prior, then propagate against fixed RGB geometry.'''

    def __init__(self, n_channels, n_propagation_iterations):
        super(FixedGeometryPropagationDecoderBlock, self).__init__()
        self.n_propagation_iterations = n_propagation_iterations
        self.use_sparse_depth_features = False
        # The geometry is guidance, not a prediction to be updated here.
        # C preceding features + C RGB skip + one geometry + one scale.
        self.fusion = ConvolutionBlock(2 * n_channels + 2, n_channels, stride=1)
        self.propagation_map = Conv2d(
            n_channels + 2, 9, kernel_size=3, stride=1, activation_func=None)
        self.scale_propagation = ScalePropagation(n_propagation_iterations)

    def forward(self, feature, skip, geometry, scale, sparse_depth, fine_geometry):
        # The preceding geometry is carried only for the common diagnostic
        # call signature. Use the precomputed finer copy of the SAME final Z.
        del geometry
        feature = functional.interpolate(
            feature, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        scale = functional.interpolate(
            scale, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        geometry = fine_geometry
        feature = self.fusion(torch.cat([
            feature, skip, geometry, scale.detach()
        ], dim=1))
        sparse = sparse_depth_at_resolution(sparse_depth, geometry.shape[-2:])
        keep_mask = (sparse > 0.0).to(sparse.dtype)
        logits = self.propagation_map(torch.cat([
            feature, geometry * scale, sparse
        ], dim=1))
        sparse_scale = sparse / geometry.clamp_min(1e-7)
        scale = self.scale_propagation(
            scale, geometry, sparse, keep_mask, logits, sparse_scale)
        # Reuse the existing six-output diagnostic convention, without a
        # rejection head, stochastic decisions, or free scale-refinement head.
        rejection_logits = torch.zeros_like(sparse).repeat(1, 2, 1, 1)
        return feature, geometry, scale, logits, rejection_logits, keep_mask


class FixedGeometryScalePropagationDepthModel(nn.Module):
    '''Two-stage baseline: RGB geometry first, then hierarchical scale only.

    Stage geometry: train the RGB U-Net with full-resolution SI supervision.
    Stage scale: freeze that entire U-Net and train only propagation guidance.
    Mode global: bypass propagation to evaluate global scaling of the exact
    same geometry checkpoint. This is an evaluation control, not a third head.
    '''

    def __init__(self,
                 min_predict_depth=0.1,
                 max_predict_depth=8.0,
                 n_channels=32,
                 n_head=4,
                 encoder_type='conv',
                 n_propagation_iterations=(2, 4, 6, 8, 12),
                 training_stage='scale',
                 propagation_mode='hierarchical'):
        super(FixedGeometryScalePropagationDepthModel, self).__init__()
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.training_stage = training_stage
        self.propagation_mode = propagation_mode
        self.n_propagation_iterations = n_propagation_iterations
        self.fixed_geometry = True
        self.detach_scaleless_depth = True
        self.use_outlier_rejection = False
        self.use_sparse_depth_features = False
        self.n_scale_refinement = 0

        self.geometry = RGBGeometryModel(n_channels, n_head, encoder_type)
        self.propagation_map = Conv2d(
            n_channels + 2, 9, kernel_size=3, stride=1, activation_func=None)
        self.scale_propagation = ScalePropagation(n_propagation_iterations[0])
        self.decoder = nn.ModuleList([
            FixedGeometryPropagationDecoderBlock(n_channels, n_iteration)
            for n_iteration in n_propagation_iterations[1:]
        ])
        # Freezing parameters, not just detaching Z, also prevents scale loss
        # from changing RGB features and indirectly changing the geometry.
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith('geometry.') if training_stage == 'geometry'
                else not name.startswith('geometry.'))
        self.train()

    def train(self, mode=True):
        super(FixedGeometryScalePropagationDepthModel, self).train(mode)
        if self.training_stage != 'geometry':
            self.geometry.eval()
        return self

    def forward(self, image, sparse_depth, return_all_outputs=False,
                return_rejection_outputs=False):
        if self.training_stage == 'geometry':
            features, geometries = self.geometry(image)
        else:
            # No geometry graph is retained during scale training. Its weights
            # and its normalization/evaluation state remain fixed throughout.
            with torch.no_grad():
                features, geometries = self.geometry(image)

        global_scale = global_scale_from_sparse(sparse_depth, geometries[0])
        if self.training_stage == 'geometry' or self.propagation_mode == 'global':
            # Metric images during SI pretraining are previews using sparse
            # global alignment; they do not contribute to geometry training.
            scales = [global_scale.expand_as(value) for value in geometries]
            keep_masks = [(sparse_depth_at_resolution(sparse_depth, value.shape[-2:]) > 0.0)
                          .to(value.dtype) for value in geometries]
        else:
            feature, geometry = features[-1], geometries[-1]
            sparse = sparse_depth_at_resolution(sparse_depth, geometry.shape[-2:])
            keep_mask = (sparse > 0.0).to(sparse.dtype)
            scale = global_scale.expand_as(geometry)
            logits = self.propagation_map(torch.cat([
                feature, geometry * scale, sparse
            ], dim=1))
            scale = self.scale_propagation(
                scale, geometry, sparse, keep_mask, logits,
                sparse / geometry.clamp_min(1e-7))
            scales, keep_masks = [scale], [keep_mask]
            for decoder_block, level in zip(self.decoder, range(3, -1, -1)):
                feature, geometry, scale, _, _, keep_mask = decoder_block(
                    feature, features[level], geometry, scale, sparse_depth,
                    geometries[level])
                scales.append(scale)
                keep_masks.append(keep_mask)
            scales, keep_masks = scales[::-1], keep_masks[::-1]

        # SI is the only geometry objective. Metric loss always uses fixed Z;
        # no metric gradient can train geometry, even during its first stage.
        metric_depths = [value.detach() * scale for value, scale in zip(geometries, scales)]
        depth = torch.clamp(
            metric_depths[0], min=self.min_predict_depth, max=self.max_predict_depth)
        if not return_all_outputs and not return_rejection_outputs:
            return depth
        return dict(
            depth=depth, scaleless_depths=geometries, scale_maps=scales,
            metric_depths_for_loss=metric_depths,
            rejection_keep_masks=keep_masks,
            rejection_logits=[torch.zeros_like(value).repeat(1, 2, 1, 1)
                              for value in keep_masks])
