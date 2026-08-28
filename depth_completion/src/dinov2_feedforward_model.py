import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as functional


# Reuse the decoder and convolution blocks already used by KBNet.
sys.path.insert(0, os.path.join(
    'external_src', 'depth_completion', 'kbnet', 'src'))
import networks
import net_utils


EPSILON = 1e-8


class DINOv2Encoder(nn.Module):
    '''Wrapper around the official DINOv2 Torch Hub backbone.'''

    N_CHANNEL = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536
    }

    def __init__(self,
                 model_name='dinov2_vits14',
                 repository='facebookresearch/dinov2:f896929',
                 freeze=True):
        super(DINOv2Encoder, self).__init__()

        self.model = torch.hub.load(
            repository,
            model_name,
            source='github',
            pretrained=True,
            skip_validation=True)

        self.output_channels = getattr(
            self.model,
            'embed_dim',
            self.N_CHANNEL[model_name])
        self.freeze = freeze

        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _forward(self, image):
        feature = self.model.get_intermediate_layers(
            image,
            n=1,
            reshape=True)[0]

        return feature

    def forward(self, image):
        # DINOv2 requires image dimensions to be divisible by its patch size.
        patch_size = getattr(self.model, 'patch_size', 14)
        if isinstance(patch_size, tuple):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size

        n_pad_height = \
            (patch_height - image.shape[-2] % patch_height) % patch_height
        n_pad_width = \
            (patch_width - image.shape[-1] % patch_width) % patch_width

        if n_pad_height > 0 or n_pad_width > 0:
            image = functional.pad(
                image,
                pad=(0, n_pad_width, 0, n_pad_height),
                mode='replicate')

        if self.freeze:
            with torch.no_grad():
                return self._forward(image)

        return self._forward(image)

    def train(self, mode=True):
        super(DINOv2Encoder, self).train(mode)
        if self.freeze:
            self.model.eval()
        return self


class LeastSquaresScaleAlignment(nn.Module):
    '''
    Aligns an RGB-only depth prior to valid sparse metric measurements.

    Scale-inconsistent points are rejected using the median log scale ratio,
    and the remaining points are used in a least-squares scale fit.
    '''

    def __init__(self,
                 min_predict_depth,
                 max_predict_depth,
                 min_scale=0.1,
                 max_scale=10.0,
                 max_scale_ratio=1.5,
                 min_valid_points=3):
        super(LeastSquaresScaleAlignment, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_log_scale_difference = float(torch.log(
            torch.tensor(max_scale_ratio)))
        self.min_valid_points = min_valid_points

    def _validity_map(self, prior_depth, sparse_depth, validity_map):
        validity_map = torch.logical_and(validity_map > 0.0, torch.logical_and(
            torch.isfinite(sparse_depth),
            torch.isfinite(prior_depth)))
        validity_map = torch.logical_and(validity_map, torch.logical_and(
            sparse_depth >= self.min_predict_depth,
            sparse_depth <= self.max_predict_depth))
        validity_map = torch.logical_and(validity_map, prior_depth > 0.0)

        return validity_map

    def _robust_weights(self, prior_depth, sparse_depth, validity_map):
        '''Rejects points whose scale ratios disagree with the valid majority.'''

        weights = torch.zeros_like(prior_depth)
        scale_ratio = sparse_depth / torch.clamp(prior_depth, min=EPSILON)
        log_scale_ratio = torch.log(torch.clamp(scale_ratio, min=EPSILON))

        # Each sample can contain a different number of valid sparse points.
        # Point selection does not need gradients; the least-squares fit below
        # remains differentiable with respect to the selected prior depths.
        with torch.no_grad():
            for batch_index in range(prior_depth.shape[0]):
                valid = validity_map[batch_index]

                if torch.sum(valid).item() < self.min_valid_points:
                    continue

                valid_log_scale_ratio = log_scale_ratio[batch_index][valid]
                median_log_scale_ratio = torch.median(valid_log_scale_ratio)
                inlier = torch.abs(
                    log_scale_ratio[batch_index] -
                    median_log_scale_ratio) <= self.max_log_scale_difference
                weights[batch_index] = torch.logical_and(
                    valid,
                    inlier).to(prior_depth.dtype)

        return weights

    def forward(self, prior_depth, sparse_depth, validity_map):
        validity_map = self._validity_map(
            prior_depth=prior_depth,
            sparse_depth=sparse_depth,
            validity_map=validity_map)
        sparse_depth = torch.where(
            validity_map,
            sparse_depth,
            torch.zeros_like(sparse_depth))

        weights = self._robust_weights(
            prior_depth=prior_depth,
            sparse_depth=sparse_depth,
            validity_map=validity_map)

        numerator = torch.sum(
            weights * prior_depth * sparse_depth,
            dim=[1, 2, 3],
            keepdim=True)
        denominator = torch.sum(
            weights * prior_depth * prior_depth,
            dim=[1, 2, 3],
            keepdim=True)
        n_inlier = torch.sum(weights, dim=[1, 2, 3], keepdim=True)

        scale = numerator / (denominator + EPSILON)
        scale = torch.clamp(
            scale,
            min=self.min_scale,
            max=self.max_scale)
        scale = torch.where(
            n_inlier >= self.min_valid_points,
            scale,
            torch.ones_like(scale))

        output_depth = torch.clamp(
            scale * prior_depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)

        return output_depth, scale, weights


class DINOv2DepthCompletionModel(nn.Module):
    '''
    RGB-only encoder-decoder followed by robust metric scale alignment.

    DINOv2 and the decoder process RGB features only. Sparse depth is introduced
    after the decoder and is used only to fit a global multiplicative scale.
    '''

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 dinov2_model_name='dinov2_vits14',
                 dinov2_repository='facebookresearch/dinov2:f896929',
                 freeze_dinov2=True):
        super(DINOv2DepthCompletionModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.encoder = DINOv2Encoder(
            model_name=dinov2_model_name,
            repository=dinov2_repository,
            freeze=freeze_dinov2)

        n_dino = self.encoder.output_channels

        self.projection8 = net_utils.Conv2d(
            n_dino, 64, kernel_size=1, activation_func=None)
        self.projection4 = net_utils.Conv2d(
            n_dino, 32, kernel_size=1, activation_func=None)
        self.projection2 = net_utils.Conv2d(
            n_dino, 16, kernel_size=1, activation_func=None)
        self.projection16 = net_utils.Conv2d(
            n_dino, 256, kernel_size=3, activation_func=None)

        # All latent and skip features below are derived from RGB only.
        self.decoder = networks.MultiScaleDecoder(
            input_channels=256,
            output_channels=1,
            n_resolution=1,
            n_filters=[128, 64, 32, 16],
            n_skips=[64 + 3, 32 + 3, 16 + 3, 3],
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            output_func='linear',
            use_batch_norm=False,
            deconv_type='up')

        self.scale_alignment = LeastSquaresScaleAlignment(
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth)

        self.register_buffer(
            'image_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'image_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _scale_shape(self, shape, scale):
        return (
            max(1, shape[0] // scale),
            max(1, shape[1] // scale))

    def _make_skip(self, dino_feature, image, shape, projection):
        image = functional.interpolate(
            image,
            size=shape,
            mode='bilinear',
            align_corners=False)
        dino_feature = functional.interpolate(
            dino_feature,
            size=shape,
            mode='bilinear',
            align_corners=False)
        dino_feature = projection(dino_feature)

        return torch.cat([dino_feature, image], dim=1)

    def forward_image(self, image):
        '''Forwards RGB through the encoder-decoder without sparse depth.'''

        shape = image.shape[-2:]
        normalized_image = (image - self.image_mean) / self.image_std
        dino_feature = self.encoder(normalized_image)

        shape2 = self._scale_shape(shape, 2)
        shape4 = self._scale_shape(shape, 4)
        shape8 = self._scale_shape(shape, 8)
        shape16 = self._scale_shape(shape, 16)

        latent = functional.interpolate(
            dino_feature,
            size=shape16,
            mode='bilinear',
            align_corners=False)
        latent = self.projection16(latent)

        skips = [
            normalized_image,
            self._make_skip(
                dino_feature, normalized_image, shape2, self.projection2),
            self._make_skip(
                dino_feature, normalized_image, shape4, self.projection4),
            self._make_skip(
                dino_feature, normalized_image, shape8, self.projection8)
        ]

        output = self.decoder(latent, skips, shape=shape)[-1]
        raw_depth = torch.sigmoid(output)
        prior_depth = self.min_predict_depth / (
            raw_depth + self.min_predict_depth / self.max_predict_depth)

        return prior_depth

    def forward(self, image, sparse_depth, validity_map):
        # The complete encoder-decoder pass occurs before sparse depth is used.
        prior_depth = self.forward_image(image)

        output_depth, _, _ = self.scale_alignment(
            prior_depth=prior_depth,
            sparse_depth=sparse_depth,
            validity_map=validity_map)

        return output_depth, prior_depth
