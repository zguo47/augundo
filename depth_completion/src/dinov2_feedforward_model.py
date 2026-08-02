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


def sparse_depth_at_shape(sparse_depth, validity_map, shape):
    """Downsample sparse depth."""

    pooled_validity = functional.adaptive_avg_pool2d(validity_map, shape)
    pooled_depth = functional.adaptive_avg_pool2d(
        sparse_depth * validity_map,
        shape)
    pooled_depth = pooled_depth / (pooled_validity + EPSILON)
    pooled_validity = (pooled_validity > 0.0).to(sparse_depth.dtype)

    return pooled_depth * pooled_validity, pooled_validity


class DINOv2Encoder(nn.Module):
    """Wrapper around the official DINOv2 Torch Hub backbone."""

    N_CHANNEL = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536
    }

    def __init__(self,
                 model_name='dinov2_vits14',
                 repository='facebookresearch/dinov2',
                 freeze=True):
        super(DINOv2Encoder, self).__init__()

        self.model = torch.hub.load(
            repository,
            model_name,
            source='github'
            pretrained=True)

        self.output_channels = getattr(
            self.model,
            'embed_dim',
            self.N_CHANNEL[model_name])
        self.freeze = freeze

        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def _reshape_tokens(self, feature, image_shape):
        if isinstance(feature, tuple):
        feature = feature[0]

        if feature.ndim == 4:
            return feature

        patch_size = getattr(self.model, 'patch_size', 14)
        if isinstance(patch_size, tuple):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size

        n_height = image_shape[-2] // patch_height
        n_width = image_shape[-1] // patch_width

        if feature.shape[1] == n_height * n_width + 1:
            feature = feature[:, 1:, :]

        if feature.shape[1] != n_height * n_width:
            raise ValueError(
                'Cannot reshape {} tokens into {} x {} patches'.format(
                    feature.shape[1], n_height, n_width))

        return feature.transpose(1, 2).reshape(
            feature.shape[0],
            feature.shape[2],
            n_height,
            n_width)

    def _forward(self, image):
        try:
            feature = self.model.get_intermediate_layers(
                image,
                n=1,
                reshape=True)[0]
        except TypeError:
            feature = self.model.get_intermediate_layers(image, n=1)[0]

        return self._reshape_tokens(feature, image.shape)

    def forward(self, image):
        if self.freeze:
            with torch.no_grad():
                return self._forward(image)

        return self._forward(image)

    def train(self, mode=True):
        super(DINOv2Encoder, self).train(mode)
        if self.freeze:
            self.model.eval()
        return self


class DINOv2DepthCompletionModel(nn.Module):
    """
    Minimal feed-forward depth completion model.

    DINOv2 extracts image features. Sparse depth and its validity map are
    concatenated with those features at several decoder resolutions. The
    existing KBNet MultiScaleDecoder then produces one dense depth map.
    """

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 dinov2_model_name='dinov2_vits14',
                 dinov2_repository='facebookresearch/dinov2',
                 freeze_dinov2=True):
        super(DINOv2DepthCompletionModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.encoder = DINOv2Encoder(
            model_name=dinov2_model_name,
            repository=dinov2_repository,
            freeze=freeze_dinov2)

        n_dino = self.encoder.output_channels

        # Project the same DINOv2 feature map for the decoder resolutions.
        self.projection8 = net_utils.Conv2d(
            n_dino, 64, kernel_size=1, activation_func=None)
        self.projection4 = net_utils.Conv2d(
            n_dino, 32, kernel_size=1, activation_func=None)
        self.projection2 = net_utils.Conv2d(
            n_dino, 16, kernel_size=1, activation_func=None)
        self.projection16 = net_utils.Conv2d(
            n_dino + 2, 256, kernel_size=3, activation_func=None)

        # Skip channels contain DINO feature, resized RGB, sparse depth and
        # validity. At full resolution only raw RGB and sparse inputs are used.
        self.decoder = networks.MultiScaleDecoder(
            input_channels=256,
            output_channels=1,
            n_resolution=1,
            n_filters=[128, 64, 32, 16],
            n_skips=[64 + 3 + 2, 32 + 3 + 2, 16 + 3 + 2, 3 + 2],
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            output_func='linear',
            use_batch_norm=False,
            deconv_type='up')

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

    def _make_skip(self,
                   dino_feature,
                   image,
                   sparse_depth,
                   validity_map,
                   shape,
                   projection):
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
        depth, validity = sparse_depth_at_shape(
            sparse_depth,
            validity_map,
            shape)

        return torch.cat([
            dino_feature,
            image,
            depth / self.max_predict_depth,
            validity
        ], dim=1)

    def forward(self, image, sparse_depth, validity_map):
        shape = image.shape[-2:]
        validity_map = torch.logical_and(
            validity_map > 0.0,
            sparse_depth > 0.0).to(sparse_depth.dtype)
        sparse_depth = torch.where(
            validity_map > 0.0,
            torch.clamp(
                sparse_depth,
                min=self.min_predict_depth,
                max=self.max_predict_depth),
            torch.zeros_like(sparse_depth))

        normalized_image = (image - self.image_mean) / self.image_std
        dino_feature = self.encoder(normalized_image)

        shape2 = self._scale_shape(shape, 2)
        shape4 = self._scale_shape(shape, 4)
        shape8 = self._scale_shape(shape, 8)
        shape16 = self._scale_shape(shape, 16)

        sparse16, validity16 = sparse_depth_at_shape(
            sparse_depth,
            validity_map,
            shape16)
        latent = functional.interpolate(
            dino_feature,
            size=shape16,
            mode='bilinear',
            align_corners=False)
        latent = self.projection16(torch.cat([
            latent,
            sparse16 / self.max_predict_depth,
            validity16
        ], dim=1))

        skips = [
            torch.cat([
                normalized_image,
                sparse_depth / self.max_predict_depth,
                validity_map
            ], dim=1),
            self._make_skip(
                dino_feature, normalized_image, sparse_depth, validity_map,
                shape2, self.projection2),
            self._make_skip(
                dino_feature, normalized_image, sparse_depth, validity_map,
                shape4, self.projection4),
            self._make_skip(
                dino_feature, normalized_image, sparse_depth, validity_map,
                shape8, self.projection8)
        ]

        output = self.decoder(latent, skips, shape=shape)[-1]
        output = torch.sigmoid(output)
        output_depth = self.min_predict_depth / (
            output + self.min_predict_depth / self.max_predict_depth)

        # Sparse samples are treated as reliable metric measurements.
        output_depth = validity_map * sparse_depth + \
            (1.0 - validity_map) * output_depth

        return output_depth
