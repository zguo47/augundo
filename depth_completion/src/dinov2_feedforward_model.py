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


def shift_tensor(x, offset_y, offset_x):
    '''
    Shifts tensor spatially and fills locations outside of the image with zero
    '''

    n_height, n_width = x.shape[-2:]
    padding_y = abs(offset_y)
    padding_x = abs(offset_x)

    x = functional.pad(
        x,
        pad=(padding_x, padding_x, padding_y, padding_y),
        mode='constant',
        value=0.0)

    start_y = padding_y + offset_y
    start_x = padding_x + offset_x

    return x[
        ...,
        start_y:start_y + n_height,
        start_x:start_x + n_width]


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


class ImageGuidedDensifier(nn.Module):
    '''
    Densifies sparse metric corrections using RGB decoder guidance.

    Sparse depth is treated only as a set of metric measurements. No learned
    convolution is applied to sparse depth or its validity map.
    '''

    OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def __init__(self,
                 min_predict_depth,
                 max_predict_depth,
                 propagation_scale=4,
                 propagation_dilations=[1, 2, 4, 8, 16, 32],
                 affinity_temperature=0.5):
        super(ImageGuidedDensifier, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.propagation_scale = propagation_scale
        self.propagation_dilations = propagation_dilations
        self.affinity_temperature = affinity_temperature

    def _propagate_once(self,
                        correction,
                        confidence,
                        guidance,
                        dilation):
        ones = torch.ones_like(confidence)

        numerator = confidence * correction
        denominator = confidence
        confidence_numerator = confidence
        affinity_denominator = ones

        for offset_y, offset_x in self.OFFSETS:
            offset_y = offset_y * dilation
            offset_x = offset_x * dilation

            neighbor_guidance = shift_tensor(
                guidance,
                offset_y,
                offset_x)
            neighbor_correction = shift_tensor(
                correction,
                offset_y,
                offset_x)
            neighbor_confidence = shift_tensor(
                confidence,
                offset_y,
                offset_x)
            neighbor_in_bounds = shift_tensor(
                ones,
                offset_y,
                offset_x)

            similarity = torch.sum(
                guidance * neighbor_guidance,
                dim=1,
                keepdim=True)
            affinity = torch.exp(
                (similarity - 1.0) / self.affinity_temperature)
            affinity = affinity * neighbor_in_bounds

            weight = affinity * neighbor_confidence
            numerator = numerator + weight * neighbor_correction
            denominator = denominator + weight
            confidence_numerator = \
                confidence_numerator + affinity * neighbor_confidence
            affinity_denominator = affinity_denominator + affinity

        correction = numerator / (denominator + EPSILON)
        confidence = confidence_numerator / (affinity_denominator + EPSILON)
        confidence = torch.clamp(confidence, min=0.0, max=1.0)

        return correction, confidence

    def forward(self,
                prior_depth,
                guidance,
                sparse_depth,
                validity_map):
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

        log_prior = torch.log(torch.clamp(prior_depth, min=EPSILON))
        log_sparse_depth = torch.log(torch.clamp(
            sparse_depth,
            min=self.min_predict_depth))

        # Align the RGB-only prior to the global metric scale of sparse depth.
        n_valid = torch.sum(validity_map, dim=[1, 2, 3], keepdim=True)
        log_scale = torch.sum(
            validity_map * (log_sparse_depth - log_prior),
            dim=[1, 2, 3],
            keepdim=True) / (n_valid + EPSILON)
        aligned_log_prior = log_prior + log_scale

        residual = validity_map * (
            log_sparse_depth - aligned_log_prior)

        propagation_shape = (
            max(1, prior_depth.shape[-2] // self.propagation_scale),
            max(1, prior_depth.shape[-1] // self.propagation_scale))

        pooled_validity = functional.adaptive_avg_pool2d(
            validity_map,
            propagation_shape)
        pooled_residual = functional.adaptive_avg_pool2d(
            residual,
            propagation_shape) / (pooled_validity + EPSILON)
        pooled_validity = (pooled_validity > 0.0).to(prior_depth.dtype)
        pooled_residual = pooled_residual * pooled_validity

        guidance = functional.interpolate(
            guidance,
            size=propagation_shape,
            mode='bilinear',
            align_corners=False)
        guidance = functional.normalize(
            guidance,
            p=2,
            dim=1,
            eps=EPSILON)

        correction = pooled_residual
        confidence = pooled_validity

        for dilation in self.propagation_dilations:
            correction, confidence = self._propagate_once(
                correction=correction,
                confidence=confidence,
                guidance=guidance,
                dilation=dilation)

            correction = pooled_validity * pooled_residual + \
                (1.0 - pooled_validity) * correction
            confidence = torch.maximum(confidence, pooled_validity)

        correction = functional.interpolate(
            confidence * correction,
            size=prior_depth.shape[-2:],
            mode='bilinear',
            align_corners=False)

        log_min_depth = torch.log(torch.tensor(
            self.min_predict_depth,
            dtype=prior_depth.dtype,
            device=prior_depth.device))
        log_max_depth = torch.log(torch.tensor(
            self.max_predict_depth,
            dtype=prior_depth.dtype,
            device=prior_depth.device))

        output_depth = torch.exp(torch.clamp(
            aligned_log_prior + correction,
            min=log_min_depth,
            max=log_max_depth))

        # Preserve reliable metric measurements exactly at their locations.
        output_depth = validity_map * sparse_depth + \
            (1.0 - validity_map) * output_depth

        return output_depth


class DINOv2DepthCompletionModel(nn.Module):
    '''
    RGB-only encoder-decoder followed by image-guided sparse densification.

    DINOv2 and the decoder process RGB features only. Sparse depth is introduced
    after the decoder as metric measurements for scale alignment and residual
    propagation.
    '''

    def __init__(self,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 dinov2_model_name='dinov2_vits14',
                 dinov2_repository='facebookresearch/dinov2:f896929',
                 freeze_dinov2=True,
                 n_guidance=8):
        super(DINOv2DepthCompletionModel, self).__init__()

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.n_guidance = n_guidance

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
            output_channels=1 + n_guidance,
            n_resolution=1,
            n_filters=[128, 64, 32, 16],
            n_skips=[64 + 3, 32 + 3, 16 + 3, 3],
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            output_func='linear',
            use_batch_norm=False,
            deconv_type='up')

        self.densifier = ImageGuidedDensifier(
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
        raw_depth = output[:, 0:1, ...]
        guidance = output[:, 1:1 + self.n_guidance, ...]

        raw_depth = torch.sigmoid(raw_depth)
        prior_depth = self.min_predict_depth / (
            raw_depth + self.min_predict_depth / self.max_predict_depth)

        return prior_depth, guidance

    def forward(self, image, sparse_depth, validity_map):
        # The complete encoder-decoder pass occurs before sparse depth is used.
        prior_depth, guidance = self.forward_image(image)

        return self.densifier(
            prior_depth=prior_depth,
            guidance=guidance,
            sparse_depth=sparse_depth,
            validity_map=validity_map)
