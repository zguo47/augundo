import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from partition_attention_model_ablation import (
    AttentionUpdate,
    ConvolutionPyramid,
    add_position_encoding)


class DecoderBlock(nn.Module):
    '''Upsample a feature, concatenate its skip, and fuse both spatial maps.'''

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels

        self.up_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(inplace=True))

        self.fusion_convolution = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(inplace=True))

    def forward(self, feature, skip):
        # This is the interpolation-based ``up`` decoder block used by KBNet.
        # Matching the skip shape also supports the pyramid's 32 -> 512 jump.
        feature = functional.interpolate(
            input=feature,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=True)
        feature = self.up_convolution(feature)

        if self.skip_channels > 0:
            feature = torch.cat([feature, skip], dim=1)

        return self.fusion_convolution(feature)


class MultiScaleDecoder(nn.Module):
    '''
    KBNet multi-scale decoder adapted to the four transitions in this pyramid.
    Coarser depth predictions are upsampled and supplied to finer decoder
    stages, in addition to the feature skip at that spatial level.
    '''

    def __init__(self,
                 input_channels,
                 output_channels,
                 n_resolution,
                 n_filters,
                 n_skips):
        super(MultiScaleDecoder, self).__init__()

        self.n_resolution = n_resolution

        # R_4 -> R_3
        self.deconv3 = DecoderBlock(
            in_channels=input_channels,
            skip_channels=n_skips[0],
            out_channels=n_filters[0])

        self.output3 = nn.Conv2d(
            n_filters[0],
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1) if n_resolution > 3 else None

        # R_3 -> R_2
        skip_channels = n_skips[1]
        if n_resolution > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = DecoderBlock(
            in_channels=n_filters[0],
            skip_channels=skip_channels,
            out_channels=n_filters[1])

        self.output2 = nn.Conv2d(
            n_filters[1],
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1) if n_resolution > 2 else None

        # R_2 -> R_1
        skip_channels = n_skips[2]
        if n_resolution > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = DecoderBlock(
            in_channels=n_filters[1],
            skip_channels=skip_channels,
            out_channels=n_filters[2])

        self.output1 = nn.Conv2d(
            n_filters[2],
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1) if n_resolution > 1 else None

        # R_1 -> R_0
        skip_channels = n_skips[3]
        if n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = DecoderBlock(
            in_channels=n_filters[2],
            skip_channels=skip_channels,
            out_channels=n_filters[3])

        self.output0 = nn.Conv2d(
            n_filters[3],
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, feature, skips):
        '''Decode the bottom feature with skips ordered from R_0 through R_3.'''
        outputs = []

        feature = self.deconv3(feature, skips[3])
        if self.output3 is not None:
            output3 = self.output3(feature)
            outputs.append(output3)
            upsample_output3 = functional.interpolate(
                input=output3,
                size=skips[2].shape[-2:],
                mode='bilinear',
                align_corners=True)

        skip = skips[2]
        if self.n_resolution > 3:
            skip = torch.cat([skip, upsample_output3], dim=1)
        feature = self.deconv2(feature, skip)

        if self.output2 is not None:
            output2 = self.output2(feature)
            outputs.append(output2)
            upsample_output2 = functional.interpolate(
                input=output2,
                size=skips[1].shape[-2:],
                mode='bilinear',
                align_corners=True)

        skip = skips[1]
        if self.n_resolution > 2:
            skip = torch.cat([skip, upsample_output2], dim=1)
        feature = self.deconv1(feature, skip)

        if self.output1 is not None:
            output1 = self.output1(feature)
            outputs.append(output1)
            upsample_output1 = functional.interpolate(
                input=output1,
                size=skips[0].shape[-2:],
                mode='bilinear',
                align_corners=True)

        skip = skips[0]
        if self.n_resolution > 1:
            skip = torch.cat([skip, upsample_output1], dim=1)
        feature = self.deconv0(feature, skip)

        outputs.append(self.output0(feature))
        return outputs


class PartitionAttentionDepthModel(nn.Module):
    '''
    RGB forms the same five-level convolution pyramid as the other two
    methods. R_1 through R_4 independently perform full self-attention and are
    restored to spatial feature maps before the KBNet multi-scale decoder.
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

        self.rgb_pyramid = ConvolutionPyramid(
            input_channels=3,
            n_channels=n_channels)

        self.rgb_full_attention = nn.ModuleList([
            AttentionUpdate(n_channels, n_head)
            for _ in range(self.n_level - 1)
        ])

        # Four decoder stages correspond to R_4 -> R_3 -> R_2 -> R_1 -> R_0.
        # Three output resolutions feed predictions from R_2 and R_1 into the
        # finer stages before producing the final full-resolution prediction.
        self.multiscale_decoder = MultiScaleDecoder(
            input_channels=n_channels,
            output_channels=1,
            n_resolution=3,
            n_filters=[n_channels, n_channels, n_channels, n_channels],
            n_skips=[n_channels, n_channels, n_channels, n_channels])

    def full_attention(self, feature, attention_block):
        '''Full self-attention across every spatial token in one level.'''
        n_batch, n_channel, n_height, n_width = feature.shape
        tokens = feature.permute(0, 2, 3, 1).reshape(
            n_batch,
            n_height * n_width,
            n_channel)
        tokens = attention_block(tokens, tokens)

        # Restore every attended sequence to its spatial feature map.
        return tokens.reshape(
            n_batch,
            n_height,
            n_width,
            n_channel).permute(0, 3, 1, 2)

    def forward(self, image):
        rgb_features = self.rgb_pyramid(image)
        rgb_features = [
            add_position_encoding(feature)
            for feature in rgb_features
        ]

        # R_1, R_2, R_3 and R_4 attend independently and in parallel.
        for level, attention_block in enumerate(
                self.rgb_full_attention,
                start=1):
            rgb_features[level] = self.full_attention(
                rgb_features[level],
                attention_block)

        raw_depths = self.multiscale_decoder(
            feature=rgb_features[-1],
            skips=rgb_features[:-1])
        raw_depth = raw_depths[-1]

        normalized_depth = torch.sigmoid(raw_depth)
        log_min_depth = math.log(self.min_predict_depth)
        log_max_depth = math.log(self.max_predict_depth)
        return torch.exp(
            log_min_depth +
            normalized_depth * (log_max_depth - log_min_depth))
