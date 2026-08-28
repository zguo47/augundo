import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn


REPOSITORY_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    '..'))
DEPTH_COMPLETION_SOURCE = os.path.join(
    REPOSITORY_ROOT,
    'depth_completion',
    'src')

if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)
if DEPTH_COMPLETION_SOURCE not in sys.path:
    sys.path.insert(0, DEPTH_COMPLETION_SOURCE)

from depth_completion_model import DepthCompletionModel


class MockDINOv2(nn.Module):
    """Small offline stand-in for DINOv2's torch.hub interface."""

    def __init__(self, embedding_channels=32, patch_size=14):
        super(MockDINOv2, self).__init__()

        self.embed_dim = embedding_channels
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            3,
            embedding_channels,
            kernel_size=patch_size,
            stride=patch_size)

    def get_intermediate_layers(self, image, n=4, reshape=False):
        feature = self.patch_embed(image)
        features = [feature + 0.01 * index for index in range(n)]

        if reshape:
            return features

        return [
            value.flatten(2).transpose(1, 2)
            for value in features
        ]


class DINOv2GuidedModelTest(unittest.TestCase):

    @mock.patch('torch.hub.load')
    def test_forward_loss_backward_and_checkpoint(self, hub_load):
        hub_load.side_effect = lambda *args, **kwargs: MockDINOv2()

        model = DepthCompletionModel(
            model_name='dinov2_guided_kitti',
            network_modules=['dinov2_vits14'],
            min_predict_depth=1.5,
            max_predict_depth=100.0,
            device=torch.device('cpu'))

        # Deliberately use a shape that is divisible by neither 14 nor 32.
        image = torch.rand(1, 3, 55, 83)
        sparse_depth = torch.zeros(1, 1, 55, 83)
        sparse_depth[:, :, 10, 12] = 8.0
        sparse_depth[:, :, 35, 60] = 32.0
        sparse_depth[:, :, 22, 40] = 18.0
        validity = (sparse_depth > 0.0).float()
        ground_truth = 2.0 + 40.0 * torch.rand(1, 1, 55, 83)

        # The encoder-decoder accepts RGB only and produces one dense prior
        # before sparse depth is introduced for global scale alignment.
        model_depth = model.model.model_depth
        prior_depth = model_depth.forward_image(image)

        self.assertEqual(prior_depth.shape, sparse_depth.shape)

        decoder_inputs = {}

        def capture_decoder_inputs(module, inputs):
            del module
            decoder_inputs['latent_channels'] = inputs[0].shape[1]
            decoder_inputs['skip_channels'] = [
                skip.shape[1] for skip in inputs[1]
            ]

        hook = model_depth.decoder.register_forward_pre_hook(
            capture_decoder_inputs)

        outputs = model.forward_depth(
            image=image,
            sparse_depth=sparse_depth,
            validity_map=validity,
            intrinsics=torch.eye(3).unsqueeze(0),
            return_all_outputs=True)

        hook.remove()

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, sparse_depth.shape)
        self.assertEqual(outputs[1].shape, sparse_depth.shape)
        self.assertTrue(torch.isfinite(outputs[0]).all())
        self.assertTrue(torch.isfinite(outputs[1]).all())
        self.assertEqual(decoder_inputs['latent_channels'], 256)
        self.assertEqual(decoder_inputs['skip_channels'], [3, 19, 35, 67])

        # Robust least squares recovers the common scale and rejects a noisy
        # but otherwise valid sparse measurement.
        test_prior = torch.tensor([[[[2.0, 3.0, 4.0, 5.0]]]])
        test_sparse = torch.tensor([[[[4.0, 6.0, 8.0, 50.0]]]])
        test_validity = torch.ones_like(test_sparse)
        aligned_depth, scale, scale_weights = model_depth.scale_alignment(
            prior_depth=test_prior,
            sparse_depth=test_sparse,
            validity_map=test_validity)

        self.assertTrue(torch.allclose(
            scale,
            torch.tensor([[[[2.0]]]]),
            atol=1e-5))
        self.assertEqual(scale_weights[0, 0, 0, 3].item(), 0.0)
        self.assertTrue(torch.allclose(
            aligned_depth,
            2.0 * test_prior,
            atol=1e-5))
        self.assertNotEqual(
            aligned_depth[0, 0, 0, 3].item(),
            test_sparse[0, 0, 0, 3].item())

        loss, loss_info = model.compute_loss(
            image0=image,
            image1=image,
            image2=image,
            output_depth0=outputs,
            sparse_depth0=sparse_depth,
            validity_map0=validity,
            intrinsics=torch.eye(3).unsqueeze(0),
            pose0to1=None,
            pose0to2=None,
            ground_truth0=ground_truth,
            supervision_type='supervised',
            w_losses={'w_supervised': 1.0})

        self.assertTrue(torch.isfinite(loss))
        self.assertIn('loss_log_l1', loss_info)
        self.assertIn('loss_prior', loss_info)

        loss.backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters_depth()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertGreater(len(trainable_gradients), 0)
        self.assertTrue(all([
            torch.isfinite(gradient).all()
            for gradient in trainable_gradients
        ]))
        self.assertEqual(
            model_depth.decoder.output0.conv.weight.shape[0],
            1)

        optimizer = torch.optim.Adam(model.parameters_depth(), lr=1e-4)

        with tempfile.TemporaryDirectory() as directory:
            model.save_model(
                checkpoint_dirpath=directory,
                step=17,
                optimizer_depth=optimizer,
                optimizer_pose=None)
            checkpoint_path = os.path.join(
                directory,
                'dinov2-guided-17.pth')
            step, restored_optimizer, restored_pose_optimizer = \
                model.restore_model(
                    restore_paths=[checkpoint_path],
                    optimizer_depth=optimizer,
                    optimizer_pose=None)

        self.assertEqual(step, 17)
        self.assertIs(restored_optimizer, optimizer)
        self.assertIsNone(restored_pose_optimizer)


if __name__ == '__main__':
    unittest.main()
