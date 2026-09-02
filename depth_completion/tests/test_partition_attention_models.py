import os
import sys
import tempfile
import unittest

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

from partition_attention_depth_completion_model import \
    PartitionAttentionDepthCompletionModel
from partition_attention_model import (
    AttentionUpdate,
    PartitionAttentionDepthModel,
    feature_to_partitions)


class PartitionAttentionModelTest(unittest.TestCase):

    def test_level_and_partition_shapes(self):
        model = PartitionAttentionDepthModel()
        image = torch.rand(1, 3, 32, 48)
        sparse_depth = torch.rand(1, 1, 32, 48)

        rgb_features = model.rgb_pyramid(image)
        sparse_features = model.sparse_pyramid(sparse_depth)
        expected_shapes = [
            (1, 32, 32, 48),
            (1, 32, 16, 24),
            (1, 32, 8, 12),
            (1, 32, 4, 6),
            (1, 32, 2, 3)
        ]

        self.assertEqual(
            [tuple(feature.shape) for feature in rgb_features],
            expected_shapes)
        self.assertEqual(
            [tuple(feature.shape) for feature in sparse_features],
            expected_shapes)
        self.assertEqual(
            [layer.kernel_size[0]
             for layer in model.rgb_pyramid.convolutions],
            [1, 2, 4, 8, 16])
        self.assertEqual(
            [layer.stride[0]
             for layer in model.rgb_pyramid.convolutions],
            [1, 2, 4, 8, 16])
        self.assertEqual(
            [layer.padding[0]
             for layer in model.rgb_pyramid.convolutions],
            [0, 0, 0, 0, 0])

        rgb_partitions = [
            feature_to_partitions(feature, n_grid)
            for feature, n_grid in zip(
                rgb_features, model.level_grids)
        ]
        self.assertEqual(
            [tuple(partitions.shape) for partitions in rgb_partitions],
            [
                (1, 16, 16, 6, 32),
                (1, 8, 8, 6, 32),
                (1, 4, 4, 6, 32),
                (1, 2, 2, 6, 32),
                (1, 1, 1, 6, 32)
            ])

    def test_attention_with_shared_channel_count(self):
        attention = AttentionUpdate(n_channels=32, n_head=4)
        query = torch.rand(2, 6, 32, requires_grad=True)
        context = torch.rand(2, 24, 32, requires_grad=True)

        output = attention(query, context)

        self.assertEqual(tuple(output.shape), (2, 6, 32))
        output.mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_forward_backward_and_checkpoint(self):
        torch.manual_seed(7)
        model = PartitionAttentionDepthCompletionModel(
            min_predict_depth=0.1,
            max_predict_depth=8.0,
            device=torch.device('cpu'))

        image = torch.rand(1, 3, 32, 48)
        sparse_depth0 = torch.zeros(1, 1, 32, 48)
        sparse_depth1 = sparse_depth0.clone()
        sparse_depth1[:, :, 3, 5] = 1.25
        sparse_depth1[:, :, 12, 19] = 4.50
        validity = (sparse_depth1 > 0.0).float()
        ground_truth = 0.1 + 7.9 * torch.rand(1, 1, 32, 48)

        output_without_points = model.forward_depth(
            image=image,
            sparse_depth=sparse_depth0,
            validity_map=torch.zeros_like(sparse_depth0),
            return_all_outputs=False)
        output = model.forward_depth(
            image=image,
            sparse_depth=sparse_depth1,
            validity_map=validity,
            return_all_outputs=False)

        self.assertEqual(tuple(output.shape), (1, 1, 32, 48))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), 0.1)
        self.assertLessEqual(output.max().item(), 8.0)
        self.assertFalse(torch.allclose(output_without_points, output))

        loss, loss_info = model.compute_loss_supervised(
            target_depth=ground_truth,
            output_depth=[output],
            w_losses={'w_supervised': 1.0})
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('loss_log_l1', loss_info)
        loss.backward()

        modules_on_the_output_path = [
            model.model_depth.rgb_local_attention,
            model.model_depth.sparse_local_attention,
            model.model_depth.rgb_fine_to_coarse_attention,
            model.model_depth.sparse_fine_to_coarse_attention,
            model.model_depth.rgb_coarse_to_fine_attention,
            model.model_depth.sparse_coarse_to_fine_attention,
            model.model_depth.rgb_from_sparse_attention,
            [model.model_depth.depth_output]
        ]
        for modules in modules_on_the_output_path:
            gradients = [
                parameter.grad
                for module in modules
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertGreater(len(gradients), 0)
            self.assertTrue(all(
                torch.isfinite(gradient).all()
                for gradient in gradients))

        # The only spatial convolutions are the five initial level creators in
        # each input branch.
        convolution_names = [
            name
            for name, module in model.model_depth.named_modules()
            if isinstance(module, nn.Conv2d)
        ]
        self.assertEqual(len(convolution_names), 10)
        self.assertTrue(all(
            name.startswith('rgb_pyramid.convolutions.') or
            name.startswith('sparse_pyramid.convolutions.')
            for name in convolution_names))

        optimizer = torch.optim.Adam(model.parameters_depth(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, 'model.pth')
            model.save_model(
                checkpoint_path=checkpoint_path,
                step=11,
                optimizer=optimizer)
            step, restored_optimizer = model.restore_model(
                restore_path=checkpoint_path,
                optimizer=optimizer)

        self.assertEqual(step, 11)
        self.assertIs(restored_optimizer, optimizer)


if __name__ == '__main__':
    unittest.main()
