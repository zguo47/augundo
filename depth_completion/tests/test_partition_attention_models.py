import os
import sys
import tempfile
import unittest

import torch


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
from partition_attention_model import PartitionAttentionDepthModel


class PartitionAttentionModelTest(unittest.TestCase):

    def test_parallel_and_sequential_partition_shapes(self):
        image = torch.rand(1, 3, 64, 96)
        expected_shapes = [
            (1, 16, 64, 96),
            (1, 16, 32, 48),
            (1, 16, 16, 24),
            (1, 16, 8, 12)
        ]

        for mode in ('parallel', 'sequential'):
            model = PartitionAttentionDepthModel(
                partition_mode=mode,
                n_channels=16,
                n_head=4,
                max_global_tokens=16,
                max_metric_queries=16)
            partitions = model.extract_partitions(image)

            self.assertEqual(
                [tuple(partition.shape) for partition in partitions],
                expected_shapes)

            if mode == 'parallel':
                self.assertEqual(
                    [layer.kernel_size[0]
                     for layer in model.pyramid.convolutions],
                    [1, 3, 7, 15])
                self.assertEqual(
                    [layer.stride[0]
                     for layer in model.pyramid.convolutions],
                    [1, 2, 4, 8])
            else:
                self.assertEqual(
                    [layer.kernel_size[0]
                     for layer in model.pyramid.convolutions],
                    [1, 3, 5, 7])
                self.assertEqual(
                    [layer.stride[0]
                     for layer in model.pyramid.convolutions],
                    [1, 2, 2, 2])

    def test_sparse_tokens_constraints_backward_and_checkpoint(self):
        model = DepthCompletionModel(
            model_name='partition_attention_kitti',
            network_modules=['partition_parallel'],
            min_predict_depth=1.5,
            max_predict_depth=100.0,
            device=torch.device('cpu'))

        # Odd dimensions exercise right/bottom padding and exact output crop.
        image = torch.rand(1, 3, 33, 49)
        sparse_depth0 = torch.zeros(1, 1, 33, 49)
        sparse_depth1 = torch.zeros(1, 1, 33, 49)
        sparse_depth1[:, :, 5, 7] = 8.0
        sparse_depth1[:, :, 25, 40] = 32.0
        validity0 = torch.zeros_like(sparse_depth0)
        validity1 = (sparse_depth1 > 0.0).float()
        ground_truth = 1.5 + 50.0 * torch.rand(1, 1, 33, 49)

        outputs0 = model.forward_depth(
            image=image,
            sparse_depth=sparse_depth0,
            validity_map=validity0,
            intrinsics=None,
            return_all_outputs=True)
        output1 = model.forward_depth(
            image=image,
            sparse_depth=sparse_depth1,
            validity_map=validity1,
            intrinsics=torch.eye(3).unsqueeze(0),
            return_all_outputs=False)

        self.assertEqual(len(outputs0), 1)
        self.assertEqual(tuple(outputs0[0].shape), (1, 1, 33, 49))
        self.assertTrue(torch.isfinite(outputs0[0]).all())
        self.assertGreaterEqual(outputs0[0].min().item(), 1.5)
        self.assertLessEqual(outputs0[0].max().item(), 100.0)
        self.assertFalse(torch.allclose(outputs0[0], output1))
        self.assertTrue(torch.allclose(
            output1[validity1.bool()],
            sparse_depth1[validity1.bool()]))

        model_depth = model.model.model_depth
        prepared_depth, prepared_validity = \
            model_depth._prepare_sparse_depth(
                image=image,
                sparse_depth=sparse_depth1,
                validity_map=validity1)
        fine_feature = model_depth.extract_partitions(image)[0]
        tokens, padding_mask, has_valid = \
            model_depth.sparse_token_encoder(
                image_feature=fine_feature,
                sparse_depth=prepared_depth,
                validity_map=prepared_validity)
        self.assertEqual(tokens.shape[1], 2)
        self.assertEqual(torch.sum(~padding_mask).item(), 2)
        self.assertTrue(has_valid.item())

        loss, loss_info = model.compute_loss(
            image0=image,
            image1=image,
            image2=image,
            output_depth0=[output1],
            sparse_depth0=sparse_depth0,
            validity_map0=validity0,
            intrinsics=torch.eye(3).unsqueeze(0),
            pose0to1=None,
            pose0to2=None,
            ground_truth0=ground_truth,
            supervision_type='supervised',
            w_losses={'w_supervised': 1.0})

        self.assertTrue(torch.isfinite(loss))
        self.assertIn('loss_log_l1', loss_info)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters_depth()
            if parameter.grad is not None
        ]
        self.assertGreater(len(gradients), 0)
        self.assertTrue(all([
            torch.isfinite(gradient).all()
            for gradient in gradients
        ]))

        communication_modules = [
            model_depth.fine_to_coarse_blocks,
            [model_depth.sparse_token_encoder],
            [model_depth.sparse_metric_attention],
            [model_depth.global_attention],
            model_depth.coarse_to_fine_blocks
        ]
        for modules in communication_modules:
            module_gradients = [
                parameter.grad
                for module in modules
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertGreater(len(module_gradients), 0)
            self.assertGreater(sum([
                torch.sum(torch.abs(gradient)).item()
                for gradient in module_gradients
            ]), 0.0)

        optimizer = torch.optim.Adam(model.parameters_depth(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            model.save_model(
                checkpoint_dirpath=directory,
                step=23,
                optimizer_depth=optimizer,
                optimizer_pose=None)
            checkpoint_path = os.path.join(
                directory,
                'partition-attention-23.pth')
            step, restored_optimizer, restored_pose_optimizer = \
                model.restore_model(
                    restore_paths=[checkpoint_path],
                    optimizer_depth=optimizer,
                    optimizer_pose=None)

        self.assertEqual(step, 23)
        self.assertIs(restored_optimizer, optimizer)
        self.assertIsNone(restored_pose_optimizer)


if __name__ == '__main__':
    unittest.main()
