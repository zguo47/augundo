import os
import sys
import unittest
from unittest import mock

from PIL import Image

import torch
import torch.nn.functional as functional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scale_propagation_model import (
    FeatureGuidedScaleUpsampling,
    PropagationDecoderBlock,
    ScalePropagationDepthModel,
    SharedEncoder,
    feature_at_sparse_points,
    geometry_at_sparse_points,
    measurement_rejection_input,
    normalize_rejection_input,
    propagate_scale,
    rejection_keep_mask,
    sparse_depth_at_resolution)
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from utils.src import loss_utils
from utils.src.transforms import Transforms


class HierarchicalScalePropagationTest(unittest.TestCase):

    def test_rejection_targets_ignore_unknown_and_ambiguous_raw_ground_truth(self):
        sparse = torch.tensor([[[[10.05, 10.3, 14., 5., 0., 120.]]]])
        target = torch.tensor([[[[10., 10., 10., 0., 10., 120.]]]])
        labels = loss_utils.rejection_supervision_targets(sparse, target)
        self.assertEqual(labels.tolist(), [[[[0, -1, 1, -1, -1, 0]]]])

    def test_balanced_rejection_loss_corrects_saturated_bad_points(self):
        logits = torch.zeros(1, 2, 1, 101)
        logits[:, 0] = 15.
        logits.requires_grad_()
        labels = torch.zeros(1, 1, 1, 101, dtype=torch.long)
        labels[:, :, :, -1] = 1
        loss = loss_utils.balanced_rejection_loss_func(logits, labels)
        expected = (2. * functional.cross_entropy(logits[:, :, :, :100], labels[:, 0, :, :100]) + \
            functional.cross_entropy(logits[:, :, :, 100:], labels[:, 0, :, 100:])) / 3.
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0, 0, -1]), 0.32)
        self.assertLess(float(logits.grad[0, 1, 0, -1]), -0.32)
        for label in [-1, 0, 1]:
            scores = torch.zeros(1, 2, 2, 3, requires_grad=True)
            value = loss_utils.balanced_rejection_loss_func(
                scores, torch.full((1, 1, 2, 3), label, dtype=torch.long))
            self.assertTrue(torch.isfinite(value))
            if label == -1:
                self.assertEqual(float(value), 0.)
            value.backward()
            self.assertTrue(torch.isfinite(scores.grad).all())

    def test_rejection_supervision_uses_native_logits_and_original_selected_gt(self):
        wrapper = self.make_wrapper(True)
        image = torch.rand(1, 3, 17, 25)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 11] = 20.
        target = torch.ones_like(sparse) * 20.
        target[:, :, 7, 11] = 5.
        normal = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        # Reset RNG around the comparison: hard rejection samples Gumbel noise.
        rng = torch.get_rng_state()
        maps, logits = wrapper.forward_depth(image, sparse, return_all_outputs=True,
                                              return_rejection_outputs=True)
        torch.set_rng_state(rng)
        without_aux = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        self.assertEqual(len(normal), 16)
        for a, b in zip(maps, without_aux):
            torch.testing.assert_close(a, b)
        value, info = wrapper.compute_loss_rejection(
            target, sparse, logits, supervision_levels=range(5))
        torch.testing.assert_close(value, torch.stack([
            info['loss_rejection_r{}'.format(r)] / (4 ** r) for r in range(5)]).sum())
        for r in range(5):
            n_selected = int((sparse_depth_at_resolution(sparse, logits[r].shape[-2:]) > 0).sum())
            self.assertEqual(int(info['rejection_reject_count_r{}'.format(r)]), n_selected)
            self.assertEqual(int(info['rejection_keep_count_r{}'.format(r)]), 0)
        value.backward()
        for stage in [wrapper.model_depth] + list(wrapper.model_depth.decoder):
            gradient = stage.rejection_map.conv.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.)
        # The same flip is applied to measurements and raw GT before selecting
        # labels. Selected original pixels still agree after augmentation.
        flipped_sparse, flipped_target = sparse.flip(-1), target.flip(-1)
        flipped = wrapper.forward_depth(image.flip(-1), flipped_sparse, return_all_outputs=True,
                                        return_rejection_outputs=True)[1]
        _, info = wrapper.compute_loss_rejection(
            flipped_target, flipped_sparse, flipped, supervision_levels=range(5))
        for r, scores in enumerate(flipped):
            n_selected = int((sparse_depth_at_resolution(flipped_sparse, scores.shape[-2:]) > 0).sum())
            self.assertEqual(int(info['rejection_reject_count_r{}'.format(r)]), n_selected)

    def test_rejection_l1_penalizes_kept_valid_points_not_missing_pixels(self):
        sparse = torch.tensor([[[[5., 0., 8., 0.]]]])
        keep = torch.tensor([[[[1., 0., 0., 0.]]]], requires_grad=True)
        value = loss_utils.rejection_sparsity_loss_func(keep, sparse)
        self.assertEqual(float(value), 0.5)
        value.backward()
        torch.testing.assert_close(keep.grad, torch.tensor([[[[0.5, 0., 0.5, 0.]]]]))
        self.assertEqual(float(loss_utils.rejection_sparsity_loss_func(torch.zeros_like(keep), sparse)), 0.)
        self.assertEqual(float(loss_utils.rejection_sparsity_loss_func(torch.ones_like(keep), sparse)), 1.)
        empty = torch.zeros_like(sparse, requires_grad=True)
        value = loss_utils.rejection_sparsity_loss_func(empty, torch.zeros_like(sparse))
        self.assertEqual(float(value), 0.)
        value.backward()
        torch.testing.assert_close(empty.grad, torch.zeros_like(empty))

    def test_rejection_l1_defaults_to_all_levels_and_respects_selection(self):
        wrapper = self.make_wrapper(True)
        sparse = torch.ones(1, 1, 17, 25)
        sizes = [(17, 25), (9, 13), (5, 7), (3, 4), (2, 2)]
        for levels in [None, (0, 3)]:
            masks = [torch.full((1, 1, h, w), float(r % 2), requires_grad=True)
                     for r, (h, w) in enumerate(sizes)]
            value, info = wrapper.compute_loss_rejection(
                None, sparse, masks, supervision_levels=levels, loss_type='l1')
            selected = range(5) if levels is None else levels
            expected = sum(float(r % 2) / (4 ** r) for r in selected)
            self.assertAlmostEqual(float(value), expected, places=6)
            value.backward()
            for r, mask in enumerate(masks):
                self.assertEqual(mask.grad is not None, r in selected)
                self.assertEqual('rejection_rejected_fraction_r{}'.format(r) in info, r in selected)
                self.assertEqual('rejection_kept_fraction_r{}'.format(r) in info, r in selected)
            self.assertFalse(any(k.startswith('rejection_reject_count') for k in info))
        scores = [torch.zeros(1, 2, 1, 1, requires_grad=True)]
        value, _ = wrapper.compute_loss_rejection(None, sparse, scores, loss_type='none')
        self.assertEqual(float(value), 0.)
        value.backward()
        torch.testing.assert_close(scores[0].grad, torch.zeros_like(scores[0]))

    def test_rejection_l1_uses_actual_forward_masks_without_resampling(self):
        wrapper = self.make_wrapper(True)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 11] = 5.
        image = torch.rand(1, 3, 17, 25)
        masks_used = []
        handles = [stage.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: masks_used.append(inputs[3]))
            for stage in [wrapper.model_depth] + list(wrapper.model_depth.decoder)]
        rng = torch.get_rng_state()
        outputs, masks = wrapper.forward_depth(
            image, sparse, return_all_outputs=True, return_rejection_masks=True)
        after = torch.get_rng_state()
        for returned, used in zip(masks, reversed(masks_used)):
            self.assertIs(returned, used)
        for handle in handles:
            handle.remove()
        torch.set_rng_state(rng)
        ordinary = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        for a, b in zip(outputs, ordinary):
            torch.testing.assert_close(a, b)
        loss, _ = wrapper.compute_loss_rejection(None, sparse, masks, loss_type='l1')
        loss.backward()
        for stage in [wrapper.model_depth] + list(wrapper.model_depth.decoder):
            gradient = stage.rejection_map.conv.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_rejection_supervision_defaults_to_all_levels_with_resolution_weights(self):
        wrapper = self.make_wrapper(True)
        sparse = torch.full((1, 1, 17, 25), 20.)
        target = torch.full_like(sparse, 5.)
        for levels in [tuple(range(5)), (0,), (0, 1)]:
            logits = [torch.zeros(1, 2, h, w, requires_grad=True)
                      for h, w in [(17, 25), (9, 13), (5, 7), (3, 4), (2, 2)]]
            if levels == tuple(range(5)):
                loss, info = wrapper.compute_loss_rejection(target, sparse, logits)
            else:
                loss, info = wrapper.compute_loss_rejection(
                    target, sparse, logits, supervision_levels=levels)
            torch.testing.assert_close(loss, torch.stack([
                info['loss_rejection_r{}'.format(r)] / (4 ** r) for r in levels]).sum())
            loss.backward()
            for r, scores in enumerate(logits):
                self.assertEqual(scores.grad is not None, r in levels)
                self.assertEqual('loss_rejection_r{}'.format(r) in info, r in levels)

    def test_configurable_propagation_and_refinement_counts(self):
        image, sparse = torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 11] = 5.
        for iterations, passes in [((1, 2, 3, 4, 6), 1), ((0, 0, 0, 0, 0), 0),
                                    ((2, 4, 6, 8, 12), 2)]:
            model = ScalePropagationDepthModel(
                n_channels=8, use_outlier_rejection=False,
                n_propagation_iterations=iterations, n_scale_refinement=passes).eval()
            stages = [model] + list(model.decoder)
            self.assertEqual([s.scale_propagation.n_propagation_iterations for s in stages],
                             list(iterations))
            with mock.patch('scale_propagation_model.ScaleUpdateBlock.forward',
                            autospec=True, side_effect=lambda module, x: module.convolutions(x)) as update:
                output = model(image, sparse)
            self.assertEqual(update.call_count, 5 * passes)
            self.assertEqual(output.shape, sparse.shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_encoder_uses_three_convolutions_per_level_without_attention_or_position(self):
        encoder = SharedEncoder(n_channels=8, n_head=4)
        with mock.patch('scale_propagation_model.add_position_encoding') as position, \
                mock.patch('scale_propagation_model.SelfAttentionBlock.forward') as attention:
            features = encoder(torch.rand(1, 3, 17, 25))
        position.assert_not_called()
        attention.assert_not_called()
        self.assertEqual([tuple(feature.shape) for feature in features], [
            (1, 8, 17, 25), (1, 8, 9, 13), (1, 8, 5, 7),
            (1, 8, 3, 4), (1, 8, 2, 2)])
        self.assertEqual(encoder.level0.convolutions[0].conv.in_channels, 3)
        for level in range(5):
            convolutions = getattr(encoder, 'level{}'.format(level)).convolutions
            self.assertEqual(len(convolutions), 3)
            self.assertEqual([block.conv.stride for block in convolutions],
                             [(1, 1) if level == 0 else (2, 2), (1, 1), (1, 1)])
        sum(feature.square().mean() for feature in features).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in encoder.parameters()))

    def test_optional_encoder_has_requested_convolutions_and_attention_order(self):
        encoder = SharedEncoder(n_channels=8, n_head=4, encoder_type='conv_attention')
        order = []
        for level, count in enumerate([3, 6, 6, 4, 4]):
            block = getattr(encoder, 'level{}'.format(level))
            self.assertEqual(len(block.convolutions), count)
            self.assertEqual([c.conv.stride for c in block.convolutions],
                             [(1, 1) if level == 0 else (2, 2)] + [(1, 1)] * (count - 1))
            block.register_forward_hook(
                lambda module, inputs, output, level=level: order.append('conv{}'.format(level)))
        for level in [3, 4]:
            attention = getattr(encoder, 'level{}_attention'.format(level))
            self.assertEqual(len(attention), 2)
            for index, block in enumerate(attention):
                block.register_forward_hook(lambda module, inputs, output, level=level, index=index:
                                            order.append('sa{}.{}'.format(level, index)))
        with mock.patch('scale_propagation_model.add_position_encoding',
                        wraps=sys.modules['scale_propagation_model'].add_position_encoding) as position:
            features = encoder(torch.rand(1, 3, 17, 25))
        self.assertEqual(position.call_count, 2)
        self.assertEqual(encoder.level0.convolutions[0].conv.in_channels, 3)
        self.assertEqual(order, ['conv0', 'conv1', 'conv2', 'conv3', 'sa3.0', 'sa3.1',
                                 'conv4', 'sa4.0', 'sa4.1'])
        self.assertEqual([tuple(f.shape) for f in features], [
            (1, 8, 17, 25), (1, 8, 9, 13), (1, 8, 5, 7), (1, 8, 3, 4), (1, 8, 2, 2)])
        sum(f.square().mean() for f in features).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in encoder.parameters()))

    def test_encoder_default_and_explicit_conv_have_identical_parameters(self):
        rng = torch.get_rng_state()
        default = SharedEncoder(n_channels=8, n_head=4)
        torch.set_rng_state(rng)
        explicit = SharedEncoder(n_channels=8, n_head=4, encoder_type='conv')
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        self.assertFalse(any('attention' in key for key in default.state_dict()))
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)

    def test_decoder_fusion_and_geometry_update_do_not_receive_raw_sparse_depth(self):
        block = PropagationDecoderBlock(
            n_channels=8, gumbel_temperature=1., use_outlier_rejection=False,
            n_scale_refinement=0).eval()
        self.assertEqual(block.fusion.convolutions[0].conv.in_channels, 18)
        feature = torch.rand(1, 8, 3, 4)
        skip = torch.rand(1, 8, 5, 7)
        geometry = torch.rand(1, 1, 3, 4) + 0.5
        scale = torch.full_like(geometry, 2.)
        fusion_inputs = []
        handle = block.fusion.register_forward_pre_hook(
            lambda module, inputs: fusion_inputs.append(inputs[0].clone()))
        with torch.no_grad():
            empty = block(feature, skip, geometry, scale, torch.zeros(1, 1, 17, 25))
            measured = block(feature, skip, geometry, scale, torch.full((1, 1, 17, 25), 7.))
        handle.remove()
        # Hold inherited state fixed: changing only raw sparse depth cannot
        # change fused features or Z. It still changes the explicit scale path.
        torch.testing.assert_close(fusion_inputs[0], fusion_inputs[1], rtol=0, atol=0)
        torch.testing.assert_close(empty[0], measured[0], rtol=0, atol=0)
        torch.testing.assert_close(empty[1], measured[1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(empty[2], measured[2]))

    def test_optional_encoder_selection_through_repository_wrapper(self):
        from depth_completion_model import DepthCompletionModel
        model = DepthCompletionModel(
            model_name='scale_propagation_kitti', network_modules=[],
            min_predict_depth=1.5, max_predict_depth=100., device=torch.device('cpu'),
            use_outlier_rejection=False, encoder_type='conv_attention',
            use_sparse_depth_features=True)
        self.assertEqual(model.model.model_depth.encoder.encoder_type, 'conv_attention')
        self.assertTrue(model.model.model_depth.use_sparse_depth_features)
        sparse = torch.zeros(1, 1, 17, 25)
        output = model.forward_depth(torch.rand(1, 3, 17, 25), sparse, torch.zeros_like(sparse))
        self.assertEqual(output.shape, sparse.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_inference_cli_forwards_encoder_selection(self):
        import runpy
        script = os.path.join(os.path.dirname(__file__), '..', 'src', 'run_depth_completion.py')
        for selected in [None, 'conv_attention']:
            argv = [script, '--image_path', 'images.txt', '--sparse_depth_path', 'sparse.txt',
                    '--intrinsics_path', 'intrinsics.txt', '--output_path', 'outputs', '--device', 'cpu']
            if selected is not None:
                argv += ['--encoder_type', selected, '--use_sparse_depth_features', 'true']
            with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.run') as run:
                runpy.run_path(script, run_name='__main__')
            self.assertEqual(run.call_args.kwargs['encoder_type'], selected or 'conv')
            self.assertEqual(run.call_args.kwargs['use_sparse_depth_features'], selected is not None)

    def test_sparse_feature_flag_controls_encoder_and_all_decoder_inputs(self):
        image = torch.rand(1, 3, 17, 25)
        sparse = torch.rand(1, 1, 17, 25) * 5.
        for encoder_type in ['conv', 'conv_attention']:
            for enabled in [False, True]:
                model = ScalePropagationDepthModel(
                    n_channels=8, encoder_type=encoder_type,
                    use_sparse_depth_features=enabled).eval()
                self.assertEqual(model.encoder.level0.convolutions[0].conv.in_channels,
                                 4 if enabled else 3)
                for block in model.decoder:
                    self.assertEqual(block.fusion.convolutions[0].conv.in_channels,
                                     19 if enabled else 18)
                seen = []
                handle = model.encoder.level0.register_forward_pre_hook(
                    lambda module, inputs: seen.append(inputs[0].clone()))
                fusion = []
                model.decoder[-1].fusion.register_forward_pre_hook(
                    lambda module, inputs: fusion.append(inputs[0].clone()))
                with torch.no_grad():
                    output = model(image, sparse)
                handle.remove()
                torch.testing.assert_close(seen[0][:, :3], image)
                if enabled:
                    torch.testing.assert_close(seen[0][:, 3:], sparse)
                    torch.testing.assert_close(fusion[0][:, -1:], sparse)
                self.assertEqual(output.shape, sparse.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_fixed_seed_repeats_all_rngs_and_keeps_data_order_independent_of_model(self):
        import random
        import numpy as np
        from depth_completion import set_random_seed, seed_worker
        results = []
        for _ in range(2):
            set_random_seed(17)
            results.append((random.random(), np.random.rand(), torch.rand(5),
                rejection_keep_mask(torch.zeros(1, 2, 3, 4), torch.ones(1, 1, 3, 4),
                                    1., training=True)))
        self.assertEqual(results[0][:2], results[1][:2])
        for a, b in zip(results[0][2:], results[1][2:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        orders = []
        for enabled in [False, True]:
            set_random_seed(17)
            ScalePropagationDepthModel(n_channels=8, use_sparse_depth_features=enabled)
            loader = torch.utils.data.DataLoader(
                torch.arange(16), batch_size=4, shuffle=True,
                generator=torch.Generator().manual_seed(17), worker_init_fn=seed_worker)
            orders.append(torch.cat(list(loader)))
        torch.testing.assert_close(orders[0], orders[1], rtol=0, atol=0)
        with mock.patch('torch.initial_seed', return_value=1234):
            seed_worker(0)
            first = (random.random(), np.random.rand())
            seed_worker(1)
            self.assertEqual(first, (random.random(), np.random.rand()))

    def test_guided_scale_initialization_matches_bilinear_for_arbitrary_shapes(self):
        upsampling = FeatureGuidedScaleUpsampling(n_channels=3)
        for coarse_size, fine_size in [
                ((2, 3), (4, 6)), ((3, 4), (5, 7)),
                ((1, 3), (1, 7)), ((3, 1), (7, 1)),
                ((1, 1), (3, 5)), ((3, 4), (3, 4))]:
            scale = torch.rand(2, 1, *coarse_size) * 30. + 1.
            skip = torch.randn(2, 3, *fine_size)
            expected = functional.interpolate(
                scale, size=fine_size, mode='bilinear', align_corners=True)
            torch.testing.assert_close(upsampling(scale, skip), expected)

    def test_decoder_scale_upsampling_is_bilinear_despite_guided_weights(self):
        block = PropagationDecoderBlock(8, 1., False)
        with torch.no_grad():
            block.scale_upsampling.weight_prediction.conv.weight.normal_(std=10.)
        scale = torch.rand(1, 1, 3, 4) + 1.
        seen = []
        block.fusion.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0][:, -1:].clone()))
        with mock.patch.object(block.scale_upsampling, 'forward',
                               side_effect=AssertionError('guided upsampling is inactive')):
            block(torch.rand(1, 8, 3, 4), torch.rand(1, 8, 5, 7),
                  torch.ones_like(scale), scale, torch.zeros(1, 1, 5, 7))
        expected = functional.interpolate(
            scale, size=(5, 7), mode='bilinear', align_corners=True)
        torch.testing.assert_close(seen[0], expected)

    def test_guided_scale_weights_use_fine_features_to_choose_a_surface(self):
        upsampling = FeatureGuidedScaleUpsampling(n_channels=1)
        with torch.no_grad():
            # Positive guidance selects left neighbors; negative selects right.
            upsampling.weight_prediction.conv.weight[[0, 2], 0, 1, 1] = 4.
            upsampling.weight_prediction.conv.weight[[1, 3], 0, 1, 1] = -4.
        scale = torch.tensor([[[[10., 30.], [10., 30.]]]]).expand(2, -1, -1, -1)
        skip = torch.ones(2, 1, 3, 3)
        skip[1] = -1.
        result = upsampling(scale, skip)
        self.assertLess(float(result[0, 0, 1, 1]), 10.01)
        self.assertGreater(float(result[1, 0, 1, 1]), 29.99)
        self.assertTrue(((result >= 10. - 1e-5) & (result <= 30. + 1e-5)).all())

    def test_guided_scale_preserves_constants_and_has_finite_gradients(self):
        upsampling = FeatureGuidedScaleUpsampling(n_channels=3)
        with torch.no_grad():
            upsampling.weight_prediction.conv.weight.normal_(std=0.2)
        skip = torch.randn(2, 3, 5, 7, requires_grad=True)
        scale = (torch.rand(2, 1, 3, 4) + 1.).requires_grad_()
        constant = torch.tensor([2., 50.]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(
            upsampling(constant.expand(2, 1, 3, 4), skip),
            constant.expand(2, 1, 5, 7))
        upsampling(scale, skip).square().mean().backward()
        for gradient in [scale.grad, skip.grad,
                         upsampling.weight_prediction.conv.weight.grad]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_selects_first_valid_value_instead_of_averaging(self):
        sparse = torch.tensor([[[[2., 40., 0., 60.],
                                [70., 80., 90., 100.],
                                [0., 110., 6., 120.],
                                [130., 140., 150., 160.]]]])
        sampled = sparse_depth_at_resolution(sparse, (2, 2))
        self.assertTrue(torch.equal(
            sampled, torch.tensor([[[[2., 60.], [110., 6.]]]])))
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (4, 4)), sparse))

    def test_preserves_measurements_between_nearest_sampling_coordinates(self):
        sparse = torch.zeros(2, 1, 4, 4)
        sparse[0, 0, 1, 1] = 3.
        sparse[0, 0, 1, 3] = 5.
        sparse[0, 0, 3, 1] = 7.
        sparse[0, 0, 3, 3] = 9.
        expected = torch.tensor([[[[3., 5.], [7., 9.]]],
                                 [[[0., 0.], [0., 0.]]]])
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (2, 2)), expected))

    def test_selects_valid_measurements_for_odd_shapes(self):
        sparse = torch.zeros(1, 1, 5, 7)
        sparse[0, 0, 0, 1] = 3.
        sparse[0, 0, 4, 5] = 7.
        expected = torch.tensor([[[[3., 0., 0.], [0., 0., 7.]]]])
        self.assertTrue(torch.equal(sparse_depth_at_resolution(sparse, (2, 3)), expected))

    def test_one_source_sends_absolute_scale_to_the_right(self):
        scale = torch.full((1, 1, 5, 5), 2.)
        geometry = torch.ones_like(scale) * 0.5
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 3.
        logits = torch.full((1, 9, 5, 5), -100.)
        logits[0, 5, 2, 2] = 100.
        result = propagate_scale(scale, sparse / geometry, (sparse > 0).float(), logits)
        expected = scale.clone()
        # The source injects 3 / 0.5 = 6; its right neighbor averages 6 and 2.
        expected[0, 0, 2, 2] = 6.
        expected[0, 0, 2, 3] = 4.
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(scale, torch.full_like(scale, 2.))

    def test_geometry_sampling_matches_upsampling_and_gathering_original_pixels(self):
        sparse = torch.zeros(2, 1, 5, 7)
        sparse[0, 0, 1, 1] = 6.
        sparse[0, 0, 4, 5] = 10.
        sparse[1, 0, 0, 2] = 12.
        sparse[1, 0, 3, 4] = 15.
        geometry = torch.arange(1., 13.).reshape(2, 1, 2, 3).requires_grad_()
        _, indices = sparse_depth_at_resolution(sparse, (2, 3), return_indices=True)
        sampled = geometry_at_sparse_points(geometry, indices, sparse.shape[-2:])
        full_geometry = functional.interpolate(
            geometry.detach(), size=sparse.shape[-2:], mode='bilinear', align_corners=True)
        expected = torch.gather(full_geometry.flatten(2), 2, indices.flatten(2)).reshape_as(sampled)
        torch.testing.assert_close(sampled, expected)
        self.assertFalse(sampled.requires_grad)

    def test_geometry_sampling_handles_full_resolution_and_single_pixel_axes(self):
        for input_size, level_size in [((3, 5), (3, 5)), ((1, 7), (1, 3)),
                                       ((7, 1), (3, 1)), ((1, 1), (1, 1))]:
            sparse = torch.ones(2, 1, *input_size)
            geometry = torch.rand(2, 1, *level_size) + 0.5
            _, indices = sparse_depth_at_resolution(sparse, level_size, return_indices=True)
            sampled = geometry_at_sparse_points(geometry, indices, input_size)
            full_geometry = functional.interpolate(
                geometry, size=input_size, mode='bilinear', align_corners=True)
            expected = torch.gather(full_geometry.flatten(2), 2, indices.flatten(2)).reshape_as(sampled)
            torch.testing.assert_close(sampled, expected)

    def test_current_cell_scale_ratio_is_injected_directly_without_redivision(self):
        sparse = torch.zeros(1, 1, 4, 4)
        sparse[0, 0, 1, 1] = 6.
        geometry = torch.tensor([[[[1., 3.], [2., 4.]]]])
        selected, indices = sparse_depth_at_resolution(sparse, (2, 2), return_indices=True)
        sampled = geometry_at_sparse_points(geometry, indices, sparse.shape[-2:])
        # At original pixel (1, 1), interpolated Z is 2, not coarse-cell Z = 1.
        self.assertAlmostEqual(float(sampled[0, 0, 0, 0]), 2., places=6)
        target = selected / geometry.detach().clamp_min(1e-7)
        scale = torch.full_like(geometry, 2.)
        logits = torch.full((1, 9, 2, 2), -100.)
        propagated = propagate_scale(scale, target, (selected > 0).float(), logits)
        expected = scale.clone()
        expected[0, 0, 0, 0] = 6.
        torch.testing.assert_close(propagated, expected)
        self.assertNotEqual(float(target[0, 0, 0, 0]), float((selected / sampled)[0, 0, 0, 0]))

    def test_every_level_uses_current_cell_ratios_and_bottom_reuses_them_for_initialization(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        observed, raw_bottom = [], []
        handles = [model.coarse_scaleless_depth.register_forward_hook(
            lambda module, inputs, output: raw_bottom.append(output))]
        for stage in [model] + list(model.decoder):
            handles.append(stage.scale_propagation.register_forward_pre_hook(
                lambda module, inputs: observed.append((module.n_propagation_iterations, inputs))))
        sparse = torch.zeros(2, 1, 17, 25)
        sparse[0, 0, 7, 11] = 5.
        sparse[1, 0, 16, 24] = 7.
        with torch.no_grad():
            model(torch.rand(2, 3, 17, 25), sparse)
        for handle in handles:
            handle.remove()
        self.assertEqual([iteration for iteration, _ in observed], [2, 4, 6, 8, 12])
        for level, (_, inputs) in enumerate(observed):
            scale, geometry, selected, keep, logits, ratio = inputs
            expected_sparse = sparse_depth_at_resolution(sparse, geometry.shape[-2:])
            torch.testing.assert_close(selected, expected_sparse)
            epsilon = 1e-3 if level == 0 else 1e-7
            torch.testing.assert_close(ratio, selected / geometry.detach().clamp_min(epsilon))
            self.assertEqual(ratio.shape, scale.shape)
        scale, _, _, keep, _, ratio = observed[0][1]
        fallback = (functional.softplus(raw_bottom[0]) + 1e-3).mean(dim=[2, 3], keepdim=True)
        expected = ((keep * ratio).sum(dim=[2, 3], keepdim=True) + fallback) / (
            keep.sum(dim=[2, 3], keepdim=True) + 1.)
        torch.testing.assert_close(scale, expected.expand_as(scale))
        self.assertFalse(any('surface_compatibility' in name for name in model.state_dict()))

    def test_scales_relay_beyond_the_original_source_neighborhood(self):
        scale = torch.full((1, 1, 5, 7), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float()
        logits = torch.full((1, 9, 5, 7), -100.)
        logits[0, 5, 2, 2] = 100.
        logits[0, 5, 2, 3] = 100.
        first = propagate_scale(scale, sparse, keep, logits, 1)
        second = propagate_scale(scale, sparse, keep, logits, 2)
        self.assertEqual(float(first[0, 0, 2, 4]), 2.)
        self.assertEqual(float(first[0, 0, 2, 3]), 4.)
        self.assertEqual(float(second[0, 0, 2, 4]), 3.)
        # A rejected source can relay scale, but cannot inject its measurement.
        sparse[0, 0, 2, 3] = 100.
        torch.testing.assert_close(
            propagate_scale(scale, sparse, keep, logits, 2), second)

    def test_measurements_are_not_restored_during_or_after_propagation(self):
        scale = torch.full((1, 1, 5, 5), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        sparse[0, 0, 2, 3] = 10.
        logits = torch.zeros(1, 9, 5, 5)
        # Neighboring dense priors also contribute: the center becomes
        # (1.5 * 6 + 0.5 * 10 + 7 * 0.5 * 2) / 5.5, not restored scale 6.
        first = propagate_scale(
            scale, sparse, (sparse > 0).float(), logits, 1)
        self.assertAlmostEqual(float(first[0, 0, 2, 2]), 21. / 5.5, places=6)
        second = propagate_scale(
            scale, sparse, (sparse > 0).float(), logits, 2)
        weight = functional.fold(torch.sigmoid(logits).reshape(1, 9, 25),
                                 (5, 5), kernel_size=3, padding=1)
        expected = torch.where(sparse > 0, sparse, scale)
        for _ in range(2):
            expected = (expected + functional.fold(
                (0.5 * expected).expand(1, 9, 5, 5).reshape(1, 9, 25),
                (5, 5), kernel_size=3, padding=1)) / (1 + weight)
        torch.testing.assert_close(second, expected)
        self.assertNotEqual(float(second[0, 0, 2, 2]), 6.)

    def test_more_iterations_relay_and_average_the_actual_scale(self):
        scale = torch.full((1, 1, 33, 33), 2.)
        target = torch.zeros_like(scale)
        target[0, 0, 16, 16] = 6.
        keep = (target > 0).float()
        logits = torch.zeros(1, 9, 33, 33)
        short = propagate_scale(scale, target, keep, logits, 2)
        longer = propagate_scale(scale, target, keep, logits, 12)
        self.assertGreater(int((longer > 2. + 1e-6).sum()), int((short > 2. + 1e-6).sum()))
        self.assertLess(float(longer.max()), float(short.max()))
        self.assertTrue(torch.isfinite(longer).all())
        self.assertTrue(((longer >= 2.) & (longer <= 6.)).all())
        # Unlike normalized anchor messages, the dense prior participates.
        changed = propagate_scale(scale * 100., target, keep, logits, 12)
        self.assertFalse(torch.allclose(changed, longer))

    def test_no_edges_inject_sources_once_and_leave_input_prior_untouched(self):
        scale = torch.rand(1, 1, 5, 7) + 1.
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        logits = torch.full((1, 9, 5, 7), -100.)
        original = scale.clone()
        expected = torch.where(sparse > 0, sparse, scale)
        torch.testing.assert_close(
            propagate_scale(scale, sparse,
                            (sparse > 0).float(), logits, 6), expected)
        torch.testing.assert_close(scale, original)

    def test_constant_scale_stays_constant_with_empty_or_rejected_sources(self):
        scale = torch.full((2, 1, 5, 7), 3.)
        geometry = torch.rand_like(scale) + 0.1
        sparse = torch.full_like(scale, 50.)
        logits = torch.randn(2, 9, 5, 7)
        keep = torch.zeros_like(scale, requires_grad=True)
        result = propagate_scale(scale, sparse / geometry, keep, logits, 6)
        torch.testing.assert_close(result, torch.full_like(scale, 3.))
        torch.testing.assert_close(scale, torch.full_like(scale, 3.))
        result.sum().backward()
        self.assertTrue(torch.isfinite(keep.grad).all())
        self.assertGreater(float(keep.grad.abs().sum()), 0.)

    def test_outgoing_boundary_messages_do_not_wrap(self):
        scale = torch.full((1, 1, 3, 3), 2.)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 0, 0] = 6.
        logits = torch.full((1, 9, 3, 3), -100.)
        logits[0, 0, 0, 0] = 100.
        expected = torch.where(sparse > 0, sparse, scale)
        torch.testing.assert_close(
            propagate_scale(scale, sparse,
                            (sparse > 0).float(), logits, 4), expected)

    def test_propagation_trains_weights_keep_and_prior_but_not_detached_geometry(self):
        scale = torch.full((1, 1, 5, 7), 2., requires_grad=True)
        geometry = torch.ones_like(scale, requires_grad=True)
        sparse = torch.zeros_like(scale)
        sparse[0, 0, 2, 2] = 6.
        keep = (sparse > 0).float().requires_grad_()
        logits = torch.zeros(1, 9, 5, 7, requires_grad=True)
        result = propagate_scale(scale, sparse / geometry.detach().clamp_min(1e-7), keep, logits, 4)
        result.square().sum().backward()
        self.assertIsNone(geometry.grad)
        for value in (keep, logits, scale):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_single_anchor_keep_gradient_does_not_cancel(self):
        scale = torch.full((1, 1, 3, 3), 2.)
        ratio = torch.zeros_like(scale)
        ratio[0, 0, 1, 1] = 6.
        keep = torch.zeros_like(scale, requires_grad=True)
        with torch.no_grad():
            keep[0, 0, 1, 1] = 0.5
        output = propagate_scale(scale, ratio, keep, torch.zeros(1, 9, 3, 3), 0)
        self.assertEqual(float(output[0, 0, 1, 1]), 4.)
        output[0, 0, 1, 1].backward()
        self.assertEqual(float(keep.grad[0, 0, 1, 1]), 4.)

    def test_rejection_input_normalizes_features_without_mutating_them(self):
        feature = torch.arange(8.).reshape(1, 8, 1, 1).expand(1, 8, 3, 4).clone()
        original = feature.clone()
        feature.requires_grad_()
        depth = torch.full((1, 1, 3, 4), 60.)
        sparse = torch.full_like(depth, 55.)
        result = normalize_rejection_input(feature, depth / 100., sparse / 100.)
        self.assertTrue(torch.equal(feature, original))
        torch.testing.assert_close(result[:, :8].mean(dim=1), torch.zeros(1, 3, 4))
        torch.testing.assert_close(result[:, :8].var(dim=1, unbiased=False), torch.ones(1, 3, 4))
        torch.testing.assert_close(result[:, -2:-1], torch.full_like(depth, 0.6))
        result[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_only_rejection_receives_scaled_depth_at_every_level(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        coarse_predictions = []
        model.coarse_scaleless_depth.register_forward_hook(
            lambda module, inputs, output: coarse_predictions.append(functional.softplus(output) + 1e-3))
        captured = []
        for stage in [model] + list(model.decoder):
            pair = dict(rejection=[], propagation=[])
            stage.rejection_map.register_forward_pre_hook(
                lambda module, inputs, pair=pair: pair['rejection'].append(inputs[0].clone()))
            stage.propagation_map.register_forward_pre_hook(
                lambda module, inputs, pair=pair: pair['propagation'].append(inputs[0].clone()))
            captured.append(pair)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[0, 0, 7, 11] = 55.
        with torch.no_grad():
            model(torch.rand(1, 3, 17, 25), sparse)
        for level, pair in enumerate(captured):
            rejection, propagation = pair['rejection'][0], pair['propagation'][0]
            torch.testing.assert_close(rejection[:, :8].mean(dim=1),
                                       torch.zeros_like(rejection[:, 0]), atol=1e-5, rtol=0.)
            selected, indices = sparse_depth_at_resolution(
                sparse, rejection.shape[-2:], return_indices=True)
            torch.testing.assert_close(rejection[:, 9:10], selected / 100.)
            expected_depth = feature_at_sparse_points(
                (coarse_predictions[0] if level == 0 else propagation[:, -2:-1]) / 100.,
                indices, sparse.shape[-2:])
            torch.testing.assert_close(rejection[:, 8:9], expected_depth)
            self.assertEqual(rejection.shape[1], 12)

    def test_measurement_rejection_input_samples_depth_and_encodes_original_offsets(self):
        feature = torch.rand(1, 8, 2, 2, requires_grad=True)
        depth = torch.tensor([[[[1., 3.], [5., 7.]]]], requires_grad=True)
        sparse = torch.zeros(1, 1, 8, 8)
        sparse[0, 0, 0, 0] = 10.
        sparse[0, 0, 3, 7] = 20.
        selected, indices = sparse_depth_at_resolution(sparse, (2, 2), return_indices=True)
        result = measurement_rejection_input(feature, depth, selected, indices, (8, 8))
        self.assertEqual(result.shape, (1, 12, 2, 2))
        torch.testing.assert_close(result[:, 8:9], feature_at_sparse_points(depth, indices, (8, 8)))
        torch.testing.assert_close(result[:, 9:10], selected)
        torch.testing.assert_close(result[0, 10:, 0, 0], torch.tensor([-.75, -.75]))
        torch.testing.assert_close(result[0, 10:, 0, 1], torch.tensor([.75, .75]))
        torch.testing.assert_close(result[0, 10:, 1], torch.zeros(2, 2))
        result.sum().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertTrue(torch.isfinite(feature.grad).all())
        # At full resolution, every measurement is at its own cell center.
        full = measurement_rejection_input(torch.rand(1, 8, 8, 8),
            torch.rand_like(sparse), sparse, torch.arange(64).reshape(1, 1, 8, 8), (8, 8))
        torch.testing.assert_close(full[:, -2:], torch.zeros(1, 2, 8, 8))
        # Odd image sizes use overlapping adaptive bins with floor/ceil edges.
        odd = torch.zeros(1, 1, 5, 7)
        odd[0, 0, 2, 3] = 10.
        selected, indices = sparse_depth_at_resolution(odd, (2, 2), return_indices=True)
        result = measurement_rejection_input(feature, depth, selected, indices, (5, 7))
        torch.testing.assert_close(result[0, -2:],
            torch.tensor([[[2./3., 2./3.], [-2./3., -2./3.]],
                          [[.75, -.75], [.75, -.75]]]))

    def test_rejection_class_balance_penalizes_false_rejection_more(self):
        labels = torch.tensor([[[[0, 1]]]])
        scores = torch.zeros(1, 2, 1, 2, requires_grad=True)
        equal = loss_utils.balanced_rejection_loss_func(scores, labels, keep_weight=1.)
        equal_gradient = torch.autograd.grad(equal, scores)[0]
        adjusted = loss_utils.balanced_rejection_loss_func(scores, labels)
        gradient = torch.autograd.grad(adjusted, scores)[0]
        self.assertGreater(float(gradient[0, 1, 0, 0]), float(equal_gradient[0, 1, 0, 0]))
        self.assertLess(float(gradient[0, 0, 0, 1]), float(equal_gradient[0, 0, 0, 1]))

    def test_rejection_masks_missing_points_and_has_finite_gradients(self):
        sparse = torch.tensor([[[[0., 3.], [4., 0.]]]])
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        keep = rejection_keep_mask(logits, sparse, 1., training=True)
        self.assertTrue(torch.equal(keep[sparse == 0], torch.zeros(2)))
        self.assertTrue(((keep == 0) | (keep == 1)).all())
        keep.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_validation_uses_same_gumbel_sample_as_training_with_same_seed(self):
        logits = torch.zeros(1, 2, 16, 16)
        sparse = torch.ones(1, 1, 16, 16)
        sparse[:, :, 0] = 0.
        torch.manual_seed(17)
        training = rejection_keep_mask(logits, sparse, 1., training=True)
        torch.manual_seed(17)
        with torch.no_grad():
            validation = rejection_keep_mask(logits, sparse, 1., training=False)
            another = rejection_keep_mask(logits, sparse, 1., training=False)
        torch.testing.assert_close(training, validation, rtol=0, atol=0)
        self.assertTrue((validation[sparse > 0] == 0).any())
        self.assertTrue((validation[sparse > 0] == 1).any())
        self.assertTrue((validation[sparse == 0] == 0).all())
        self.assertFalse(torch.equal(validation, another))

    def make_wrapper(self, rejection=False):
        wrapper = ScalePropagationDepthCompletionModel.__new__(ScalePropagationDepthCompletionModel)
        wrapper.min_predict_depth = 0.01
        wrapper.max_predict_depth = 100.
        wrapper.model_depth = ScalePropagationDepthModel(
            n_channels=8, min_predict_depth=0.01, max_predict_depth=100.,
            use_outlier_rejection=rejection)
        return wrapper

    def test_odd_shapes_empty_inputs_and_backward(self):
        torch.manual_seed(4)
        for rejection in (True, False):
            wrapper = self.make_wrapper(rejection)
            image = torch.rand(2, 3, 17, 25)
            sparse = torch.zeros(2, 1, 17, 25)
            sparse[1, 0, 7, 11] = 5.
            outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
            self.assertEqual(len(outputs), 16)
            self.assertTrue(all(value.shape == sparse.shape for value in outputs))
            self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
            target = torch.rand_like(sparse) * 5 + 1.
            loss, info = wrapper.compute_loss_supervised(target, outputs, {})
            loss.backward()
            gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(wrapper.model_depth.coarse_scaleless_depth.conv.weight.grad.abs().sum()), 0.)
            self.assertEqual(len([key for key in info if key.startswith('loss_si_r')]), 5)

    def test_fresh_rejection_at_each_level_with_confident_gumbel_eval(self):
        model = ScalePropagationDepthModel(n_channels=8).eval()
        decisions = []
        with torch.no_grad():
            model.rejection_map.conv.weight.zero_()
            model.rejection_bias.copy_(torch.tensor([-100., 100.]).view(1, 2, 1, 1))
            for i, block in enumerate(model.decoder):
                block.rejection_map.conv.weight.zero_()
                bias = [100., -100.] if i % 2 == 0 else [-100., 100.]
                block.rejection_bias.copy_(torch.tensor(bias).view(1, 2, 1, 1))
                block.register_forward_hook(
                    lambda module, inputs, output: decisions.append(output[-1].clone()))
            image = torch.rand(1, 3, 17, 25)
            sparse = torch.ones(1, 1, 17, 25) * 5.
            first = model(image, sparse)
            second = model(image, sparse)
        self.assertTrue(torch.equal(first, second))
        for i, keep in enumerate(decisions[:4]):
            self.assertTrue(torch.equal(keep, torch.full_like(keep, 1. if i % 2 == 0 else 0.)))

    def test_constant_bottom_scale_is_carried_to_every_level_without_measurements(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        received = []
        model.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: received.append(inputs[0].clone()))
        with torch.no_grad():
            outputs = model(torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25), True)
        initial = received[0]
        torch.testing.assert_close(initial, initial[:, :, :1, :1].expand_as(initial))
        for geometry, scale in zip(outputs['scaleless_depths'], outputs['scale_maps']):
            torch.testing.assert_close(geometry.mean(dim=[2, 3]), torch.ones(1, 1))
            torch.testing.assert_close(scale, initial[:, :, :1, :1].expand_as(scale))
        self.assertEqual(model.n_propagation_iterations, (2, 4, 6, 8, 12))
        self.assertEqual([block.n_propagation_iterations for block in model.decoder], [4, 6, 8, 12])
        for stage in [model] + list(model.decoder):
            self.assertEqual(int(torch.count_nonzero(
                stage.scale_update.convolutions[-1].conv.weight)), 0)

    def test_scale_head_can_change_scale_without_sparse_measurements(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        image, sparse = torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25)
        with torch.no_grad():
            before = model(image, sparse, True)['scale_maps'][0]
            model.decoder[-1].scale_update.convolutions[-1].conv.weight.fill_(0.1)
            after = model(image, sparse, True)['scale_maps'][0]
        self.assertFalse(torch.equal(before, after))

    def test_every_level_refines_after_propagation_without_restoring_accepted_scales(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        observations, order = [], []
        for level, stage in enumerate([model] + list(model.decoder)):
            record = {'refinements': []}
            observations.append(record)
            with torch.no_grad():
                stage.rejection_map.conv.weight.zero_()
                bias = [-100., 100.] if level == 1 else [100., -100.]
                stage.rejection_bias.copy_(torch.tensor(bias).reshape(1, 2, 1, 1))
                # Route the current scale through the nonlinear block so the
                # second pass must see a different scale and discrepancy.
                for convolution in stage.scale_update.convolutions:
                    convolution.conv.weight.zero_()
                stage.scale_update.convolutions[0].conv.weight[0, -4, 1, 1] = 1.
                stage.scale_update.convolutions[1].conv.weight[0, 0, 1, 1] = 1.
                stage.scale_update.convolutions[2].conv.weight[0, 0, 1, 1] = 0.05
            def propagated(module, inputs, output, record=record, level=level):
                order.append(('propagate', level))
                record.update(inherited=inputs[0], propagated=output, keep=inputs[3], ratio=inputs[5])
            def refined(module, inputs, output, record=record, level=level):
                order.append(('refine', level))
                record['refinements'].append((inputs[0], output))
            stage.scale_propagation.register_forward_hook(propagated)
            stage.scale_update.register_forward_hook(refined)
        sparse = torch.zeros(1, 1, 33, 49)
        sparse[0, 0, 9, 13] = 5.
        with torch.no_grad():
            outputs = model(torch.rand(1, 3, 33, 49), sparse, True)
        self.assertEqual(order, [(operation, level) for level in range(5)
                                 for operation in ['propagate', 'refine', 'refine']])
        for record, actual in zip(observations, reversed(outputs['scale_maps'])):
            inherited = record['inherited']
            propagated = record['propagated']
            keep, ratio = record['keep'], record['ratio']
            expected = propagated
            for context, update in record['refinements']:
                self.assertEqual(context.shape[1], 8 + 5)
                torch.testing.assert_close(context[:, -4:-3], expected)
                torch.testing.assert_close(context[:, -3:-2], keep * ratio)
                torch.testing.assert_close(context[:, -2:-1], keep)
                torch.testing.assert_close(context[:, -1:], keep * (ratio - expected))
                expected = expected * torch.exp(torch.tanh(update))
            torch.testing.assert_close(actual, expected)
            self.assertTrue((actual > propagated).all())
        self.assertEqual(int(observations[1]['keep'].sum()), 0)

    def test_post_propagation_correction_trains_at_accepted_and_unmeasured_pixels(self):
        block = PropagationDecoderBlock(8, 1., False, max_predict_depth=100.)
        updates = []
        def observe(module, inputs, output):
            output.retain_grad()
            updates.append(output)
        block.scale_update.register_forward_hook(observe)
        sparse = torch.zeros(1, 1, 5, 7)
        sparse[0, 0, 2, 3] = 5.
        output = block(torch.rand(1, 8, 3, 4), torch.rand(1, 8, 5, 7),
                       torch.ones(1, 1, 3, 4), torch.full((1, 1, 3, 4), 2.), sparse)
        output[2].sum().backward()
        self.assertEqual(len(updates), 2)
        for update in updates:
            gradient = update.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[sparse > 0] > 0).all())
            self.assertTrue((gradient[sparse == 0] > 0).all())
        self.assertGreater(float(
            block.scale_update.convolutions[-1].conv.weight.grad.abs().sum()), 0.)

    def test_metric_losses_train_scale_heads_but_not_geometry(self):
        model = self.make_wrapper().model_depth
        # Move past the identity initialization to exercise the message and
        # upsampling paths rather than a spatially constant initial scale.
        with torch.no_grad():
            for stage in [model] + list(model.decoder):
                stage.scale_update.convolutions[-1].conv.weight.fill_(0.01)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 11] = 5.
        outputs = model(torch.rand(1, 3, 17, 25), sparse, True)
        sum(torch.log1p((value - 3.).abs()).mean()
            for value in outputs['metric_depths_for_loss']).backward()
        for stage, name in [(model, 'coarse_scaleless_depth')] + [
                (block, 'depth_update') for block in model.decoder]:
            gradient = getattr(stage, name).conv.weight.grad
            self.assertIsNone(gradient)
        self.assertTrue(any(stage.scale_update.convolutions[-1].conv.weight.grad.abs().sum() > 0
                            for stage in [model] + list(model.decoder)))
        self.assertTrue(all(
            block.scale_upsampling.weight_prediction.conv.weight.grad is None
            for block in model.decoder))

    def test_detach_flag_through_repo_wrapper_controls_only_metric_geometry_gradients(self):
        from depth_completion_model import DepthCompletionModel
        detached = DepthCompletionModel(
            model_name='scale_propagation_kitti', network_modules=[],
            min_predict_depth=1.5, max_predict_depth=100., device=torch.device('cpu'),
            use_outlier_rejection=False, detach_scaleless_depth=True)
        attached = DepthCompletionModel(
            model_name='scale_propagation_kitti', network_modules=[],
            min_predict_depth=1.5, max_predict_depth=100., device=torch.device('cpu'),
            use_outlier_rejection=False, detach_scaleless_depth=False)
        attached.model.model_depth.load_state_dict(detached.model.model_depth.state_dict())
        image = torch.rand(1, 3, 17, 25)
        sparse = torch.ones(1, 1, 17, 25) * 5.
        outputs = []
        for wrapper, detach in [(detached, True), (attached, False)]:
            model = wrapper.model.model_depth
            self.assertEqual(model.detach_scaleless_depth, detach)
            result = wrapper.forward_depth(image, sparse, torch.ones_like(sparse), return_all_outputs=True)
            outputs.append(result)
            sum(torch.log1p((value - 3.).abs()).mean() for value in result[11:16]).backward()
            for stage, name in [(model, 'coarse_scaleless_depth')] + [
                    (block, 'depth_update') for block in model.decoder]:
                gradient = getattr(stage, name).conv.weight.grad
                if detach:
                    self.assertIsNone(gradient)
                else:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.abs().sum()), 0.)
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b)

    def test_si_losses_do_not_train_scale_propagation_or_rejection_heads(self):
        model = self.make_wrapper(True).model_depth
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, ::3, ::3] = 5.
        outputs = model(torch.rand(1, 3, 17, 25), sparse, True)
        loss = sum(loss_utils.scale_invariant_loss_func(
            value, torch.rand_like(value) + 1., torch.ones_like(value))
            for value in outputs['scaleless_depths'])
        loss.backward()
        for stage in [model] + list(model.decoder):
            for name in ['scale_update', 'propagation_map', 'rejection_map']:
                self.assertTrue(all(parameter.grad is None
                                    for parameter in getattr(stage, name).parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for block in model.decoder for parameter in block.scale_upsampling.parameters()))
        self.assertGreater(float(model.coarse_scaleless_depth.conv.weight.grad.abs().sum()), 0.)

    def test_si_loss_is_scale_invariant_and_ignores_invalid_targets(self):
        geometry = torch.tensor([[[[1., 2.], [4., 8.]]],
                                 [[[2., 4.], [3., 5.]]]], requires_grad=True)
        target = torch.tensor([[[[2., 5.], [7., 0.]]],
                               [[[3., 4.], [9., 0.]]]])
        weights = (target > 0).float()
        loss = loss_utils.scale_invariant_loss_func(geometry, target, weights)
        scaled = geometry * torch.tensor([7., 0.2]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(
            loss, loss_utils.scale_invariant_loss_func(scaled, target, weights))
        loss.backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertTrue((geometry.grad[target == 0] == 0).all())

    def test_empty_target_has_finite_zero_loss(self):
        geometry = torch.ones(2, 1, 3, 4, requires_grad=True)
        loss = loss_utils.scale_invariant_loss_func(
            geometry, torch.zeros_like(geometry), torch.zeros_like(geometry))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue((geometry.grad == 0).all())

    def test_decoder_losses_have_the_requested_resolution_weights(self):
        wrapper = self.make_wrapper()
        target = torch.ones(1, 1, 3, 4) * 2.
        geometry = [torch.ones_like(target) for _ in range(5)]
        scale = [torch.ones_like(target) for _ in range(5)]
        metric = [torch.ones_like(target) * (3. + r) for r in range(5)]
        outputs = [target] + geometry + scale + metric
        loss, info = wrapper.compute_loss_supervised(target, outputs, {})
        expected = sum(torch.log(torch.tensor(2. + r)) / (4 ** r) for r in range(5))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(float(info['loss_si']), 0.)

    def test_bottom_losses_contribute_with_one_over_256_weight(self):
        wrapper = self.make_wrapper()
        target = torch.ones(1, 1, 2, 2) * 2.
        geometry = [torch.ones_like(target) for _ in range(4)] + [
            torch.tensor([[[[0.5, 1.], [1., 1.5]]]], requires_grad=True)]
        metric = [target.clone() for _ in range(4)] + [
            torch.full_like(target, 5., requires_grad=True)]
        outputs = [target] + geometry + [torch.ones_like(target)] * 5 + metric
        loss, info = wrapper.compute_loss_supervised(target, outputs, {})
        expected_si = loss_utils.scale_invariant_loss_func(
            geometry[-1], target, torch.ones_like(target))
        expected_metric = torch.log(torch.tensor(4.))
        torch.testing.assert_close(info['loss_si_r4'], expected_si)
        torch.testing.assert_close(info['loss_log_l1_r4'], expected_metric)
        torch.testing.assert_close(loss, (expected_si + expected_metric) / 256.)
        loss.backward()
        self.assertGreater(float(geometry[-1].grad.abs().sum()), 0.)
        self.assertGreater(float(metric[-1].grad.abs().sum()), 0.)

    def test_all_auxiliary_outputs_follow_geometric_reversal(self):
        wrapper = self.make_wrapper()
        image, sparse = torch.rand(1, 3, 17, 25), torch.zeros(1, 1, 17, 25)
        outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        expected = [value.flip(-1).clone() for value in outputs]
        # Adapt only the test environment to this repository's older Pillow API.
        lanczos = getattr(Image, 'Resampling', Image).LANCZOS
        with mock.patch.object(Image, 'ANTIALIAS', lanczos, create=True):
            transforms = Transforms()
        reversed_outputs = transforms.reverse_transform(
            outputs, {'random_horizontal_flip': torch.tensor([True])})
        for original, aligned in zip(expected, reversed_outputs):
            torch.testing.assert_close(aligned, original)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
