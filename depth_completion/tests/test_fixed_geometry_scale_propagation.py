import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as functional
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Match CLI import order: load this repository's datasets before the shared
# KBNet helpers add their own source directory (which also has datasets.py).
import depth_completion as depth_completion_pipeline
from fixed_geometry_scale_propagation_model import (
    FixedGeometryScalePropagationDepthModel,
    RGBGeometryModel,
    global_scale_from_sparse)
from fixed_geometry_scale_propagation_depth_completion_model import \
    FixedGeometryScalePropagationDepthCompletionModel
from depth_completion_model import DepthCompletionModel
from depth_completion_debug import DepthCompletionDebug
from utils.src import loss_utils


class FixedGeometryScalePropagationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.image = torch.rand(2, 3, 17, 25)
        self.sparse = torch.zeros(2, 1, 17, 25)
        self.sparse[:, :, 4, 7] = 2.0
        self.sparse[:, :, 12, 20] = 6.0
        self.target = 1.0 + torch.rand_like(self.sparse) * 6.0

    def wrapper(self, stage='scale', **kwargs):
        return FixedGeometryScalePropagationDepthCompletionModel(
            min_predict_depth=0.1, max_predict_depth=100.0,
            device=torch.device('cpu'), training_stage=stage,
            n_propagation_iterations=(1, 2, 3, 4, 6), **kwargs)

    def test_geometry_is_rgb_only_and_all_levels_share_one_reference(self):
        wrapper = self.wrapper()
        wrapper.train()
        first = wrapper.model_depth(self.image, self.sparse, return_all_outputs=True)
        second = wrapper.model_depth(self.image, self.sparse * 5.0, return_all_outputs=True)
        geometry = first['scaleless_depths'][0]
        self.assertEqual(geometry.shape, self.sparse.shape)
        torch.testing.assert_close(geometry.mean(dim=[2, 3]), torch.ones(2, 1))
        for value, other in zip(first['scaleless_depths'], second['scaleless_depths']):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
            expected = functional.interpolate(
                geometry, size=value.shape[-2:], mode='bilinear', align_corners=True)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            self.assertFalse(value.requires_grad)
        self.assertEqual([value.shape[-2:] for value in first['scale_maps']],
                         [(17, 25), (9, 13), (5, 7), (3, 4), (2, 2)])

    def test_stage_switch_freezes_actual_network_not_only_output(self):
        for stage in ['geometry', 'scale']:
            wrapper = self.wrapper(stage)
            wrapper.train()
            model = wrapper.model_depth
            self.assertEqual(model.geometry.training, stage == 'geometry')
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.requires_grad,
                                 name.startswith('geometry.') == (stage == 'geometry'))
            optimizer_ids = {id(value) for value in wrapper.parameters_depth()}
            self.assertEqual(optimizer_ids, {id(value) for value in model.parameters()
                                             if value.requires_grad})
            wrapper.eval()
            self.assertFalse(any(module.training for module in model.modules()))

    def test_joint_default_trains_both_networks_in_one_step_without_checkpoint(self):
        wrapper = FixedGeometryScalePropagationDepthCompletionModel(
            device=torch.device('cpu'), min_predict_depth=0.1, max_predict_depth=100.0,
            n_propagation_iterations=(1, 1, 1, 1, 1))
        self.assertEqual(wrapper.training_stage, 'joint')
        model = wrapper.model_depth
        self.assertTrue(model.geometry.training)
        self.assertTrue(all(value.requires_grad for value in model.parameters()))
        geometry_before = model.geometry.depth_output[-1].conv.weight.detach().clone()
        propagation_before = model.propagation_map.conv.weight.detach().clone()
        optimizer = torch.optim.Adam(wrapper.parameters_depth(), lr=1e-3)
        outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
        loss, info = wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_supervised=2.0, w_si=3.0, w_log_l1=0.5))
        torch.testing.assert_close(loss, 6.0 * info['loss_si'] + info['loss_log_l1'])
        loss.backward()
        self.assertGreater(float(model.geometry.depth_output[-1].conv.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.propagation_map.conv.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(model.geometry.depth_output[-1].conv.weight, geometry_before))
        self.assertFalse(torch.equal(model.propagation_map.conv.weight, propagation_before))

    def test_joint_detach_flag_controls_metric_gradients_through_all_geometry_routes(self):
        for detach in [True, False]:
            wrapper = self.wrapper('joint', detach_scaleless_depth=detach)
            outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
            _, info = wrapper.compute_loss_supervised(self.target, outputs, {})
            gradients = torch.autograd.grad(
                info['loss_log_l1_weighted'], list(wrapper.model_depth.geometry.parameters()),
                allow_unused=True, retain_graph=True)
            if detach:
                self.assertTrue(all(value is None for value in gradients))
            else:
                self.assertTrue(any(value is not None and value.abs().sum() > 0.0 for value in gradients))
            info['loss_si_weighted'].backward()
            self.assertGreater(float(wrapper.model_depth.geometry.depth_output[-1].conv.weight.grad.abs().sum()), 0.0)

    def test_geometry_stage_trains_resolution_weighted_auxiliary_si_heads(self):
        wrapper = self.wrapper('geometry')
        outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
        loss, info = wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_supervised=2.0, w_si=3.0, w_log_l1=1000.0))
        expected = sum(
            loss_utils.scale_invariant_loss_func(value, self.target, torch.ones_like(self.target)) / (4 ** r)
            for r, value in enumerate([outputs[1]] + outputs[16:20]))
        torch.testing.assert_close(loss, 6.0 * expected)
        torch.testing.assert_close(info['loss_si'], sum(
            info['loss_si_r{}'.format(r)] / (4 ** r) for r in range(5)))
        self.assertEqual(float(info['loss_log_l1_weighted']), 0.0)
        loss.backward()
        self.assertTrue(any(value.grad is not None and value.grad.abs().sum() > 0
                            for value in wrapper.model_depth.geometry.parameters()))
        self.assertTrue(all(value.grad is None for name, value in wrapper.model_depth.named_parameters()
                            if not name.startswith('geometry.')))
        for head in wrapper.model_depth.geometry.auxiliary_depth_outputs:
            self.assertIsNotNone(head.conv.weight.grad)
            self.assertGreater(float(head.conv.weight.grad.abs().sum()), 0.0)

    def test_auxiliary_predictions_never_replace_propagation_geometry(self):
        wrapper = self.wrapper('joint')
        model = wrapper.model_depth
        first = model(self.image, self.sparse, return_all_outputs=True)
        native_auxiliary = first['auxiliary_scaleless_depths']
        self.assertEqual([value.shape[-2:] for value in native_auxiliary],
                         [(17, 25), (9, 13), (5, 7), (3, 4), (2, 2)])
        self.assertIs(native_auxiliary[0], first['scaleless_depths'][0])
        with torch.no_grad():
            for head in model.geometry.auxiliary_depth_outputs:
                head.conv.weight.add_(torch.randn_like(head.conv.weight))
        second = model(self.image, self.sparse, return_all_outputs=True)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            native_auxiliary[1:], second['auxiliary_scaleless_depths'][1:])))
        for key in ['scaleless_depths', 'scale_maps', 'metric_depths_for_loss']:
            for a, b in zip(first[key], second[key]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(first['depth'], second['depth'], rtol=0, atol=0)

    def test_metric_loss_does_not_train_auxiliary_heads_even_without_detach(self):
        wrapper = self.wrapper('joint', detach_scaleless_depth=False)
        outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
        _, info = wrapper.compute_loss_supervised(self.target, outputs, {})
        heads = list(wrapper.model_depth.geometry.auxiliary_depth_outputs.parameters())
        gradients = torch.autograd.grad(info['loss_log_l1_weighted'], heads, allow_unused=True)
        self.assertTrue(all(value is None for value in gradients))

    def test_inference_does_not_evaluate_auxiliary_heads(self):
        model = self.wrapper().model_depth
        calls = []
        handles = [head.register_forward_hook(lambda *args: calls.append(1))
                   for head in model.geometry.auxiliary_depth_outputs]
        ordinary = model(self.image, self.sparse)
        self.assertEqual(calls, [])
        detailed = model(self.image, self.sparse, return_all_outputs=True)
        self.assertEqual(len(calls), 4)
        torch.testing.assert_close(ordinary, detailed['depth'], rtol=0, atol=0)
        for handle in handles:
            handle.remove()

    def test_scale_training_updates_propagation_but_cannot_change_geometry(self):
        wrapper = self.wrapper()
        model = wrapper.model_depth
        before = {name: value.clone() for name, value in model.geometry.state_dict().items()}
        geometry_before = model.geometry(self.image)[1][0].detach().clone()
        optimizer = torch.optim.Adam(wrapper.parameters_depth(), lr=1e-3, weight_decay=0.01)
        outputs = wrapper.forward_depth(self.image, self.sparse, return_all_outputs=True)
        self.assertEqual(len(outputs), 20)
        loss, info = wrapper.compute_loss_supervised(
            self.target, outputs, dict(w_supervised=2.0, w_si=1000.0, w_log_l1=3.0))
        expected = sum(info['loss_log_l1_r{}'.format(r)] / (4 ** r) for r in range(5))
        torch.testing.assert_close(loss, 6.0 * expected)
        self.assertEqual(float(info['loss_si_weighted']), 0.0)
        loss.backward()
        self.assertTrue(all(value.grad is None for value in model.geometry.parameters()))
        self.assertGreater(float(model.propagation_map.conv.weight.grad.abs().sum()), 0.0)
        for block in model.decoder:
            self.assertGreater(float(block.propagation_map.conv.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        for name, value in model.geometry.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(model.geometry(self.image)[1][0], geometry_before, rtol=0, atol=0)

    def test_global_mode_bypasses_all_learned_propagation(self):
        wrapper = self.wrapper()
        model = wrapper.model_depth
        model.propagation_mode = 'global'
        calls = []
        handles = [module.register_forward_hook(lambda *args: calls.append(1))
                   for name, module in model.named_modules() if name.endswith('scale_propagation')]
        output = model(self.image, self.sparse, return_all_outputs=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [])
        geometry = output['scaleless_depths'][0]
        scale = global_scale_from_sparse(self.sparse, geometry)
        for value in output['scale_maps']:
            torch.testing.assert_close(value, scale.expand_as(value))
        torch.testing.assert_close(output['depth'], (geometry * scale).clamp(0.1, 100.0))

    def test_no_measurements_keep_neutral_scale_and_finite_depth(self):
        model = self.wrapper().model_depth
        output = model(self.image, torch.zeros_like(self.sparse), return_all_outputs=True)
        for scale in output['scale_maps']:
            torch.testing.assert_close(scale, torch.ones_like(scale))
        self.assertTrue(torch.isfinite(output['depth']).all())
        self.assertTrue((output['depth'] >= 0.1).all())

    def test_encoder_options_and_even_sizes(self):
        for encoder_type in ['conv', 'conv_attention']:
            model = RGBGeometryModel(n_channels=8, encoder_type=encoder_type)
            features, geometries = model(torch.rand(1, 3, 16, 32))
            self.assertEqual(len(features), 5)
            self.assertEqual([value.shape[-2:] for value in geometries],
                             [(16, 32), (8, 16), (4, 8), (2, 4), (1, 2)])
            self.assertTrue(all((value > 0.0).all() for value in geometries))

    def test_geometry_checkpoint_initializes_only_geometry_then_full_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry_path = os.path.join(directory, 'geometry.pth')
            first = self.wrapper('geometry')
            first.save_model(geometry_path, step=123)
            second = self.wrapper(geometry_restore_path=geometry_path)
            for a, b in zip(first.model_depth.geometry.parameters(), second.model_depth.geometry.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(torch.equal(first.model_depth.propagation_map.conv.weight,
                                         second.model_depth.propagation_map.conv.weight))
            optimizer = torch.optim.Adam(second.parameters_depth(), lr=1e-3)
            outputs = second.forward_depth(self.image, self.sparse, return_all_outputs=True)
            second.compute_loss_supervised(self.target, outputs, {})[0].backward()
            optimizer.step()
            scale_path = os.path.join(directory, 'scale.pth')
            second.save_model(scale_path, 7, optimizer)
            third = self.wrapper()
            resumed_optimizer = torch.optim.Adam(third.parameters_depth())
            step, _ = third.restore_model(scale_path, resumed_optimizer)
            self.assertEqual(step, 7)
            self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))
            torch.testing.assert_close(
                second.forward_depth(self.image, self.sparse),
                third.forward_depth(self.image, self.sparse), rtol=0, atol=0)

    def test_main_wrapper_routing_saving_and_scale_logging_layout(self):
        wrapper = DepthCompletionModel(
            'fixed_geometry_scale_propagation_kitti', [], 0.1, 100.0,
            device=torch.device('cpu'), fixed_geometry_stage='geometry')
        self.assertIsInstance(wrapper.model, FixedGeometryScalePropagationDepthCompletionModel)
        output = wrapper.forward_depth(self.image, self.sparse, None, return_all_outputs=True)
        maps = wrapper.scale_outputs(output)
        self.assertEqual(len(output), 20)
        self.assertEqual(len(maps['scaleless_depths']), 5)
        self.assertEqual(len(maps['scale_maps']), 5)
        self.assertTrue(all(value.shape == self.sparse.shape for value in output))
        with tempfile.TemporaryDirectory() as directory:
            wrapper.save_model(directory, 5)
            self.assertTrue(os.path.isfile(os.path.join(
                directory, 'fixed-geometry-scale-propagation-5.pth')))

    def test_existing_model_route_is_unchanged(self):
        wrapper = DepthCompletionModel(
            'scale_propagation_void', [], 0.1, 8.0, device=torch.device('cpu'))
        self.assertEqual(type(wrapper.model).__name__, 'ScalePropagationDepthCompletionModel')
        self.assertFalse(hasattr(wrapper.model.model_depth, 'fixed_geometry'))

    def test_debug_hooks_capture_scale_stages_without_geometry_rewrites(self):
        wrapper = DepthCompletionModel(
            'fixed_geometry_scale_propagation_kitti', [], 0.1, 100.0,
            device=torch.device('cpu'), n_propagation_iterations=(1, 1, 1, 1, 1))
        wrapper.eval()
        with tempfile.TemporaryDirectory() as directory:
            debug = DepthCompletionDebug(wrapper, directory, interval=1, max_samples=1)
            debug.begin('validation', 1, active=True)
            with torch.no_grad():
                wrapper.forward_depth(self.image[:1], self.sparse[:1], None, return_all_outputs=True)
            self.assertEqual(len(debug.trace), 5)
            self.assertIn('fixed_geometry', [value['stage'] for value in debug.trace['decoder.3']])
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            propagation = [value for value in records if value['event'] == 'scale_propagation']
            self.assertEqual(len(propagation), 5)
            self.assertTrue(all(not value['restores_anchors'] for value in propagation))
            for handle in debug.handles:
                handle.remove()

    def test_training_cli_passes_independent_stage_and_geometry_path(self):
        script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_depth_completion.py')
        argv = [script, '--train_images_path', 'images.txt', '--train_sparse_depth_path', 'sparse.txt',
                '--checkpoint_path', 'outputs', '--device', 'cpu',
                '--model_name', 'fixed_geometry_scale_propagation_kitti',
                '--fixed_geometry_stage', 'scale', '--geometry_restore_path', 'geometry.pth']
        with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.train') as train:
            runpy.run_path(script, run_name='__main__')
        self.assertEqual(train.call_args.kwargs['fixed_geometry_stage'], 'scale')
        self.assertEqual(train.call_args.kwargs['geometry_restore_path'], 'geometry.pth')

    def test_run_cli_supports_identical_checkpoint_global_control(self):
        script = os.path.join(os.path.dirname(__file__), '..', 'src', 'run_depth_completion.py')
        argv = [script, '--image_path', 'images.txt', '--sparse_depth_path', 'sparse.txt',
                '--intrinsics_path', 'intrinsics.txt', '--output_path', 'outputs', '--device', 'cpu',
                '--fixed_geometry_mode', 'global', '--n_propagation_iterations', '1', '2', '3', '4', '6']
        with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.run') as run:
            runpy.run_path(script, run_name='__main__')
        self.assertEqual(run.call_args.kwargs['fixed_geometry_mode'], 'global')
        self.assertEqual(run.call_args.kwargs['n_propagation_iterations'], [1, 2, 3, 4, 6])

    def test_both_stages_through_actual_training_loop_and_tensorboard(self):
        from utils.src import data_utils, log_utils
        data_loader = torch.utils.data.DataLoader
        def local_loader(*args, **kwargs):
            # The desktop sandbox does not permit Torch's shared-memory worker
            # manager. Exercise the real datasets/loops with in-process loading.
            kwargs['num_workers'] = 0
            return data_loader(*args, **kwargs)
        script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_depth_completion.py')
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, 'image.png')
            sparse_path = os.path.join(directory, 'sparse.png')
            target_path = os.path.join(directory, 'target.png')
            intrinsics_path = os.path.join(directory, 'intrinsics.npy')
            Image.fromarray((self.image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(image_path)
            data_utils.save_depth(self.sparse[0, 0].numpy(), sparse_path)
            data_utils.save_depth(self.target[0, 0].numpy(), target_path)
            np.save(intrinsics_path, np.eye(3, dtype=np.float32))
            lists = []
            for index, path in enumerate([image_path, sparse_path, target_path, intrinsics_path]):
                list_path = os.path.join(directory, '{}.txt'.format(index))
                with open(list_path, 'w') as stream:
                    stream.write(path + '\n')
                lists.append(list_path)
            geometry_checkpoint = None
            for stage in ['geometry', 'scale', 'joint']:
                destination = os.path.join(directory, stage)
                argv = [script, '--train_images_path', lists[0], '--train_sparse_depth_path', lists[1],
                        '--train_ground_truth_path', lists[2], '--train_intrinsics_path', lists[3],
                        '--val_image_path', lists[0], '--val_sparse_depth_path', lists[1],
                        '--val_ground_truth_path', lists[2], '--val_intrinsics_path', lists[3],
                        '--model_name', 'fixed_geometry_scale_propagation_void',
                        '--fixed_geometry_stage', stage, '--checkpoint_path', destination,
                        '--device', 'cpu', '--n_thread', '0', '--n_batch', '1',
                        '--n_height', '17', '--n_width', '25', '--supervision_type', 'supervised',
                        '--w_losses', 'w_supervised=1.0', 'w_si=1.0', 'w_log_l1=1.0',
                        '--learning_schedule', '1', '--learning_rates', '0.0001',
                        '--n_step_per_checkpoint', '1', '--n_step_per_summary', '1',
                        '--start_step_validation', '1',
                        '--n_image_per_summary', '1', '--debug_depth', 'true', '--debug_interval', '1',
                        '--n_propagation_iterations', '1', '1', '1', '1', '1']
                if stage == 'scale':
                    argv += ['--geometry_restore_path', geometry_checkpoint]
                with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.train') as capture:
                    runpy.run_path(script, run_name='__main__')
                # The local test runtime has Pillow >=10; the training server's
                # older transforms use its former ANTIALIAS spelling.
                with contextlib.redirect_stdout(io.StringIO()), \
                        mock.patch.object(Image, 'ANTIALIAS', getattr(Image, 'Resampling', Image).LANCZOS, create=True), \
                        mock.patch.object(log_utils.plt.cm, 'get_cmap', log_utils.plt.get_cmap, create=True), \
                        mock.patch('torch.utils.data.DataLoader', side_effect=local_loader):
                    depth_completion_pipeline.train(**capture.call_args.kwargs)
                checkpoint_path = os.path.join(destination, 'checkpoints-1',
                                               'fixed-geometry-scale-propagation-1.pth')
                self.assertTrue(os.path.isfile(checkpoint_path))
                self.assertTrue(os.path.isdir(os.path.join(destination, 'events-train')))
                with open(os.path.join(destination, 'debug', 'diagnostics.jsonl')) as stream:
                    records = [json.loads(line) for line in stream]
                self.assertTrue(any(value['event'] == 'training_prediction' for value in records))
                manifest = next(value for value in records if value['event'] == 'experiment_manifest')
                self.assertTrue(any('fixed_geometry_scale_propagation_depth_completion_model.py' in path
                                    for path in manifest['source_files']))
                if stage == 'geometry':
                    geometry_checkpoint = checkpoint_path
                elif stage == 'scale':
                    first = torch.load(geometry_checkpoint, map_location='cpu')['geometry_state_dict']
                    second = torch.load(checkpoint_path, map_location='cpu')['geometry_state_dict']
                    for name in first:
                        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            # Exercise standalone inference as well as the training validator.
            # Saved depth must retain the original odd image dimensions.
            run_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'run_depth_completion.py')
            for mode in ['hierarchical', 'global']:
                destination = os.path.join(directory, 'evaluation_' + mode)
                argv = [run_script, '--image_path', lists[0], '--sparse_depth_path', lists[1],
                        '--ground_truth_path', lists[2], '--intrinsics_path', lists[3],
                        '--model_name', 'fixed_geometry_scale_propagation_void',
                        '--fixed_geometry_mode', mode, '--restore_paths', checkpoint_path,
                        '--output_path', destination, '--device', 'cpu', '--save_outputs',
                        '--n_propagation_iterations', '1', '1', '1', '1', '1']
                with mock.patch.object(sys, 'argv', argv), mock.patch('depth_completion.run') as capture:
                    runpy.run_path(run_script, run_name='__main__')
                with contextlib.redirect_stdout(io.StringIO()), \
                        mock.patch.object(Image, 'ANTIALIAS', getattr(Image, 'Resampling', Image).LANCZOS, create=True), \
                        mock.patch('torch.utils.data.DataLoader', side_effect=local_loader):
                    depth_completion_pipeline.run(**capture.call_args.kwargs)
                saved = data_utils.load_depth(os.path.join(
                    destination, 'outputs', 'output_depth', '0000000000.png'))
                self.assertEqual(saved.shape, (17, 25))
                self.assertTrue(np.isfinite(saved).all())
                with open(os.path.join(destination, 'results.txt')) as stream:
                    self.assertIn('fixed_geometry_mode=' + mode, stream.read())


if __name__ == '__main__':
    unittest.main()
