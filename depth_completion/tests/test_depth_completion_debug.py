import copy
import json
import os
import sys
import tempfile
import unittest
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from depth_completion_debug import DepthCompletionDebug
from depth_completion_debug_analysis import anchor_distance, boundary_mask
from scale_propagation_model import ScalePropagationDepthModel


class DepthDebugTest(unittest.TestCase):
    def test_validation_rejection_diagnostics_use_actual_gumbel_mask(self):
        model = ScalePropagationDepthModel(n_channels=8).eval()
        with torch.no_grad():
            for stage in [model] + list(model.decoder):
                stage.rejection_map.conv.weight.zero_()
                stage.rejection_bias.zero_()
        sparse = torch.ones(1, 1, 17, 25) * 5.
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            debug.begin('validation', 1)
            with torch.no_grad():
                model(torch.rand(1, 3, 17, 25), sparse)
            for level, observation in debug.rejection_observations.items():
                torch.testing.assert_close(
                    observation['keep'], debug.anchor_context[level]['keep'].bool())
            keep = debug.rejection_observations['decoder.3']['keep']
            self.assertTrue(keep.any())
            self.assertTrue((~keep).any())
            debug.rejection_quality(sparse, sparse)
            report = debug.rejection_reports[(None, 'decoder.3')]
            self.assertEqual(report['decision_rule'], 'sampled_gumbel')
            self.assertEqual(report['confusion']['false_reject'], int((~keep).sum()))
            debug.close()

    def test_fusion_stage_maps_are_correct_with_and_without_sparse_features(self):
        for enabled in [False, True]:
            model = ScalePropagationDepthModel(n_channels=8, use_sparse_depth_features=enabled).eval()
            with tempfile.TemporaryDirectory() as directory:
                debug = self.make_debug(model, directory)
                debug.begin('validation', 1)
                with torch.no_grad():
                    model(torch.rand(1, 3, 17, 25), torch.ones(1, 1, 17, 25) * 5.)
                for entries in debug.trace.values():
                    stages = {r['stage']: r for r in entries}
                    if 'after_upsampling' not in stages:
                        continue
                    before, after = stages['before_upsampling'], stages['after_upsampling']
                    for name in ['geometry', 'scale']:
                        value = torch.from_numpy(before[name])[None, None]
                        expected = torch.nn.functional.interpolate(
                            value, size=after[name].shape, mode='bilinear', align_corners=True)
                        np.testing.assert_allclose(after[name], expected.numpy()[0, 0],
                                                   atol=1e-5, rtol=1e-5)
                debug.close()

    def test_distance_and_boundary_masks_ignore_missing_targets(self):
        mask = np.zeros((5, 7), dtype=bool)
        self.assertTrue(np.isinf(anchor_distance(mask)).all())
        mask[2, 3] = True
        self.assertEqual(anchor_distance(mask)[2, 5], 2.)
        target = np.ones((5, 7))
        target[:, :2] = 0
        self.assertFalse(boundary_mask(target, target > 0).any())
        target[:, 4:] = 3
        self.assertTrue(boundary_mask(target, target > 0).any())

    def test_stages_snapshots_probes_and_fixed_history(self):
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False).eval()
        image = torch.rand(1, 3, 17, 25)
        sparse = torch.zeros(1, 1, 17, 25)
        sparse[:, :, 7, 9] = 5.
        target = torch.ones_like(sparse) * 6.
        dataset = SimpleNamespace(image_paths=['fixed.png'])
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            rng = torch.get_rng_state().clone()
            for step in (1000, 2000):
                debug.start_validation(step)
                debug.begin('validation', step, sample=0)
                with torch.no_grad():
                    output = model(image, sparse)
                debug.validation_sample(0, image, sparse, target, output, 0., 100., dataset)
                debug.finish_validation()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(debug.trace)
            snapshot = np.load(os.path.join(directory, 'debug', 'fixed_0.npz'))
            self.assertIn('decoder.3__refinement_2__scale', snapshot.files)
            self.assertIn('bottom__after_propagation__outgoing_weights', snapshot.files)
            self.assertIn('bottom__selected_indices', snapshot.files)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'debug', 'worst_0_stages.npz')))
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            stages = [r for r in records if r['event'] == 'stage_accuracy']
            self.assertTrue(any(r['stage'] == 'before_upsampling' for r in stages))
            self.assertTrue(any(r['stage'] == 'after_geometry_update' for r in stages))
            self.assertTrue(any('accepted_anchor_distance_errors' in r for r in stages))
            self.assertTrue(any('multiplier' in r for r in stages))
            self.assertTrue(any(r['event'] == 'fixed_error_tracks' for r in records))
            self.assertTrue(any(r['event'] == 'scale_propagation_ablation' for r in records))
            self.assertEqual(len([r for r in stages if r['step'] == 1000]), 56)
            debug.close()

    def test_objective_probe_preserves_backward_and_existing_accumulated_gradients(self):
        from utils.src import loss_utils
        model = ScalePropagationDepthModel(n_channels=8, use_outlier_rejection=False)
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            debug.begin('train', 1)
            outputs = model(torch.rand(1, 3, 17, 25), torch.ones(1, 1, 17, 25) * 5, True)
            z = outputs['scaleless_depths'][0]
            metric = outputs['metric_depths_for_loss'][0]
            si = loss_utils.scale_invariant_loss_func(z, torch.ones_like(z) * 6, torch.ones_like(z))
            terms = dict(loss_si=si, loss_log_l1=torch.log1p((metric - 6).abs()).mean())
            parameters = list(model.parameters())
            for p in parameters:
                p.grad = torch.ones_like(p)
            state = torch.get_rng_state().clone()
            expected = torch.autograd.grad(sum(terms.values()), parameters, allow_unused=True, retain_graph=True)
            debug.gradient_objectives(terms)
            self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters))
            sum(terms.values()).backward()
            for p, g in zip(parameters, expected):
                torch.testing.assert_close(p.grad, torch.ones_like(p) if g is None else 1 + g)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            debug.close()

    def test_stage_accuracy_uses_common_target_and_reports_real_improvement(self):
        model = ScalePropagationDepthModel(n_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            debug.begin('validation', 1000, sample=0)
            z = torch.ones(1, 1, 2, 3)
            debug.capture_stage('bottom', 'initial_global_scale', z, z * 2)
            debug.capture_stage('bottom', 'refinement_1', z, z * 4, multiplier=z * 2)
            target = np.full((5, 7), 4., dtype=np.float32)
            debug.analyze_stages(np.zeros((1, 3, 5, 7)), np.zeros_like(target),
                                 target, target, np.ones_like(target, dtype=bool), {}, 'fixed_0')
            before, after = debug.stage_reports
            self.assertAlmostEqual(before['metric']['mae_mm'], 2000., places=4)
            self.assertAlmostEqual(before['aligned_geometry']['mae_mm'], 0., places=4)
            self.assertAlmostEqual(after['metric']['mae_mm'], 0., places=4)
            self.assertEqual(after['change']['improved_fraction'], 1.)
            self.assertAlmostEqual(after['change']['mean_absolute_error_change_m'], -2., places=6)
            debug.close()

    def make_debug(self, model, directory):
        wrapper = SimpleNamespace(
            min_predict_depth=1.5, max_predict_depth=100.,
            model=SimpleNamespace(model_depth=model))
        return DepthCompletionDebug(wrapper, directory, interval=1, max_samples=1)

    def test_observation_preserves_training_and_rng(self):
        torch.manual_seed(1)
        for rejection in (True, False):
            reference = ScalePropagationDepthModel(
                n_channels=8, min_predict_depth=1.5, max_predict_depth=100.,
                use_outlier_rejection=rejection)
            observed = copy.deepcopy(reference)
            image = torch.rand(1, 3, 17, 25)
            sparse = torch.zeros(1, 1, 17, 25)
            sparse[:, :, ::3, ::3] = 10.
            target = torch.ones_like(sparse) * 12.
            initial_rng = torch.get_rng_state()
            expected = reference(image, sparse)
            expected.mean().backward()
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(initial_rng)
            with tempfile.TemporaryDirectory() as directory:
                debug = self.make_debug(observed, directory)
                debug.begin('train', 1)
                debug.inputs('network_input', image, sparse, target)
                debug.sparse_levels(sparse)
                actual = observed(image, sparse)
                debug.training_prediction(actual, target, sparse, {'loss': actual.mean()})
                actual.mean().backward()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for a, b in zip(reference.parameters(), observed.parameters()):
                    if a.grad is None:
                        self.assertIsNone(b.grad)
                    else:
                        self.assertTrue(torch.equal(a.grad, b.grad))
                optimizer = torch.optim.Adam(observed.parameters(), lr=1e-4)
                debug.before_step(optimizer)
                optimizer.step()
                debug.after_step()
                debug.close()
                with open(debug.path) as stream:
                    records = [json.loads(line) for line in stream]
                self.assertEqual(sum(r['event'] == 'decoder' for r in records), 4)
                self.assertTrue(any(r['event'] == 'output_gradient' for r in records))
                gradients = next(r for r in records if r['event'] == 'gradients_before_optimizer')
                self.assertEqual(set(gradients['rejection_heads']),
                                 {'bottom', 'decoder.0', 'decoder.1', 'decoder.2', 'decoder.3'})
                if rejection:
                    self.assertEqual(sum(r['event'] == 'rejection' for r in records), 5)
                    self.assertTrue(any(g['grad_norm'] > 0.
                                        for g in gradients['rejection_heads'].values()))
                else:
                    self.assertFalse(any(r['event'] == 'rejection' for r in records))

    def test_inverse_failure_and_worst_sample_capture(self):
        model = ScalePropagationDepthModel(n_channels=8)
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            image = torch.zeros(1, 3, 16, 16)
            sparse = torch.zeros(1, 1, 16, 16)
            target = torch.ones_like(sparse) * 10.
            prediction = target.clone()
            prediction[0, 0, 2, 3] = 0.
            dataset = SimpleNamespace(image_paths=['example.png'])
            debug.start_validation(1000)
            debug.validation_sample(0, image, sparse, target, prediction, 0., 100., dataset)
            debug.finish_validation()
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            failure = next(r for r in records if r['event'] == 'inverse_failure')
            self.assertEqual(failure['coordinates_yx'], [[2, 3]])
            self.assertEqual(failure['prediction_values'], [0.])
            self.assertEqual(failure['native_imae'], 'inf')
            saved = np.load(os.path.join(directory, 'debug', 'inverse_failure_0.npz'))
            self.assertEqual(saved['prediction'][2, 3], 0.)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'debug', 'worst_0.npz')))

    def test_sparse_counts_and_rejection_ratios_use_actual_selection(self):
        model = ScalePropagationDepthModel(
            n_channels=8, use_outlier_rejection=False).eval()
        propagation_inputs = []
        handles = [stage.scale_propagation.register_forward_pre_hook(
            lambda module, inputs: propagation_inputs.append(inputs))
            for stage in [model] + list(model.decoder)]
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            sparse = torch.zeros(1, 1, 16, 16)
            # Nearest sampling misses this point at every coarse level.
            sparse[0, 0, 7, 7] = 5.
            debug.begin('validation', 1)
            debug.sparse_levels(sparse)
            with torch.no_grad():
                model(torch.rand(1, 3, 16, 16), sparse)
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            levels = [r for r in records if r['event'] == 'sparse_sampling']
            self.assertEqual([r['measurement_count'] for r in levels], [1] * 5)
            blocks = [r for r in records if r['event'] == 'decoder']
            self.assertEqual([r['keep_fraction_at_sparse'] for r in blocks], [1.] * 4)
            self.assertEqual([r['propagation_iterations'] for r in blocks], [4, 6, 8, 12])
            scales = [r for r in records if r['event'] == 'scale_propagation']
            self.assertEqual(len(scales), 5)
            self.assertTrue(all(not r['restores_anchors'] for r in scales))
            self.assertTrue(all(r['output_stage'] == 'before_refinement'
                                for r in scales))
            refinements = [r for r in records if r['event'] == 'scale_refinement']
            self.assertEqual(len(refinements), 10)
            self.assertEqual([r['refinement_iteration'] for r in refinements], [1, 2] * 5)
            for first, second in zip(refinements[::2], refinements[1::2]):
                self.assertEqual(first['level'], second['level'])
                self.assertEqual(first['scale_after_refinement'], second['scale_before_refinement'])
            self.assertTrue(all(not r['restores_anchors'] for r in refinements))
            self.assertTrue(all('scale_after_anchor_restoration' not in r
                                for r in refinements))
            self.assertTrue(all(r['propagation_activation'] == 'sigmoid' for r in scales))
            for record, inputs in zip(scales, propagation_inputs):
                weights = torch.sigmoid(inputs[4])
                accepted = inputs[3] > 0
                self.assertNotIn('surface_compatibility', record)
                self.assertIn('accepted_scale_residual', record)
                self.assertEqual(record['accepted_scale_residual_scope'], 'propagated_scale_before_refinement')
                self.assertIn('propagated_scale', record)
                self.assertIn('injected_scale', record)
                self.assertNotIn('anchor_support', record)
                self.assertEqual(record['sparse_scale_coordinates'],
                                 'current_resolution_grid_pixel')
                self.assertAlmostEqual(record['keep_fraction_at_sparse'],
                                       float(inputs[3].sum() / (inputs[2] > 0).sum().clamp_min(1)), places=6)
                self.assertAlmostEqual(record['sparse_scale_target']['mean'],
                                       float(inputs[5][accepted].double().mean()), places=6)
                self.assertAlmostEqual(record['outgoing_weight_sum']['mean'],
                                       float(weights.sum(dim=1).double().mean()), places=6)
            for record, inputs in zip(blocks, propagation_inputs[1:]):
                weights = torch.sigmoid(inputs[4])
                self.assertAlmostEqual(record['outgoing_weight_mean'],
                                       float(weights.mean()), places=6)
            self.assertTrue(all('scaleless_depth' in r and 'scale_map' in r for r in blocks))
        for handle in handles:
            handle.remove()

    def test_empty_ground_truth_does_not_crash_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(ScalePropagationDepthModel(n_channels=8), directory)
            image = torch.zeros(1, 3, 16, 16)
            depth = torch.zeros(1, 1, 16, 16)
            debug.start_validation(1)
            debug.validation_sample(0, image, depth, depth, depth, 0., 100., SimpleNamespace())
            debug.finish_validation()
            debug.close()

    def test_rejection_quality_uses_selected_original_measurement_coordinates(self):
        model = ScalePropagationDepthModel(n_channels=8, max_predict_depth=100.).eval()
        with torch.no_grad():
            for stage in [model] + list(model.decoder):
                stage.rejection_map.conv.weight.zero_()
                stage.rejection_bias.copy_(torch.tensor([100., -100.]).reshape(1, 2, 1, 1))
            model.decoder[0].rejection_bias.copy_(
                torch.tensor([-100., 100.]).reshape(1, 2, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            debug = self.make_debug(model, directory)
            sparse = torch.zeros(1, 1, 16, 16)
            target = torch.zeros_like(sparse)
            sparse[0, 0, 7, 7] = 20.
            sparse[0, 0, 7, 15] = 5.
            target[0, 0, 7, 7] = 5.
            target[0, 0, 7, 15] = 5.
            image = torch.rand(1, 3, 16, 16)
            debug.begin('validation', 1)
            with torch.no_grad():
                model(image, sparse)
            debug.inputs('validation_input', image, sparse, target)
            debug.close()
            with open(debug.path) as stream:
                records = [json.loads(line) for line in stream]
            inputs = {r['level']: r for r in records if r['event'] == 'rejection'}
            self.assertEqual(inputs['bottom']['measurement_count'], 1)
            for level in ['decoder.0', 'decoder.1', 'decoder.2', 'decoder.3']:
                self.assertEqual(inputs[level]['measurement_count'], 2)
            self.assertEqual(inputs['decoder.3']['measurement_x_offset']['max'], 0.)
            self.assertTrue(all(r['depth_context_location'] == 'current_resolution_grid_pixel'
                                for r in inputs.values()))
            quality = {r['level']: r for r in records if r['event'] == 'rejection_quality'}
            self.assertEqual(len(quality), 5)
            bottom = quality['bottom']['categories']
            self.assertEqual(bottom['all_with_ground_truth']['count'], 1)
            self.assertEqual(bottom['disagreement_over_3m']['count'], 1)
            self.assertEqual(bottom['disagreement_over_3m']['rejection_fraction'], 0.)
            coarse = quality['decoder.0']['categories']
            self.assertEqual(coarse['accurate_within_0_1m']['count'], 1)
            self.assertEqual(coarse['accurate_within_0_1m']['rejection_fraction'], 1.)
            self.assertEqual(coarse['disagreement_over_3m']['rejection_fraction'], 1.)
            finest = quality['decoder.3']['categories']
            self.assertEqual(finest['all_with_ground_truth']['count'], 2)
            self.assertEqual(finest['disagreement_over_3m']['rejection_fraction'], 0.)
            margins = [r for r in records if r['event'] == 'rejection']
            self.assertEqual(margins[0]['keep_minus_reject_at_sparse']['mean'], 200.)
            self.assertEqual(margins[1]['keep_minus_reject_at_sparse']['mean'], -200.)
            self.assertAlmostEqual(margins[0]['scaled_sparse_input']['max'], 0.2)

    def test_cli_accumulation_validation_and_disabled_equivalence(self):
        import matplotlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ['images', 'sparse', 'target']:
                folder = root / name
                folder.mkdir()
                path = folder / '0000000000.png'
                if name == 'images':
                    values = np.full((32, 48, 3), 128, dtype=np.uint8)
                else:
                    values = np.full((32, 48), 3072, dtype=np.uint16)
                    if name == 'sparse':
                        values[:] = 0
                        values[::4, ::4] = 2560
                Image.fromarray(values).save(path)
                paths[name] = path
            paths['intrinsics'] = root / 'intrinsics.npy'
            np.save(paths['intrinsics'], np.eye(3, dtype=np.float32))
            for name, path in paths.items():
                (root / (name + '.txt')).write_text((str(path) + '\n') * 2)
            arguments = [
                '--model_name', 'scale_propagation_kitti', '--supervision_type', 'supervised',
                '--seed', '17',
                '--w_losses', 'w_supervised=1.0',
                '--use_outlier_rejection', 'false', '--min_predict_depth', '1.5',
                '--max_predict_depth', '100', '--n_batch', '1', '--n_height', '32', '--n_width', '48',
                '--learning_rates', '0.0001', '--learning_schedule', '1',
                '--n_step_per_gradient_accumulation', '2', '--n_step_per_summary', '1',
                '--n_step_per_checkpoint', '1', '--start_step_validation', '1',
                '--min_evaluate_depth', '0', '--max_evaluate_depth', '100',
                '--debug_interval', '1', '--debug_max_samples', '1', '--device', 'cpu', '--n_thread', '0']
            for flag, name in [
                    ('train_images_path', 'images'), ('train_sparse_depth_path', 'sparse'),
                    ('train_ground_truth_path', 'target'), ('train_intrinsics_path', 'intrinsics'),
                    ('val_image_path', 'images'), ('val_sparse_depth_path', 'sparse'),
                    ('val_ground_truth_path', 'target'), ('val_intrinsics_path', 'intrinsics')]:
                arguments.extend(['--' + flag, str(root / (name + '.txt'))])
            script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_depth_completion.py')
            sys.path.insert(0, os.path.dirname(script))
            for enabled in ['false', 'true', 'rejection_false', 'rejection_true',
                            'l1_false', 'l1_true', 'none', 'none_reference', 'encoder_true']:
                torch.manual_seed(3)
                np.random.seed(3)
                debug_enabled = 'true' if enabled in ['none', 'none_reference'] else enabled.split('_')[-1]
                argv = [script] + arguments + ['--debug_depth', debug_enabled,
                                               '--checkpoint_path', str(root / enabled)]
                if enabled == 'encoder_true':
                    argv += ['--encoder_type', 'conv_attention', '--use_sparse_depth_features', 'true']
                if enabled in ['none', 'none_reference']:
                    argv += ['--debug_depth', 'true', '--use_outlier_rejection', 'true',
                             '--supervise_outlier_rejection', 'true',
                             '--rejection_loss_type', 'none' if enabled == 'none' else 'cross_entropy']
                    if enabled == 'none_reference':
                        argv += ['--supervise_outlier_rejection', 'false']
                if enabled.startswith('l1_'):
                    argv += ['--supervise_outlier_rejection', 'true', '--use_outlier_rejection', 'true',
                             '--rejection_loss_type', 'l1', '--w_rejection_l1', '0.03']
                if enabled.startswith('rejection_'):
                    argv += ['--supervise_outlier_rejection', 'true', '--use_outlier_rejection', 'true']
                    argv += ['--n_propagation_iterations', '1', '2', '3', '4', '6',
                             '--n_scale_refinement', '1']
                # The repository targets older Pillow; adapt only this test's environment.
                lanczos = getattr(Image, 'Resampling', Image).LANCZOS
                get_cmap = getattr(
                    matplotlib.cm, 'get_cmap', lambda name: matplotlib.colormaps[name])
                with mock.patch.object(Image, 'ANTIALIAS', lanczos, create=True), \
                        mock.patch.object(matplotlib.cm, 'get_cmap', get_cmap, create=True), \
                        mock.patch.object(sys, 'argv', argv):
                    runpy.run_path(script, run_name='__main__')
            records = [json.loads(line) for line in
                       (root / 'true/debug/diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['event'] == 'gradients_before_optimizer' for r in records), 1)
            self.assertEqual(sum(r['event'] == 'validation_sample' for r in records), 2)
            self.assertFalse((root / 'false/debug').exists())
            manifest = next(r for r in records if r['event'] == 'experiment_manifest')
            self.assertTrue(manifest['metric_geometry_detached'])
            self.assertEqual(manifest['fixed_sample_indices'], [0])
            self.assertTrue(Path(manifest['source_directory'], 'rng_start.pt').exists())
            self.assertTrue(any(r['event'] == 'gradient_objectives' for r in records))
            self.assertTrue(any(r['event'] == 'stage_accuracy' for r in records))
            self.assertTrue(any(r['images_seen_since_start'] == 2 for r in records))
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            for enabled in ['false', 'true']:
                for phase in ['train', 'val']:
                    events = EventAccumulator(str(root / enabled / ('events-' + phase))).Reload()
                    prefix = 'train' if phase == 'train' else 'eval'
                    self.assertIn(prefix + '/scaleless_depths/r0', events.Tags()['images'])
                    self.assertIn(prefix + '/scale_maps/r4', events.Tags()['images'])
                    self.assertIn(prefix + '/scaleless_comparison/r0', events.Tags()['images'])
                    self.assertIn(prefix + '/scaleless_comparison/r4', events.Tags()['images'])
                    if phase == 'train':
                        self.assertIn('train_loss_si_r3', events.Tags()['scalars'])
                        self.assertIn('train_loss_log_l1_r0', events.Tags()['scalars'])
                        self.assertIn('train_loss_si_r4', events.Tags()['scalars'])
                        self.assertIn('train_loss_log_l1_r4', events.Tags()['scalars'])
            checkpoints = []
            for enabled in ['false', 'true']:
                path = next((root / enabled).glob('checkpoints-1*'))
                if path.is_dir():
                    path = next(path.glob('scale-propagation-*.pth'))
                checkpoints.append(torch.load(str(path), map_location='cpu')['model_state_dict'])
            for name in checkpoints[0]:
                self.assertTrue(torch.equal(checkpoints[0][name], checkpoints[1][name]), name)
            supervised = [json.loads(line) for line in
                          (root / 'rejection_true/debug/diagnostics.jsonl').read_text().splitlines()]
            reports = [r for r in supervised if r['event'] == 'rejection_supervision']
            self.assertTrue(any(r['phase'] == 'train' and r['reject_target_count'] > 0 for r in reports))
            self.assertTrue(any(r['phase'] == 'validation' for r in reports))
            self.assertTrue(all(r['supervision_enabled'] for r in reports))
            config = next(r for r in supervised if r['event'] == 'configuration')
            self.assertEqual(config['settings']['rejection_supervision_levels'], list(range(5)))
            self.assertEqual(config['settings']['seed'], 17)
            self.assertEqual(config['settings']['data_order_seed'], 17)
            self.assertEqual(config['settings']['validation_loader_seed'], 18)
            manifest = next(r for r in supervised if r['event'] == 'experiment_manifest')
            self.assertEqual(manifest['torch_initial_seed'], 17)
            self.assertEqual(config['settings']['rejection_keep_weight'], 2.)
            self.assertEqual(config['settings']['rejection_resolution_weights'],
                             [1., .25, .0625, .015625, .00390625])
            self.assertTrue(all(r['resolution_weight'] > 0 for r in reports))
            self.assertEqual(config['propagation_iterations'], [1, 2, 3, 4, 6])
            self.assertEqual(config['settings']['n_scale_refinement'], 1)
            refinements = [r for r in supervised if r['event'] == 'scale_refinement']
            self.assertTrue(refinements)
            self.assertTrue(all(r['refinement_iteration'] == 1 for r in refinements))
            train = next(r for r in supervised if r['event'] == 'training_prediction')
            self.assertIn('loss_rejection_weighted', train['loss_info'])
            gradients = next(r for r in supervised if r['event'] == 'gradient_objectives')
            self.assertTrue(any(v['rejection_norm'] > 0 for v in gradients['groups'].values()))
            summary = json.loads((root / 'rejection_true/debug/validation_latest.json').read_text())
            self.assertEqual(len(summary['rejection_supervision']), 5)
            snapshot = np.load(root / 'rejection_true/debug/fixed_0.npz')
            self.assertIn('bottom__after_propagation__rejection_logits', snapshot.files)
            self.assertTrue((snapshot['bottom__after_propagation__rejection_labels'] == 1).any())
            extra = []
            for enabled in ['rejection_false', 'rejection_true']:
                path = next((root / enabled).glob('checkpoints-1*'))
                if path.is_dir():
                    path = next(path.glob('scale-propagation-*.pth'))
                extra.append(torch.load(str(path), map_location='cpu')['model_state_dict'])
            for name in extra[0]:
                self.assertTrue(torch.equal(extra[0][name], extra[1][name]), name)
            l1_records = [json.loads(line) for line in
                          (root / 'l1_true/debug/diagnostics.jsonl').read_text().splitlines()]
            l1_config = next(r for r in l1_records if r['event'] == 'configuration')['settings']
            self.assertEqual(l1_config['rejection_loss_type'], 'l1')
            self.assertTrue(l1_config['rejection_loss_active'])
            self.assertFalse(l1_config['rejection_supervision_active'])
            self.assertEqual(l1_config['rejection_supervision_levels'], list(range(5)))
            self.assertEqual(l1_config['encoder_input'], 'rgb_only')
            self.assertFalse(l1_config['decoder_fusion_uses_sparse_depth'])
            self.assertEqual(l1_config['sparse_scale_coordinates'], 'current_resolution_grid_pixel')
            self.assertEqual(l1_config['w_rejection_l1'], 0.03)
            self.assertEqual(l1_config['rejection_loss_weight'], 0.03)
            self.assertEqual(l1_config['l1_regularized_mask'], 'keep')
            l1 = next(r for r in l1_records if r['event'] == 'rejection_sparsity')
            self.assertEqual(len(l1['fractions']), 5)
            self.assertTrue(all(k.startswith('rejection_kept_fraction_') for k in l1['fractions']))
            self.assertEqual(l1['regularized_mask'], 'keep')
            self.assertEqual(l1['loss_weight'], 0.03)
            self.assertTrue(0. <= l1['loss'] <= sum(4. ** -r for r in range(5)))
            training = next(r for r in l1_records if r['event'] == 'training_prediction')['loss_info']
            self.assertAlmostEqual(training['loss_rejection_weighted'], 0.03 * training['loss_rejection'], places=6)
            for r in range(5):
                self.assertIn('rejection_kept_fraction_r{}'.format(r), training)
            objectives = next(r for r in l1_records if r['event'] == 'gradient_objectives')
            self.assertTrue(any(v['rejection_norm'] > 0. for v in objectives['groups'].values()))
            none_records = [json.loads(line) for line in
                            (root / 'none/debug/diagnostics.jsonl').read_text().splitlines()]
            none_config = next(r for r in none_records if r['event'] == 'configuration')['settings']
            self.assertFalse(none_config['rejection_loss_active'])
            none_loss = next(r for r in none_records if r['event'] == 'training_prediction')['loss_info']
            self.assertNotIn('loss_rejection', none_loss)
            for pair in [('l1_false', 'l1_true'), ('none', 'none_reference')]:
                weights = []
                for name in pair:
                    path = next((root / name).glob('checkpoints-1*'))
                    if path.is_dir():
                        path = next(path.glob('scale-propagation-*.pth'))
                    weights.append(torch.load(str(path), map_location='cpu')['model_state_dict'])
                for name in weights[0]:
                    self.assertTrue(torch.equal(weights[0][name], weights[1][name]), (pair, name))
            encoder_records = [json.loads(line) for line in
                               (root / 'encoder_true/debug/diagnostics.jsonl').read_text().splitlines()]
            encoder_config = next(r for r in encoder_records if r['event'] == 'configuration')
            self.assertEqual(encoder_config['settings']['encoder_type'], 'conv_attention')
            self.assertTrue(encoder_config['settings']['use_sparse_depth_features'])
            self.assertTrue(encoder_config['settings']['decoder_fusion_uses_sparse_depth'])
            self.assertEqual(encoder_config['settings']['encoder_input'], 'rgb_sparse')
            path = next((root / 'encoder_true').glob('checkpoints-1*'))
            if path.is_dir():
                path = next(path.glob('scale-propagation-*.pth'))
            weights = torch.load(str(path), map_location='cpu')['model_state_dict']
            self.assertIn('encoder.level1.convolutions.5.conv.weight', weights)
            self.assertIn('encoder.level3_attention.1.attention.in_proj_weight', weights)
            self.assertEqual(weights['encoder.level0.convolutions.0.conv.weight'].shape[1], 4)
            self.assertEqual(weights['decoder.0.fusion.convolutions.0.conv.weight'].shape[1], 67)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
