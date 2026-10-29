import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json
import zipfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import diagnose_scale_propagation as diagnostic
import diagnose_hierarchical_rejection as hierarchy
import diagnose_anchor_retention as anchors


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.wrapper = diagnostic.FixedGeometryScalePropagationDepthCompletionModel(
            device=torch.device('cpu'), encoder_type='conv', min_predict_depth=0.1,
            max_predict_depth=8, use_outlier_rejection=True,
            n_propagation_iterations=(1, 1, 1, 1, 1), n_scale_refinement=1)
        self.wrapper.eval()
        self.image = torch.rand(1, 3, 17, 25)
        self.target = torch.ones(1, 1, 17, 25) * 2
        self.target[:, :, :, 12:] = 5
        self.sparse = torch.zeros_like(self.target)
        self.sparse[:, :, 4, 5] = 2
        self.sparse[:, :, 10, 20] = 1  # Deliberately incorrect anchor.
        self.args = diagnostic.parser().parse_args([
            '--checkpoint', 'unused', '--output', 'unused', '--images', 'unused',
            '--sparse_depths', 'unused', '--ground_truths', 'unused',
            '--points_per_group', '1', '--max_evaluate_depth', '8'])

    def run_forward(self, **kwargs):
        return diagnostic.forward(self.wrapper, self.image, self.sparse,
                                  self.target, self.args, 42, **kwargs)

    def test_replay_gradients_and_patch_cleanup(self):
        native = diagnostic.original.rejection_keep_mask
        predicted, base = self.run_forward(gradients=True)
        diagnostic.seed_everything(42)
        with torch.no_grad():
            untouched = self.wrapper.forward_depth(self.image, self.sparse)
        torch.testing.assert_close(predicted, untouched, rtol=0, atol=0)
        replay, trial = self.run_forward(frozen=base.frozen)
        torch.testing.assert_close(predicted, replay, rtol=0, atol=0)
        self.assertIs(diagnostic.original.rejection_keep_mask, native)
        self.assertEqual(len(base.gradients), 5)
        self.assertTrue(all(np.isfinite(g).all() for pair in base.gradients.values() for g in pair))
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in self.wrapper.model_depth.modules()))
        for r in range(5):
            torch.testing.assert_close(base.frozen[r], trial.frozen[r])

    def test_original_model_supported(self):
        self.wrapper = diagnostic.ScalePropagationDepthCompletionModel(
            device=torch.device('cpu'), encoder_type='conv', min_predict_depth=0.1,
            max_predict_depth=8, use_outlier_rejection=True,
            n_propagation_iterations=(1,)*5, n_scale_refinement=1)
        self.wrapper.eval()
        prediction, base = self.run_forward(gradients=True)
        replay, _ = self.run_forward(frozen=base.frozen)
        torch.testing.assert_close(prediction, replay, rtol=0, atol=0)
        self.assertEqual(len(base.contexts), 5)

    def test_point_intervention_changes_only_one_gate(self):
        _, base = self.run_forward()
        index = 4 * 25 + 5
        _, trial = self.run_forward(frozen=base.frozen, intervention={
            'kind': 'point', 'level': 0, 'index': index,
            'keep': 1 - float(base.frozen[0].flatten()[index])})
        for r in range(1, 5):
            torch.testing.assert_close(base.frozen[r], trial.frozen[r])
        self.assertEqual(int((base.frozen[0] != trial.frozen[0]).sum()), 1)

    def test_no_mixing_keeps_injection_and_oracle_reject_keeps_good(self):
        _, base = self.run_forward()
        _, trial = self.run_forward(frozen=base.frozen, intervention={'kind': 'no_mixing', 'level': 0})
        context = trial.contexts[0]
        expected = context['keep'] * context['ratio'] + (1 - context['keep']) * context['scale']
        stage = next(value for r, name, value in trial.stages if r == 0 and name == 'after_propagation')
        np.testing.assert_allclose(stage, context['geometry'] * expected)
        _, reject = self.run_forward(frozen=base.frozen, intervention={'kind': 'oracle_reject', 'level': 0})
        self.assertEqual(float(reject.frozen[0][0, 0, 10, 20]), 0)
        self.assertEqual(float(reject.frozen[0][0, 0, 4, 5]), float(base.frozen[0][0, 0, 4, 5]))

    def test_oracle_fold_direction_and_unknown_edges(self):
        scale = torch.tensor([[[[1., 9.]]]])
        logits = torch.full((1, 9, 1, 2), -100.)
        logits[0, 5, 0, 0] = 100.  # Left source sends right, not the reverse.
        keep = torch.zeros_like(scale)
        unknown = torch.zeros_like(scale)
        actual = diagnostic.oracle_propagation(scale, scale, keep, logits, 1, unknown)
        torch.testing.assert_close(actual, torch.tensor([[[[1., 5.]]]]))
        separated = diagnostic.oracle_propagation(scale, scale, keep, logits, 1, scale)
        torch.testing.assert_close(separated, scale)

    def test_complete_sample_report(self):
        records = []
        before = {key: value.clone() for key, value in self.wrapper.model_depth.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            diagnostic.diagnose_sample(self.wrapper, self.image, self.sparse, self.target,
                self.args, 0, Path(directory), lambda kind, **values: records.append(dict(type=kind, **values)))
            diagnostic.write_summary(Path(directory), records, 1)
            self.assertTrue((Path(directory) / 'sample_0_maps.npz').exists())
            self.assertTrue((Path(directory) / 'report.md').exists())
        self.assertEqual(sum(row['type'] == 'stage_intervention' for row in records), 24)
        self.assertTrue(any(row['type'] == 'point_counterfactual' for row in records))
        self.assertTrue(any(row['type'] == 'worst_pixel_neighbors' for row in records))
        for key, value in self.wrapper.model_depth.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_hierarchy_oracle_restores_good_and_preserves_r0(self):
        # Good, bad, unknown available, missing; all initially rejected.
        sparse = torch.tensor([[[[2., 9., 3., 0.]]]])
        labels = torch.tensor([[[[0, 1, -1, -1]]]])
        selected = {r: sparse.clone() for r in range(5)}
        classified = {r: labels.clone() for r in range(5)}
        learned = {r: torch.zeros_like(sparse) for r in range(5)}
        masks = hierarchy.policy_masks(learned, selected, classified)
        for name in hierarchy.POLICIES:
            if name != 'A_keep_all':
                torch.testing.assert_close(masks[name][0], learned[0])
        for r in range(1, 5):
            torch.testing.assert_close(masks['E_oracle_hierarchy'][r], torch.tensor([[[[1., 0., 1., 0.]]]]))
            for level in range(1, 5):
                expected = torch.tensor([[[[1., 0., 1., 0.]]]]) if level == r else (sparse > 0).float()
                torch.testing.assert_close(masks['D_oracle_r{}'.format(r)][level], expected)
        self.assertEqual(float(learned[1].sum()), 0)  # Policy construction never mutates baseline.

    def test_hierarchy_paired_run_and_summary(self):
        self.args.seeds = [42, 43]
        rows = []
        before = {k: v.clone() for k, v in self.wrapper.model_depth.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            hierarchy.diagnose_hierarchy_sample(self.wrapper, self.image, self.sparse, self.target,
                self.args, 0, directory, lambda kind, **values: rows.append(dict(type=kind, **values)), save_maps=True)
            hierarchy.write_hierarchy_summary(directory, rows, 1)
            self.assertTrue((directory / 'hierarchy_sample_0_seed_42.npz').exists())
            self.assertFalse((directory / 'hierarchy_sample_0_seed_43.npz').exists())
            summary = json.loads((directory / 'hierarchy_summary.json').read_text())
            self.assertEqual(len(summary['policies']), 8)
            self.assertTrue(all(c['n_images'] == 1 for c in summary['comparisons']))
            self.assertTrue(all('image_bootstrap_95pct_m' not in c for c in summary['comparisons']))
        policies = [r for r in rows if r['type'] == 'hierarchy_result']
        self.assertEqual(len(policies), 16)
        for seed in self.args.seeds:
            group = [r for r in policies if r['seed'] == seed]
            r0 = next(r['rejection']['r0'] for r in group if r['policy'] == 'C_learned')
            for row in group:
                if row['policy'] != 'A_keep_all':
                    self.assertEqual(row['rejection']['r0'], r0)
                self.assertAlmostEqual(row['delta_sse_vs_B'], row['sse_added_vs_B'] - row['sse_removed_vs_B'], places=6)
                if row['policy'] == 'B_r0_only':
                    self.assertEqual(row['delta_sse_vs_B'], 0)
        for key, value in self.wrapper.model_depth.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in self.wrapper.model_depth.modules()))

    def test_paired_intervals_cluster_images_not_seeds(self):
        rows = [dict(sample=sample, seed=seed, policy=policy,
                     metrics={'rmse_m': value})
                for sample in (0, 1) for seed in (42, 43, 44)
                for policy, value in [('B_r0_only', 2.), ('E_oracle_hierarchy', 1.)]]
        result = hierarchy.paired_comparison(rows, 'E_oracle_hierarchy', 'B_r0_only')
        self.assertEqual(result['n_images'], 2)
        self.assertEqual(result['mean_image_rmse_delta_m'], -1)
        self.assertEqual(result['image_bootstrap_95pct_m'], [-1., -1.])

    def test_soft_anchor_formula_and_prediction_gradients(self):
        # The left accepted source is 9; its neighbor predicts 1. Only the
        # right pixel sends left, with weight 0.5. Initial injection is kept.
        scale = torch.tensor([[[[2., 1.]]]], requires_grad=True)
        ratio = torch.tensor([[[[9., 0.]]]])
        keep = torch.tensor([[[[1., 0.]]]], requires_grad=True)
        logits = torch.full((1, 9, 1, 2), -100.)
        logits[0, 3, 0, 1] = 0.
        logits.requires_grad_()
        old = diagnostic.original.propagate_scale(scale, ratio, keep, logits, 1, 0.)
        soft = diagnostic.original.propagate_scale(scale, ratio, keep, logits, 1, 2.)
        self.assertAlmostEqual(float(old[0, 0, 0, 0]), 9.5 / 1.5, places=5)
        self.assertAlmostEqual(float(soft[0, 0, 0, 0]), 27.5 / 3.5, places=5)
        self.assertLess(abs(float(soft[0, 0, 0, 0]) - 9), abs(float(old[0, 0, 0, 0]) - 9))
        soft[0, 0, 0, 0].backward()
        self.assertGreater(float(scale.grad[0, 0, 0, 1]), 0)
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(keep.grad.abs().sum()), 0)
        # The pre-existing one-time hard injection still replaces the inherited
        # scale at this accepted pixel; soft tethering does not undo that fact.
        self.assertEqual(float(scale.grad[0, 0, 0, 0]), 0)

    def test_soft_anchor_zero_eligibility_preserves_old_update(self):
        scale = torch.rand(1, 1, 3, 5)
        ratio = torch.rand_like(scale) * 10
        keep = (torch.rand_like(scale) > .5).float()
        logits = torch.randn(1, 9, 3, 5)
        old = diagnostic.original.propagate_scale(scale, ratio, keep, logits, 6, 0.)
        masked = diagnostic.original.propagate_scale(
            scale, ratio, keep, logits, 6, 2., torch.zeros_like(keep))
        torch.testing.assert_close(old, masked, rtol=0, atol=0)

    def test_anchor_strength_reaches_every_level_without_new_parameters(self):
        for wrapper_class in (diagnostic.ScalePropagationDepthCompletionModel,
                              diagnostic.FixedGeometryScalePropagationDepthCompletionModel):
            kwargs = dict(device=torch.device('cpu'), encoder_type='conv',
                          n_propagation_iterations=(1,) * 5)
            previous = wrapper_class(propagation_anchor_strength=0., **kwargs)
            updated = wrapper_class(propagation_anchor_strength=2., **kwargs)
            updated.model_depth.load_state_dict(previous.model_depth.state_dict(), strict=True)
            modules = [m for m in updated.model_depth.modules()
                       if isinstance(m, diagnostic.original.ScalePropagation)]
            self.assertEqual(len(modules), 5)
            self.assertTrue(all(m.propagation_anchor_strength == 2. for m in modules))

    def test_anchor_suite_frozen_masks_and_checkpoint_unchanged(self):
        # Production defaults enable tethering, but the diagnostic baseline
        # must explicitly be the previous equation, as in the CLI loader.
        for module in self.wrapper.model_depth.modules():
            if isinstance(module, diagnostic.original.ScalePropagation):
                module.propagation_anchor_strength = 0.
        self.args.seeds = [42]
        self.args.anchor_strengths = [2.]
        self.args.n_propagation_iterations = [1] * 5
        rows = []
        before = {k: v.clone() for k, v in self.wrapper.model_depth.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            anchors.diagnose_anchor_sample(
                self.wrapper, self.image, self.sparse, self.target, self.args, 0,
                directory, lambda kind, **values: rows.append(dict(type=kind, **values)),
                save_maps=True)
            anchors.write_anchor_summary(directory, rows)
            summary = json.loads((directory / 'anchor_summary.json').read_text())
            self.assertEqual(len(summary['policies']), 5)
            self.assertEqual(len(summary['comparisons']), 4)
            self.assertTrue((directory / 'anchor_sample_0_seed_42.npz').exists())
        self.assertEqual(sum(r['type'] == 'anchor_local_replay' for r in rows), 15)
        self.assertEqual(next(r['max_depth_difference_m'] for r in rows if r['type'] == 'anchor_replay'), 0)
        for key, value in self.wrapper.model_depth.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in self.wrapper.model_depth.modules()))

    def test_cli_produces_shareable_zip_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'model.pth'
            self.wrapper.save_model(str(checkpoint), 123)
            for name, value in [('rgb', self.image), ('sparse', self.sparse), ('gt', self.target)]:
                if name == 'rgb':
                    pixels = (value[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                else:
                    pixels = (value[0, 0].numpy() * 256).astype(np.uint16)
                Image.fromarray(pixels).save(str(root / (name + '.png')))
                (root / (name + '.txt')).write_text(str(root / (name + '.png')) + '\n')
            arguments = ['diagnose', '--checkpoint', str(checkpoint), '--output', str(root / 'result'),
                         '--images', str(root / 'rgb.txt'), '--sparse_depths', str(root / 'sparse.txt'),
                         '--ground_truths', str(root / 'gt.txt'), '--device', 'cpu', '--encoder_type', 'conv',
                         '--indices', '0', '--points_per_group', '0', '--skip_gradients',
                         '--min_predict_depth', '0.1', '--max_predict_depth', '8',
                         '--n_propagation_iterations', '1', '1', '1', '1', '1']
            with patch.object(sys, 'argv', arguments):
                diagnostic.main()
            with zipfile.ZipFile(root / 'result.zip') as bundle:
                self.assertIn('result/report.md', bundle.namelist())
                provenance = json.loads(bundle.read('result/provenance.json'))
                self.assertEqual(provenance['checkpoint_step'], 123)
                rows = [json.loads(line) for line in bundle.read('result/diagnostics.jsonl').splitlines()]
                self.assertTrue(any(row['type'] == 'input_audit' for row in rows))
            arguments[arguments.index('--output') + 1] = str(root / 'hierarchy')
            arguments += ['--suite', 'hierarchy', '--seeds', '42', '43', '--hierarchy_map_samples', '0']
            with patch.object(sys, 'argv', arguments):
                diagnostic.main()
            with zipfile.ZipFile(root / 'hierarchy.zip') as bundle:
                self.assertIn('hierarchy/hierarchy_summary.json', bundle.namelist())
                rows = [json.loads(line) for line in bundle.read('hierarchy/diagnostics.jsonl').splitlines()]
                self.assertEqual(sum(r['type'] == 'hierarchy_result' for r in rows), 16)
            arguments[arguments.index('--output') + 1] = str(root / 'anchors')
            arguments[arguments.index('--suite') + 1] = 'anchors'
            arguments += ['--anchor_strengths', '2']
            with patch.object(sys, 'argv', arguments):
                diagnostic.main()
            with zipfile.ZipFile(root / 'anchors.zip') as bundle:
                self.assertIn('anchors/anchor_summary.json', bundle.namelist())
                rows = [json.loads(line) for line in bundle.read('anchors/diagnostics.jsonl').splitlines()]
                self.assertEqual(sum(r['type'] == 'anchor_result' for r in rows), 10)


if __name__ == '__main__':
    unittest.main()
