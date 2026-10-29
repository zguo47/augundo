'''Checkpoint-only causal probes for the two hierarchical scale models.

Nothing in the training path imports this module. Hooks and function patches
exist only during one diagnostic forward and are removed even on failure.
Ground truth interventions are diagnostic oracles, never inference features.
Run from the repository root; see bash/diagnose/README.md.
'''

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scale_propagation_model as original
import fixed_geometry_scale_propagation_model as revised
from scale_propagation_depth_completion_model import ScalePropagationDepthCompletionModel
from fixed_geometry_scale_propagation_depth_completion_model import FixedGeometryScalePropagationDepthCompletionModel
from utils.src import data_utils, loss_utils


def resize(value, size):
    return functional.interpolate(value, size=size, mode='bilinear', align_corners=True)


def array(value):
    return value.detach().cpu().numpy()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def boundary_mask(target):
    '''Only label edges whose two endpoints have measured positive GT.'''
    result = torch.zeros_like(target, dtype=torch.bool)
    for axis in (-2, -1):
        left = target.narrow(axis, 0, target.shape[axis] - 1)
        right = target.narrow(axis, 1, target.shape[axis] - 1)
        edge = (left > 0) & (right > 0) & \
            ((left - right).abs() > torch.maximum(left.new_tensor(0.3), 0.05 * torch.minimum(left, right)))
        result.narrow(axis, 0, target.shape[axis] - 1).logical_or_(edge)
        result.narrow(axis, 1, target.shape[axis] - 1).logical_or_(edge)
    return functional.max_pool2d(result.float(), 5, stride=1, padding=2).bool()


def error_metrics(prediction, target, mask):
    '''Meters for MAE/RMSE, 1/km for inverse errors, and m^2 for SSE.'''
    selected = prediction[mask]
    actual = target[mask]
    if selected.numel() == 0:
        return {'count': 0}
    difference = selected - actual
    finite = torch.isfinite(selected)
    result = dict(count=selected.numel(), nonfinite=int((~finite).sum()),
                  nonpositive=int((selected <= 0).sum()))
    if not finite.all():
        return result
    result.update(mae_m=float(difference.abs().mean()),
                  rmse_m=float(difference.square().mean().sqrt()),
                  sse_m2=float(difference.double().square().sum()))
    if (selected > 0).all():
        inverse = 1000.0 / selected - 1000.0 / actual
        result.update(imae_per_km=float(inverse.abs().mean()),
                      irmse_per_km=float(inverse.square().mean().sqrt()))
    return result


def summaries(value, mask):
    selected = value[mask]
    if not selected.numel():
        return {'count': 0}
    finite = selected[torch.isfinite(selected)]
    result = {'count': selected.numel(), 'nonfinite': int(selected.numel() - finite.numel())}
    if finite.numel():
        result.update(mean=float(finite.mean()), mean_abs=float(finite.abs().mean()),
                      min=float(finite.min()), max=float(finite.max()))
    return result


def oracle_propagation(scale, ratio, keep, logits, iterations, gt):
    '''Suppress outgoing edges across observed GT jumps; unknown edges stay.

    unfold places each destination GT in the source pixel's 3x3 channel order.
    fold sends that source's weighted scale to the corresponding destination.
    This removes numerator AND denominator contributions, not merely logits.
    At coarse resolutions GT is nearest-sampled, not filled or averaged.
    '''
    batch, _, height, width = scale.shape
    neighbor = functional.unfold(gt, 3, padding=1).reshape(batch, 9, height, width)
    known = (gt > 0) & (neighbor > 0)
    jump = (gt - neighbor).abs() > torch.maximum(gt.new_tensor(0.3), 0.05 * torch.minimum(gt, neighbor))
    weights = torch.sigmoid(logits) * (~(known & jump)).to(scale.dtype)
    incoming_weight = functional.fold(weights.reshape(batch, 9, -1), (height, width), 3, padding=1)
    scale = keep * ratio + (1 - keep) * scale
    for _ in range(iterations):
        incoming = functional.fold((weights * scale).reshape(batch, 9, -1), (height, width), 3, padding=1)
        scale = (scale + incoming) / (1 + incoming_weight)
    return scale


class Probe:
    '''One forward with optional frozen gates and a single intervention.

    Gate order in both models is r4, r3, r2, r1, r0. Frozen masks isolate a
    decision from later stochastic rejections, while downstream features and
    weights are recomputed normally. This is a controlled effect, not the
    total effect of resampling the whole network after the intervention.
    '''

    def __init__(self, wrapper, target, sparse, frozen=None, intervention=None):
        self.wrapper = wrapper
        self.target = target
        self.sparse = sparse
        self.frozen = frozen
        self.intervention = intervention or {}
        self.gates = {}
        self.contexts = {}
        self.stages = []
        self.handles = []
        self.stack = ExitStack()
        self.gate_count = 0
        self.adaptation_seen = set()

    def save_stage(self, level, name, geometry, scale):
        # CPU maps avoid holding activations from every stage on the GPU.
        self.stages.append((level, name, array(geometry * scale)))

    def gate(self, rejection_logits, sparse_depth, temperature, training):
        level = 4 - self.gate_count
        self.gate_count += 1
        keep = self.native_gate(rejection_logits, sparse_depth, temperature, training)
        if self.frozen is not None:
            keep = self.frozen[level].to(keep)
        chosen = self.intervention
        if chosen.get('level') == level:
            if chosen['kind'] == 'point':
                keep = keep.clone()
                keep.flatten()[chosen['index']] = chosen['keep']
            elif chosen['kind'] == 'oracle_reject':
                selected, indices = original.sparse_depth_at_resolution(self.sparse, keep.shape[-2:], True)
                matched = torch.gather(self.target.flatten(2), 2, indices.flatten(2)).reshape_as(selected)
                labels = loss_utils.rejection_supervision_targets(selected, matched)
                # Never add new anchors. Only remove known GT-disagreeing ones.
                keep = keep * (labels != 1).to(keep.dtype)
        self.gates[level] = (keep, rejection_logits)
        return keep

    def propagation(self, module, inputs, output, level):
        scale, geometry, sparse, keep, logits, ratio = inputs
        chosen = self.intervention
        if chosen.get('kind') == 'separation' and level == 0:
            from diagnose_propagation_separation import intervene
            output, self.separation_details = intervene(
                inputs, self.target, module.n_propagation_iterations,
                module.propagation_anchor_strength, chosen)
        if chosen.get('kind') == 'soft_anchor' and level in chosen['levels']:
            anchor_mask = None
            if chosen.get('good_only'):
                _, _, labels, _ = point_groups(self.sparse, self.target, scale.shape[-2:])
                anchor_mask = (labels == 0).to(keep)
            output = original.propagate_scale(
                scale, ratio, keep, logits, module.n_propagation_iterations,
                chosen['strength'], anchor_mask)
        if chosen.get('level') == level:
            if chosen['kind'] == 'no_mixing':
                output = keep * ratio + (1 - keep) * scale
            elif chosen['kind'] == 'oracle_edges':
                gt = functional.interpolate(self.target, size=scale.shape[-2:], mode='nearest')
                output = oracle_propagation(scale, ratio, keep, logits,
                                            module.n_propagation_iterations, gt)
        self.contexts[level] = {name: array(value) for name, value in zip(
            ['scale', 'geometry', 'sparse', 'keep', 'logits', 'ratio'], inputs)}
        self.save_stage(level, 'before_propagation', geometry, scale)
        self.save_stage(level, 'after_propagation', geometry, output)
        return output

    def refinement(self, module, inputs, output, level, pre):
        # Both heads have C+5 inputs, but scalar channels have different meaning.
        values = inputs[0]
        geometry, scale = values[:, -5:-4], values[:, -4:-3]
        if pre:
            # With multiple refinement passes, geometry changes only once.
            # Do not report a fictitious repeated geometry update each pass.
            if level not in self.adaptation_seen:
                self.save_stage(level, 'after_geometry_update', geometry, scale)
                self.adaptation_seen.add(level)
        if self.intervention.get('level') == level and \
                self.intervention.get('kind') == ('no_adaptation' if pre else 'no_refinement'):
            output = torch.zeros_like(output)
        self.save_stage(level, 'after_adaptation' if pre else 'after_refinement',
                        geometry, scale * torch.exp(torch.tanh(output)))
        return output

    def __enter__(self):
        self.native_gate = original.rejection_keep_mask
        self.stack.enter_context(patch.object(original, 'rejection_keep_mask', self.gate))
        self.stack.enter_context(patch.object(revised, 'rejection_keep_mask', self.gate))
        model = self.wrapper.model_depth
        for level, block in [(4, model)] + [(3 - i, block) for i, block in enumerate(model.decoder)]:
            if level < 4:
                # Fusion still sees interpolated old geometry/scale before
                # the geometry update. Optional raw sparse is its last channel.
                offset = 1 if block.use_sparse_depth_features else 0
                def before_fusion(module, inputs, r=level, extra=offset):
                    values = inputs[0]
                    end = values.shape[1] - extra
                    self.save_stage(r, 'after_upsampling_before_geometry',
                                    values[:, end-2:end-1], values[:, end-1:end])
                self.handles.append(block.fusion.register_forward_pre_hook(before_fusion))
            self.handles.append(block.scale_propagation.register_forward_hook(
                lambda module, inputs, output, r=level: self.propagation(module, inputs, output, r)))
            for name in ('scale_update', 'post_scale_update'):
                if hasattr(block, name):
                    head = getattr(block, name)
                    pre = getattr(head, 'pre_propagation_refinement', False)
                    self.handles.append(head.register_forward_hook(
                        lambda module, inputs, output, r=level, p=pre:
                        self.refinement(module, inputs, output, r, p)))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        return self.stack.__exit__(*exception)


def objective(wrapper, outputs, probe, target, sparse, args):
    loss, info = wrapper.compute_loss_supervised(target, outputs, {
        'w_supervised': args.w_supervised, 'w_si': args.w_si, 'w_log_l1': args.w_log_l1})
    if args.rejection_loss != 'none':
        index = 0 if args.rejection_loss == 'l1' else 1
        values = [probe.gates[r][index] for r in range(5)]
        rejection, _ = wrapper.compute_loss_rejection(
            target, sparse, values, supervision_levels=args.rejection_levels,
            loss_type=args.rejection_loss, keep_weight=args.rejection_keep_weight)
        loss = loss + args.w_rejection * rejection
        info['loss_rejection_unweighted'] = rejection
    info['objective'] = loss
    return loss, info


def forward(wrapper, image, sparse, target, args, seed, frozen=None, intervention=None, gradients=False):
    seed_everything(seed)
    with torch.set_grad_enabled(gradients), Probe(wrapper, target, sparse, frozen, intervention) as probe:
        outputs = wrapper.forward_depth(image, sparse, return_all_outputs=True)
        loss, info = objective(wrapper, outputs, probe, target, sparse, args)
        if gradients:
            tensors = [tensor for r in range(5) for tensor in probe.gates[r]]
            grads = torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
            probe.gradients = {
                r: [array(g) if g is not None else np.zeros_like(array(t))
                    for t, g in zip(tensors[2*r:2*r+2], grads[2*r:2*r+2])]
                for r in range(5)}
            metric_grads = torch.autograd.grad(info['loss_log_l1_weighted'], tensors, allow_unused=True)
            probe.metric_gradients = {
                r: [array(g) if g is not None else np.zeros_like(array(t))
                    for t, g in zip(tensors[2*r:2*r+2], metric_grads[2*r:2*r+2])]
                for r in range(5)}
        probe.losses = {name: float(value.detach()) for name, value in info.items()}
        probe.frozen = {r: keep.detach().clone() for r, (keep, _) in probe.gates.items()}
        probe.scores = {r: array(logits) for r, (_, logits) in probe.gates.items()}
        probe.gates.clear()
        return outputs[0].detach().cpu(), probe


def stage_report(probe, target, mask, worst):
    result, previous = [], None
    for level, name, native in probe.stages:
        prediction = resize(torch.from_numpy(native), target.shape[-2:])
        squared = (prediction - target).square()
        row = dict(level=level, stage=name, **error_metrics(prediction, target, mask))
        row['worst_final_1pct_sse_m2'] = float(squared[worst].double().sum())
        if previous is not None:
            delta = (squared - previous)[mask]
            row.update(sse_added_m2=float(delta.clamp_min(0).double().sum()),
                       sse_removed_m2=float((-delta).clamp_min(0).double().sum()))
        result.append(row)
        previous = squared
    return result


def point_groups(sparse, target, size):
    selected, indices = original.sparse_depth_at_resolution(sparse, size, True)
    matched = torch.gather(target.flatten(2), 2, indices.flatten(2)).reshape_as(selected)
    labels = loss_utils.rejection_supervision_targets(selected, matched)
    edge = torch.gather(boundary_mask(target).flatten(2), 2, indices.flatten(2)).reshape_as(selected)
    return selected, indices, labels, edge


def diagnose_sample(wrapper, image, sparse, target, args, sample, directory, emit):
    '''Do not optimize weights. Every trial starts with identical model state.'''
    device_target = target.to(wrapper.device)
    seed = args.seed + sample
    prediction, base = forward(wrapper, image, sparse, device_target, args, seed, gradients=not args.skip_gradients)
    target = target.cpu()
    mask = torch.isfinite(target) & (target > args.min_evaluate_depth) & (target < args.max_evaluate_depth)
    mask &= target > 0
    if not mask.any():
        raise ValueError('No evaluated GT pixels in sample {}'.format(sample))
    errors = (prediction - target).square()
    worst = torch.zeros_like(mask)
    valid_indices = mask.flatten().nonzero().flatten()
    count = max(1, int(valid_indices.numel() * 0.01))
    worst.flatten()[valid_indices[torch.topk(errors.flatten()[valid_indices], count).indices]] = True
    metrics = error_metrics(prediction, target, mask)
    emit('baseline', sample=sample, metrics=metrics, losses=base.losses)
    measured = sparse.detach().cpu()
    overlap = mask & (measured > 0)
    emit('input_audit', sample=sample, shape=list(image.shape),
         sparse_count=int((measured > 0).sum()), evaluated_gt_count=int(mask.sum()),
         sparse_vs_gt=error_metrics(measured, target, overlap),
         output_vs_sparse=error_metrics(prediction, measured, (measured > 0) & torch.isfinite(measured)))
    groups = {'at_sparse': sparse.cpu() > 0, 'away_sparse': sparse.cpu() <= 0,
              'gt_boundary_5px': boundary_mask(target), 'worst_final_1pct': worst}
    for lower, upper in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, float('inf'))]:
        groups['depth_{}_{}'.format(lower, upper)] = (target >= lower) & (target < upper)
    for name, group in groups.items():
        emit('error_group', sample=sample, group=name, metrics=error_metrics(prediction, target, mask & group))
    emit('stages', sample=sample, stages=stage_report(base, target, mask, worst))
    # Save small maps for independent re-analysis. No RGB, GT, or model weights
    # are included by default; native predictions can still reveal scene shape.
    maps = {'prediction': prediction.numpy()}
    for i, (level, stage, value) in enumerate(base.stages):
        maps['stage_{}_r{}_{}'.format(i, level, stage)] = value

    # Replay is a prerequisite for interpreting small counterfactual changes.
    replay, repeat = forward(wrapper, image, sparse, device_target, args, seed, base.frozen)
    delta = float((prediction - replay).abs().max())
    emit('replay', sample=sample, max_depth_difference_m=delta,
         objective_difference=repeat.losses['objective'] - base.losses['objective'])
    if delta > 1e-5:
        raise RuntimeError('Frozen-mask replay differs; do not interpret counterfactuals.')
    del repeat

    for level in range(4, -1, -1):
        context = base.contexts[level]
        selected, indices, labels, edges = point_groups(sparse.cpu(), target, context['scale'].shape[-2:])
        keep = base.frozen[level].cpu()
        probability = torch.softmax(torch.from_numpy(base.scores[level]), 1)[:, 0:1]
        ratio_gap = torch.from_numpy(context['ratio'] - context['scale'])
        maps['r{}_keep'.format(level)] = keep.numpy()
        maps['r{}_keep_probability'.format(level)] = probability.numpy()
        maps['r{}_labels'.format(level)] = labels.numpy()
        maps['r{}_ratio_minus_scale'.format(level)] = ratio_gap.numpy()
        emit('rejection', sample=sample, level=level,
             available=int((selected > 0).sum()), accepted=int(keep.sum()),
             good_kept=int(((labels == 0) & (keep > 0)).sum()),
             good_rejected=int(((labels == 0) & (keep == 0)).sum()),
             bad_kept=int(((labels == 1) & (keep > 0)).sum()),
             bad_rejected=int(((labels == 1) & (keep == 0)).sum()))
        for name, group in [('good', labels == 0), ('bad', labels == 1),
                            ('boundary_known', edges & (labels >= 0))]:
            row = dict(sample=sample, level=level, group=name,
                       keep_probability=summaries(probability, group),
                       ratio_minus_inherited_scale=summaries(ratio_gap, group))
            if not args.skip_gradients:
                grad_keep, grad_logits = base.gradients[level]
                row['dobjective_dkeep'] = summaries(torch.from_numpy(grad_keep), group)
                row['dobjective_dkeep_logit'] = summaries(torch.from_numpy(grad_logits[:, 0:1]), group)
                metric_keep, metric_logits = base.metric_gradients[level]
                row['dmetric_loss_dkeep'] = summaries(torch.from_numpy(metric_keep), group)
                row['dmetric_loss_dkeep_logit'] = summaries(torch.from_numpy(metric_logits[:, 0:1]), group)
                maps['r{}_dobjective_dkeep'.format(level)] = grad_keep
                maps['r{}_dobjective_dlogits'.format(level)] = grad_logits
                maps['r{}_dmetric_loss_dkeep'.format(level)] = metric_keep
                maps['r{}_dmetric_loss_dlogits'.format(level)] = metric_logits
            emit('gradient_group', **row)
            candidates = group.flatten().nonzero().flatten().numpy()
            rng = np.random.RandomState(seed + 17 * level)
            for index in rng.permutation(candidates)[:args.points_per_group]:
                trials = {}
                for decision in (0, 1):
                    changed, trial = forward(wrapper, image, sparse, device_target, args, seed,
                        base.frozen, dict(kind='point', level=level, index=int(index), keep=decision))
                    trials[decision] = dict(objective=trial.losses['objective'],
                                            **error_metrics(changed, target, mask))
                    trials[decision]['local_stages'] = [
                        dict(stage=name, **error_metrics(resize(torch.from_numpy(value), target.shape[-2:]), target, mask))
                        for r, name, value in trial.stages if r == level and name in
                        ('after_propagation', 'after_refinement')]
                coordinate = int(indices.flatten()[int(index)])
                row = dict(sample=sample, level=level, group=name, index=int(index),
                           original_y=coordinate // target.shape[-1], original_x=coordinate % target.shape[-1],
                           keep_probability=float(probability.flatten()[int(index)]),
                           baseline_keep=float(keep.flatten()[int(index)]),
                           reject=trials[0], keep=trials[1],
                           keep_minus_reject_objective=trials[1]['objective'] - trials[0]['objective'],
                           keep_minus_reject_sse_m2=trials[1].get('sse_m2', float('nan')) - trials[0].get('sse_m2', float('nan')))
                if not args.skip_gradients:
                    row['local_dobjective_dkeep'] = float(base.gradients[level][0].reshape(-1)[int(index)])
                emit('point_counterfactual', **row)
        kinds = ['no_mixing', 'oracle_reject', 'oracle_edges', 'no_refinement']
        if level != 4 and getattr(wrapper.model_depth, 'pre_propagation_refinement', False):
            kinds.append('no_adaptation')
        for kind in kinds:
            changed, trial = forward(wrapper, image, sparse, device_target, args, seed,
                                     base.frozen, dict(kind=kind, level=level))
            measured = error_metrics(changed, target, mask)
            emit('stage_intervention', sample=sample, level=level, intervention=kind,
                 metrics=measured, objective=trial.losses['objective'],
                 delta_objective=trial.losses['objective'] - base.losses['objective'],
                 delta_sse_m2=measured.get('sse_m2', float('nan')) - metrics.get('sse_m2', float('nan')),
                 stages=stage_report(trial, target, mask, worst))
        print('sample {}: completed r{} interventions'.format(sample, level), flush=True)

    # Trace immediate incoming neighbors at the worst full-resolution pixels.
    # These are local mixing paths, NOT a claim to recover every multi-hop
    # anchor influence or identify the original source of the entire error.
    context = base.contexts[0]
    height, width = target.shape[-2:]
    weights = torch.sigmoid(torch.from_numpy(context['logits']))
    for index in valid_indices[torch.topk(errors.flatten()[valid_indices], min(10, valid_indices.numel())).indices]:
        y, x = divmod(int(index), width)
        sources = []
        for channel in range(9):
            dy, dx = channel // 3 - 1, channel % 3 - 1
            sy, sx = y - dy, x - dx
            if 0 <= sy < height and 0 <= sx < width:
                sources.append(dict(y=sy, x=sx, outgoing_weight=float(weights[0, channel, sy, sx]),
                                    **{key: float(context[key][0, 0, sy, sx])
                                       for key in ('scale', 'geometry', 'sparse', 'keep', 'ratio')}))
        emit('worst_pixel_neighbors', sample=sample, y=y, x=x, prediction=float(prediction[0, 0, y, x]),
             ground_truth=float(target[0, 0, y, x]), sources=sources)
    np.savez_compressed(str(directory / 'sample_{}_maps.npz'.format(sample)), **maps)
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', required=True)
    result.add_argument('--output', required=True, help='New directory; existing directories are not overwritten.')
    result.add_argument('--images', required=True)
    result.add_argument('--sparse_depths', required=True)
    result.add_argument('--ground_truths', required=True)
    result.add_argument('--model', choices=['original', 'adaptation'], default='adaptation')
    result.add_argument('--indices', nargs='+', type=int, default=None)
    result.add_argument('--suite', choices=['isolated', 'hierarchy', 'anchors', 'separation'], default='isolated')
    result.add_argument('--propagation_anchor_strength', type=float, default=2.0,
                        help='Native baseline strength for separation suite; match checkpoint training')
    result.add_argument('--anchor_strengths', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                        help='Persistent source weights for paired anchor tests; not trained parameters')
    result.add_argument('--sample_count', type=int, default=None,
                        help='Uniformly spaced dataset indices; 0 means all. Explicit --indices takes precedence.')
    result.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Paired hierarchy repetitions; defaults to --seed.')
    result.add_argument('--hierarchy_map_samples', type=int, default=3,
                        help='Save policy maps for the first N selected images, first seed only.')
    result.add_argument('--points_per_group', type=int, default=2)
    result.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--encoder_type', choices=['conv', 'conv_attention'], default='conv_attention')
    result.add_argument('--n_propagation_iterations', nargs=5, type=int, default=[1, 2, 3, 4, 6])
    result.add_argument('--n_scale_refinement', type=int, default=1)
    boolean = lambda value: value.lower() == 'true' if value.lower() in ('true', 'false') else result.error('Boolean must be true or false')
    result.add_argument('--detach_scaleless_depth', type=boolean, default=True)
    result.add_argument('--use_sparse_depth_features', type=boolean, default=False)
    result.add_argument('--min_predict_depth', type=float, default=1.5)
    result.add_argument('--max_predict_depth', type=float, default=100.0)
    result.add_argument('--min_evaluate_depth', type=float, default=0.0)
    result.add_argument('--max_evaluate_depth', type=float, default=100.0)
    result.add_argument('--w_supervised', type=float, default=1.0)
    result.add_argument('--w_si', type=float, default=1.0)
    result.add_argument('--w_log_l1', type=float, default=1.0)
    result.add_argument('--rejection_loss', choices=['none', 'cross_entropy', 'l1'], default='none')
    result.add_argument('--w_rejection', type=float, default=0.1, help='For L1, set this to the training w_rejection_l1.')
    result.add_argument('--rejection_levels', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    result.add_argument('--rejection_keep_weight', type=float, default=2.0)
    result.add_argument('--skip_gradients', action='store_true', help='Lower memory; still runs discrete interventions.')
    return result


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_summary(directory, records, elapsed):
    lines = ['# Scale propagation diagnosis', '',
             'No optimizer updates. Evaluation images are unaugmented and uncropped.',
             'The objective uses the configured training losses on these evaluation inputs.',
             'Other rejection masks are frozen in all interventions. Negative delta SSE is better.',
             'GT labels/oracles are imperfect diagnostics, not inference proposals.', '',
             '| Sample | MAE (m) | RMSE (m) |', '|---|---:|---:|']
    for row in records:
        if row['type'] == 'baseline':
            metric = row['metrics']
            lines.append('| {} | {:.4f} | {:.4f} |'.format(row['sample'], metric.get('mae_m', float('nan')), metric.get('rmse_m', float('nan'))))
    lines += ['', '## Rejection decisions', '',
              '| Sample | Level | Good rejected / known good | Bad rejected / known bad |',
              '|---|---|---:|---:|']
    for row in records:
        if row['type'] == 'rejection':
            lines.append('| {} | r{} | {} / {} | {} / {} |'.format(
                row['sample'], row['level'], row['good_rejected'], row['good_kept'] + row['good_rejected'],
                row['bad_rejected'], row['bad_kept'] + row['bad_rejected']))
    lines += ['', '## Single-point decision tests', '',
              'Positive keep-minus-reject values favor **rejecting** the point. Small values need replay/noise checks.',
              '', '| Sample | Level | Group | Original (y,x) | Keep probability | Delta objective | Delta SSE (m²) |',
              '|---|---|---|---|---:|---:|---:|']
    for row in records:
        if row['type'] == 'point_counterfactual':
            lines.append('| {} | r{} | {} | ({},{}) | {:.6f} | {:.8f} | {:.5f} |'.format(
                row['sample'], row['level'], row['group'], row['original_y'], row['original_x'],
                row['keep_probability'], row['keep_minus_reject_objective'], row['keep_minus_reject_sse_m2']))
    lines += ['', '## Stage interventions (negative is improvement)', '',
              '| Sample | Level | Intervention | Delta SSE (m²) | Delta objective |', '|---|---|---|---:|---:|']
    for row in records:
        if row['type'] == 'stage_intervention':
            lines.append('| {} | r{} | {} | {:.4f} | {:.7f} |'.format(row['sample'], row['level'], row['intervention'], row['delta_sse_m2'], row['delta_objective']))
    lines += ['', '## How to interpret', '',
              '- Point tests: positive keep-minus-reject SSE means rejecting that point improves RMSE.',
              '- Positive keep-minus-reject objective means the configured loss also favors rejection.',
              '- If those signs disagree, the training objective and RMSE disagree for that decision.',
              '- If both favor rejection but probability stays near 1, inspect the local gate/logit gradients.',
              '- A near-zero discrete effect suggests that this particular gate has little leverage.',
              '- no_mixing retains sparse injection; oracle_edges only suppresses edges with two known GT endpoints.',
              '- no_refinement disables only the post-propagation head; no_adaptation disables only the pre-head.',
              '- Stage SSE changes are sequential local effects; intervention final SSE includes downstream repair.',
              '- Worst-pixel neighbors are immediate paths, not full multi-hop source attribution.',
              '- Do not generalize from three images: repeat with more indices and seeds after locating a signal.',
              '', 'Elapsed: {:.1f} seconds.'.format(elapsed),
              'Share the sibling ZIP. It includes source/config provenance, JSONL records, and native diagnostic maps.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    args = parser().parse_args()
    if args.points_per_group < 0 or any(r not in range(5) for r in args.rejection_levels):
        raise ValueError('Use nonnegative point count and rejection levels 0..4.')
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    # Deterministic replay, including CUDA kernels supported by torch 1.10.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    paths = [data_utils.read_paths(path) for path in (args.images, args.sparse_depths, args.ground_truths)]
    if len({len(values) for values in paths}) != 1:
        raise ValueError('Image/sparse/GT lists have unequal lengths.')
    if (args.sample_count is not None and args.sample_count < 0) or args.hierarchy_map_samples < 0:
        raise ValueError('Sample/map counts must be nonnegative.')
    if args.indices is None:
        if args.sample_count is None:
            args.indices = [0, 499, 999]
        else:
            count = len(paths[0]) if args.sample_count == 0 else min(args.sample_count, len(paths[0]))
            args.indices = np.linspace(0, len(paths[0]) - 1, count, dtype=int).tolist()
    args.indices = list(dict.fromkeys(args.indices))
    args.seeds = list(dict.fromkeys(args.seeds if args.seeds is not None else [args.seed]))
    if not args.indices:
        raise ValueError('The selected dataset is empty.')
    if not all(0 <= index < len(paths[0]) for index in args.indices):
        raise ValueError('Sample indices outside dataset; choose --indices explicitly.')
    wrapper_class = FixedGeometryScalePropagationDepthCompletionModel if args.model == 'adaptation' else ScalePropagationDepthCompletionModel
    wrapper = wrapper_class(
        device=torch.device(args.device), min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth, encoder_type=args.encoder_type,
        detach_scaleless_depth=args.detach_scaleless_depth, use_outlier_rejection=True,
        n_propagation_iterations=args.n_propagation_iterations,
        n_scale_refinement=args.n_scale_refinement, use_sparse_depth_features=args.use_sparse_depth_features,
        propagation_anchor_strength=args.propagation_anchor_strength if args.suite == 'separation' else 0.0)
    step, _ = wrapper.restore_model(args.checkpoint)
    wrapper.eval()
    source_files = [Path(__file__), Path(original.__file__), Path(revised.__file__),
                    Path(loss_utils.__file__), Path(sys.modules[ScalePropagationDepthCompletionModel.__module__].__file__),
                    Path(sys.modules[FixedGeometryScalePropagationDepthCompletionModel.__module__].__file__),
                    Path(sys.modules[original.Conv2d.__module__].__file__), Path(data_utils.__file__)]
    if args.suite == 'hierarchy':
        import diagnose_hierarchical_rejection as hierarchy
        source_files.append(Path(hierarchy.__file__))
    if args.suite == 'anchors':
        import diagnose_anchor_retention as anchors
        import diagnose_hierarchical_rejection as hierarchy
        source_files.extend([Path(anchors.__file__), Path(hierarchy.__file__)])
    if args.suite == 'separation':
        import diagnose_propagation_separation as separation
        import diagnose_hierarchical_rejection as hierarchy
        source_files.extend([Path(separation.__file__), Path(hierarchy.__file__)])
    provenance = dict(arguments=vars(args), checkpoint_step=step, checkpoint_sha256=digest(args.checkpoint),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      python_version=sys.version,
                      device_name=torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
                      source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_files},
                      list_sha256={path: digest(path) for path in (args.images, args.sparse_depths, args.ground_truths)},
                      units='depth meters; inverse errors 1/km; SSE m^2',
                      selected_file_sha256={values[i]: digest(values[i]) for i in args.indices for values in paths},
                      selected_paths={str(i): [values[i] for values in paths] for i in args.indices})
    (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    source_dir = directory / 'sources'
    source_dir.mkdir()
    for path in source_files:
        (source_dir / path.name).write_bytes(path.read_bytes())
    records = []
    with (directory / 'diagnostics.jsonl').open('w') as stream:
        def emit(kind, **values):
            row = dict(type=kind, **values)
            # Nonfinite numbers are explicit nulls, not nonstandard JSON NaN.
            def clean(value):
                if isinstance(value, float) and not np.isfinite(value):
                    return None
                if isinstance(value, dict):
                    return {k: clean(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [clean(v) for v in value]
                return value
            stream.write(json.dumps(clean(row)) + '\n')
            stream.flush()
            records.append(row)
        for sample_order, index in enumerate(args.indices):
            image = torch.from_numpy(data_utils.load_image(paths[0][index], normalize=True, data_format='CHW')).unsqueeze(0).to(wrapper.device)
            sparse = torch.from_numpy(data_utils.load_depth(paths[1][index], data_format='CHW')).unsqueeze(0).to(wrapper.device)
            target = torch.from_numpy(data_utils.load_depth(paths[2][index], data_format='CHW')).unsqueeze(0)
            if image.shape[-2:] != target.shape[-2:] or sparse.shape != target.shape:
                raise ValueError('Image/sparse/GT are not spatially aligned for sample {}'.format(index))
            if args.suite == 'separation':
                separation.diagnose_sample(
                    wrapper, image, sparse, target, args, index, directory, emit,
                    save_maps=sample_order < args.hierarchy_map_samples)
                separation.write_summary(directory, records)
            elif args.suite == 'anchors':
                anchors.diagnose_anchor_sample(
                    wrapper, image, sparse, target, args, index, directory, emit,
                    save_maps=sample_order < args.hierarchy_map_samples)
                anchors.write_anchor_summary(directory, records)
            elif args.suite == 'hierarchy':
                hierarchy.diagnose_hierarchy_sample(
                    wrapper, image, sparse, target, args, index, directory, emit,
                    save_maps=sample_order < args.hierarchy_map_samples)
                hierarchy.write_hierarchy_summary(directory, records, time.time() - started)
            else:
                diagnose_sample(wrapper, image, sparse, target, args, index, directory, emit)
                write_summary(directory, records, time.time() - started)
    archive = directory.with_suffix('.zip') if directory.suffix else Path(str(directory) + '.zip')
    with zipfile.ZipFile(str(archive), 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                bundle.write(str(path), str(path.relative_to(directory.parent)))
    print('Complete. Share: {}'.format(archive), flush=True)


if __name__ == '__main__':
    main()
