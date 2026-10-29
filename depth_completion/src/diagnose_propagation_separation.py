'''Full-resolution factorial oracles for measurement, geometry and connections.

Only r0 propagation is intervened upon. Coarse predictions, learned weights,
and keep decisions are held fixed. GT is never resized or filled. The geometry
oracle changes the coordinate system in which scale is propagated, preserving
the incoming metric prediction; it does not simply return GT as the prediction.
'''

import itertools
import json

import numpy as np
import torch
import torch.nn.functional as functional

from diagnose_scale_propagation import forward, error_metrics, stage_report, original
from diagnose_hierarchical_rejection import pool_metrics, paired_comparison
from utils.src import loss_utils


def policies():
    result = {}
    for a, g, c in itertools.product([False, True], repeat=3):
        name = ''.join(k for k, active in zip('AGC', [a, g, c]) if active) or 'baseline'
        result[name] = dict(kind='separation', measurements=a, geometry=g, connections=c)
    result['injection_only'] = dict(kind='separation', no_mixing=True)
    return result


def intervene(inputs, target, iterations, strength, policy):
    '''Return scale in the ORIGINAL geometry coordinates for downstream heads.

    A corrects only accepted measurements labeled bad by the existing GT rule.
    G substitutes GT-relative geometry only where GT is valid. Re-express the
      same incoming metric depth in that geometry, propagate, then convert back.
    C removes outgoing edges across known GT discontinuities from both sums.
      This is an edge oracle, not perfect surface segmentation. Unknown edges
      remain unchanged. Smooth depth changes on one surface can be misclassified.
    '''
    scale, geometry, sparse, keep, logits, ratio = inputs
    if target.shape != scale.shape:
        raise ValueError('Separation requires full-resolution GT and propagation.')
    known = torch.isfinite(target) & (target > policy.get('min_depth', 0.)) & \
        (target < policy.get('max_depth', float('inf')))
    gt = torch.where(known, target, torch.zeros_like(target))
    labels = loss_utils.rejection_supervision_targets(sparse, gt)
    bad = known & (labels == 1) & (keep > 0)
    measured = torch.where(bad, gt, sparse) if policy.get('measurements') else sparse
    local_geometry = geometry
    local_scale = scale
    local_ratio = ratio
    if policy.get('geometry') and known.any():
        # Align GT to predicted geometry's units. Global scaling cancels when
        # the result is converted back; relative geometry is what changes.
        alignment = torch.exp((torch.log(gt[known]) - torch.log(geometry[known].clamp_min(1e-7))).mean())
        local_geometry = torch.where(known, gt / alignment, geometry)
        local_scale = scale * geometry / local_geometry.clamp_min(1e-7)
    if policy.get('measurements') or policy.get('geometry'):
        local_ratio = measured / local_geometry.clamp_min(1e-7)
    batch, _, height, width = scale.shape
    weights = torch.sigmoid(logits)
    neighbor_gt = functional.unfold(gt, 3, padding=1).reshape(batch, 9, height, width)
    neighbor_known = functional.unfold(known.float(), 3, padding=1).reshape_as(weights) > 0
    inside = functional.unfold(torch.ones_like(gt), 3, padding=1).reshape_as(weights) > 0
    eligible = known & neighbor_known
    jump = eligible & ((gt - neighbor_gt).abs() > torch.maximum(
        gt.new_tensor(.3), .05 * torch.minimum(gt, neighbor_gt)))
    off_center = torch.ones_like(weights, dtype=torch.bool)
    off_center[:, 4] = False
    metadata = dict(known_gt_pixels=int(known.sum()), total_pixels=gt.numel(),
                    accepted_known_bad=int(bad.sum()), accepted_points=int((keep > 0).sum()),
                    known_offcenter_edges=int((eligible & off_center).sum()),
                    inside_offcenter_edges=int((inside & off_center).sum()),
                    jump_edges=int(jump.sum()),
                    jump_weight_sum=float(weights[jump].sum()),
                    known_edge_weight_sum=float(weights[eligible & off_center].sum()))
    if policy.get('connections'):
        weights = weights * (~jump).to(weights)
    if policy.get('no_mixing'):
        weights = torch.zeros_like(weights)
    local_scale = keep * local_ratio + (1. - keep) * local_scale
    anchor_weight = strength * keep
    incoming_weight = functional.fold(weights.reshape(batch, 9, -1), (height, width), 3, padding=1)
    for _ in range(iterations):
        incoming = functional.fold((weights * local_scale).reshape(batch, 9, -1),
                                   (height, width), 3, padding=1)
        local_scale = (local_scale + incoming + anchor_weight * local_ratio) / \
            (1. + incoming_weight + anchor_weight)
    return local_scale * local_geometry / geometry.clamp_min(1e-7), metadata


def diagnose_sample(wrapper, image, sparse, target, args, sample, directory, emit, save_maps=False):
    target = target.cpu()
    valid = torch.isfinite(target) & (target > max(0., args.min_evaluate_depth)) & \
        (target < args.max_evaluate_depth)
    if not valid.any():
        raise ValueError('No evaluated GT for sample {}'.format(sample))
    for seed_index, seed in enumerate(args.seeds):
        actual_seed = (seed + 1000003 * sample) % (2 ** 32)
        output, base = forward(wrapper, image, sparse, target.to(wrapper.device), args, actual_seed)
        replay, check = forward(wrapper, image, sparse, target.to(wrapper.device), args, actual_seed, base.frozen)
        delta = float((output - replay).abs().max())
        if not torch.isfinite(output).all() or delta > 1e-5:
            raise RuntimeError('Native baseline is nonfinite or frozen replay failed.')
        emit('separation_replay', sample=sample, seed=seed, max_difference_m=delta,
             native_anchor_strength=args.propagation_anchor_strength)
        del check, replay
        squared = (output - target).square()
        indices = valid.flatten().nonzero().flatten()
        worst = torch.zeros_like(valid)
        worst.flatten()[indices[torch.topk(squared.flatten()[indices], max(1, int(indices.numel() * .01))).indices]] = True
        labels = loss_utils.rejection_supervision_targets(sparse.cpu(), target)
        groups = dict(baseline_worst_1pct=worst, away_sparse=sparse.cpu() <= 0,
                      accepted_good=(base.frozen[0].cpu() > 0) & (labels == 0),
                      accepted_bad=(base.frozen[0].cpu() > 0) & (labels == 1),
                      gt_over_40m=target > 40)
        values = base.contexts[0]
        inputs = tuple(torch.from_numpy(values[k]) for k in ['scale', 'geometry', 'sparse', 'keep', 'logits', 'ratio'])
        maps = dict(target=target.numpy(), sparse=sparse.cpu().numpy(), evaluated=valid.numpy())
        native_after = next(torch.from_numpy(d) for r, name, d in base.stages if r == 0 and name == 'after_propagation')
        for name, policy in policies().items():
            policy.update(min_depth=max(0., args.min_evaluate_depth), max_depth=args.max_evaluate_depth)
            local_scale, coverage = intervene(inputs, target, args.n_propagation_iterations[-1],
                                               args.propagation_anchor_strength, policy)
            local_depth = local_scale * inputs[1]
            if name == 'baseline':
                # Independent replay must match the production implementation.
                torch.testing.assert_close(local_depth, native_after, rtol=1e-5, atol=1e-4)
                prediction, probe = output, base
            else:
                prediction, probe = forward(wrapper, image, sparse, target.to(wrapper.device), args,
                                            actual_seed, base.frozen, policy)
                if any(not torch.equal(probe.frozen[r], base.frozen[r]) for r in range(5)):
                    raise RuntimeError('Rejection decisions changed during separation.')
                actual_after = next(torch.from_numpy(d) for r, stage, d in probe.stages
                                    if r == 0 and stage == 'after_propagation')
                torch.testing.assert_close(local_depth, actual_after, rtol=1e-4, atol=1e-3)
            emit('separation_result', sample=sample, seed=seed, actual_seed=actual_seed,
                 policy=name, coverage=coverage, metrics=error_metrics(prediction, target, valid),
                 local_metrics=error_metrics(local_depth, target, valid),
                 losses=probe.losses, groups={k: error_metrics(prediction, target, valid & m) for k,m in groups.items()},
                 local_groups={k: error_metrics(local_depth, target, valid & m) for k,m in groups.items()},
                 stages=stage_report(probe, target, valid, worst))
            if save_maps and seed_index == 0:
                maps[name + '_final'] = prediction.numpy()
                maps[name + '_after_propagation'] = local_depth.numpy()
            if name != 'baseline':
                del probe
        if save_maps and seed_index == 0:
            maps.update({'baseline_r0_' + k: v for k,v in values.items()})
            np.savez_compressed(str(directory / 'separation_sample_{}_seed_{}.npz'.format(sample, seed)), **maps)
        print('separation sample {}, seed {} complete'.format(sample, seed), flush=True)


def write_summary(directory, records):
    rows = [r for r in records if r['type'] == 'separation_result']
    names = list(policies())
    result = dict(policies={}, comparisons={}, scope='r0 propagation only; native-strength baseline')
    for name in names:
        selected = [r for r in rows if r['policy'] == name]
        result['policies'][name] = dict(
            metrics=pool_metrics([r['metrics'] for r in selected]),
            local_metrics=pool_metrics([r['local_metrics'] for r in selected]),
            groups={k: pool_metrics([r['groups'][k] for r in selected]) for k in selected[0]['groups']},
            coverage_totals={k: sum(r['coverage'][k] for r in selected) for k in selected[0]['coverage']})
    # Compare each factor both alone and in the presence of the other factors.
    pairs = [(n, 'baseline') for n in names if n != 'baseline']
    for factor in 'AGC':
        for name in names:
            if name == 'injection_only' or factor in name:
                continue
            other = '' if name == 'baseline' else name
            added = ''.join(k for k in 'AGC' if k in other + factor)
            if (added, name) not in pairs:
                pairs.append((added, name))
    for metric_key in ['metrics', 'local_metrics']:
        selected = [dict(r, metrics=r[metric_key]) for r in rows]
        result['comparisons'][metric_key] = [paired_comparison(selected, p, ref) for p,ref in pairs]
    (directory / 'separation_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Propagation cause separation', '',
             'A: correct accepted GT-inconsistent measurements inside propagation.',
             'G: GT-relative propagation geometry, preserving the incoming metric depth.',
             'C: block observed GT-discontinuity edges; unknown edges remain.',
             'All interventions affect r0 only; eight factorial combinations plus injection-only.', '',
             '| Policy | After propagation RMSE mm | Final RMSE mm | Final MAE mm |', '|---|---:|---:|---:|']
    for name,r in result['policies'].items():
        lines.append('| {} | {:.2f} | {:.2f} | {:.2f} |'.format(name,
            1000*r['local_metrics']['rmse_m'],1000*r['metrics']['rmse_m'],1000*r['metrics']['mae_m']))
    lines += ['', 'Inspect paired image-bootstrap comparisons in separation_summary.json.',
              'Compare local and final gains: downstream refinement may preserve or undo them.',
              'GT coverage limits every oracle. G uses GT shape and is a diagnostic, not a deployable fix or guaranteed performance bound.',
              'C uses depth jumps as a surface proxy; no gain does not rule out inappropriate connections.',
              'Factor effects interact and must not be interpreted as additive percentages of total RMSE.',
              'Post-refinement still receives original geometry and sparse observations, including those corrected only inside A.',
              'No retraining or changes to upstream levels. This isolates the immediate full-resolution failure, not all sources of inherited error.',
              'Native baseline uses the configured strength and its own sampled rejection decisions. Share this ZIP.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
