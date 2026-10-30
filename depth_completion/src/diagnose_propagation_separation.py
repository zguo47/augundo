'''Single-level factorial oracles for measurement, geometry and connections.

Only one level's propagation is intervened upon per trial. Upstream predictions,
local learned weights and all keep decisions are held fixed. The geometry
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


def policies(extended=False):
    result = {}
    for a, g, c in itertools.product([False, True], repeat=3):
        name = ''.join(k for k, active in zip('AGC', [a, g, c]) if active) or 'baseline'
        result[name] = dict(kind='separation', measurements=a, geometry=g, connections=c)
    result['injection_only'] = dict(kind='separation', no_mixing=True)
    if extended:
        # No exchange retains injection; complete bypass also removes injection.
        # G must not improve either control just by changing scale coordinates.
        result.update({
            'G_injection_only': dict(geometry=True, no_mixing=True),
            'bypass': dict(bypass=True),
            'G_bypass': dict(geometry=True, bypass=True),
            'G_anchor': dict(geometry=True, geometry_scope='anchor'),
            'G_nonanchor': dict(geometry=True, geometry_scope='nonanchor'),
            'E': dict(exact_anchor=True),
            'AE': dict(measurements=True, exact_anchor=True),
            'matched': dict(matched_graph=True),
            'matched_G': dict(matched_graph=True, geometry=True),
            'matched_C': dict(matched_graph=True, connections=True),
            'matched_GC': dict(matched_graph=True, geometry=True, connections=True)
        })
        for policy in result.values():
            policy['kind'] = 'separation'
    return result


def targets_at_level(target, sparse, size):
    '''Keep grid-location GT separate from the selected measurement's GT.

    At coarse levels sample the nearest original GT pixel to each grid point
    using align-corners coordinates. Never average or fill unknown GT. These
    samples are point observations, NOT dense ground truth for an entire cell.
    A uses the selected measurement's original coordinate instead, so it does
    not falsely call a good foreground measurement a bad background measurement.
    '''
    height, width = size
    y = torch.linspace(0, target.shape[-2] - 1, height, device=target.device).round().long()
    x = torch.linspace(0, target.shape[-1] - 1, width, device=target.device).round().long()
    grid_target = target.index_select(-2, y).index_select(-1, x)
    _, indices = original.sparse_depth_at_resolution(sparse, size, True)
    anchor_target = torch.gather(target.flatten(2), 2, indices.flatten(2)).reshape_as(grid_target)
    return grid_target, anchor_target


def known_depth(target, policy):
    return torch.isfinite(target) & (target > policy.get('min_depth', 0.)) & \
        (target < policy.get('max_depth', float('inf')))


def observable_graph(known):
    batch, _, height, width = known.shape
    neighbors = functional.unfold(known.float(), 3, padding=1).reshape(batch, 9, height, width) > 0
    off_center = torch.ones_like(neighbors)
    off_center[:, 4] = False
    edges = known & neighbors & off_center
    return edges, edges.any(dim=1, keepdim=True)


def intervene(inputs, target, iterations, strength, policy, anchor_target=None):
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
        raise ValueError('GT samples must match the intervened level.')
    known = known_depth(target, policy)
    gt = torch.where(known, target, torch.zeros_like(target))
    anchor_target = target if anchor_target is None else anchor_target
    anchor_known = known_depth(anchor_target, policy)
    anchor_gt = torch.where(anchor_known, anchor_target, torch.zeros_like(anchor_target))
    labels = loss_utils.rejection_supervision_targets(sparse, anchor_gt)
    bad = anchor_known & (labels == 1) & (keep > 0)
    measured = torch.where(bad, anchor_gt, sparse) if policy.get('measurements') else sparse
    eligible_edges, incident = observable_graph(known)
    geometry_mask = known.clone()
    geometry_target = gt
    scope = policy.get('geometry_scope')
    if scope == 'anchor':
        # At r0 this partitions G exactly. At coarse levels it explicitly tests
        # source geometry at the selected measurement, not at the grid point.
        geometry_mask = anchor_known & (keep > 0)
        geometry_target = anchor_gt
    elif scope == 'nonanchor':
        geometry_mask = known & (keep <= 0)
    if policy.get('matched_graph'):
        geometry_mask = geometry_mask & incident
    local_geometry = geometry
    local_scale = scale
    local_ratio = ratio
    if policy.get('geometry') and geometry_mask.any():
        # Align GT to predicted geometry's units. Global scaling cancels when
        # the result is converted back; relative geometry is what changes.
        # Use the SAME alignment for anchor/non-anchor/full G at r0. Different
        # subset alignments would add an artificial scale discontinuity.
        alignment_mask = known if known.any() else geometry_mask
        alignment_gt = gt if known.any() else geometry_target
        alignment = torch.exp((torch.log(alignment_gt[alignment_mask]) -
                               torch.log(geometry[alignment_mask].clamp_min(1e-7))).mean())
        local_geometry = torch.where(geometry_mask, geometry_target / alignment, geometry)
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
                    known_edge_weight_sum=float(weights[eligible & off_center].sum()),
                    geometry_changed_pixels=int(geometry_mask.sum()) if policy.get('geometry') else 0,
                    observable_graph_pixels=int(incident.sum()),
                    known_anchor_gt_pixels=int((anchor_known & (keep > 0)).sum()))
    if policy.get('matched_graph'):
        # All matched policies use identical known-endpoint connectivity, even
        # their control. Preserve self edges. Compare to matched, not baseline.
        allowed = eligible_edges.clone()
        allowed[:, 4] = True
        weights = weights * allowed.to(weights)
    if policy.get('connections'):
        weights = weights * (~jump).to(weights)
    if policy.get('no_mixing'):
        weights = torch.zeros_like(weights)
    if policy.get('bypass'):
        return local_scale * local_geometry / geometry.clamp_min(1e-7), metadata
    local_scale = keep * local_ratio + (1. - keep) * local_scale
    anchor_weight = strength * keep
    incoming_weight = functional.fold(weights.reshape(batch, 9, -1), (height, width), 3, padding=1)
    for _ in range(iterations):
        incoming = functional.fold((weights * local_scale).reshape(batch, 9, -1),
                                   (height, width), 3, padding=1)
        local_scale = (local_scale + incoming + anchor_weight * local_ratio) / \
            (1. + incoming_weight + anchor_weight)
        if policy.get('exact_anchor'):
            # Diagnostic only: enforce anchors after EVERY exchange. Production
            # still uses soft anchors; no hard restoration is added to training.
            local_scale = torch.where(keep > 0, local_ratio, local_scale)
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
        maps = dict(target=target.numpy(), sparse=sparse.cpu().numpy(), evaluated=valid.numpy())
        for level in dict.fromkeys(args.separation_levels):
            values = base.contexts[level]
            inputs = tuple(torch.from_numpy(values[k]) for k in ['scale', 'geometry', 'sparse', 'keep', 'logits', 'ratio'])
            local_target, anchor_target = targets_at_level(target, sparse.cpu(), inputs[0].shape[-2:])
            limits = dict(min_depth=max(0., args.min_evaluate_depth), max_depth=args.max_evaluate_depth)
            local_valid = known_depth(local_target, limits)
            _, incident = observable_graph(local_valid)
            # The same region is scored for every policy. For final outputs,
            # nearest expansion defines a region, not fabricated GT values.
            final_incident = functional.interpolate(incident.float(), size=target.shape[-2:], mode='nearest') > 0
            level_groups = dict(groups, observable_graph=final_incident)
            clean_anchor_gt = torch.where(known_depth(anchor_target, limits), anchor_target, torch.zeros_like(anchor_target))
            local_labels = loss_utils.rejection_supervision_targets(inputs[2], clean_anchor_gt)
            local_groups = dict(away_sparse=inputs[2] <= 0,
                                accepted_good=(inputs[3] > 0) & (local_labels == 0),
                                accepted_bad=(inputs[3] > 0) & (local_labels == 1),
                                gt_over_40m=local_target > 40, observable_graph=incident)
            native_after = next(torch.from_numpy(d) for r, name, d in base.stages
                                if r == level and name == 'after_propagation')
            local_outputs, final_outputs = {}, {}
            for name, policy in policies(args.separation_protocol == 'extended').items():
                policy.update(limits, level=level)
                local_scale, coverage = intervene(inputs, local_target, args.n_propagation_iterations[4-level],
                                                   args.propagation_anchor_strength, policy, anchor_target)
                local_depth = local_scale * inputs[1]
                if name == 'baseline':
                    torch.testing.assert_close(local_depth, native_after, rtol=1e-5, atol=1e-4)
                    prediction, probe = output, base
                else:
                    prediction, probe = forward(wrapper, image, sparse, target.to(wrapper.device), args,
                                                actual_seed, base.frozen, policy)
                    if any(not torch.equal(probe.frozen[r], base.frozen[r]) for r in range(5)):
                        raise RuntimeError('Rejection decisions changed during separation.')
                    # Inputs at the intervened level must be identical; lower
                    # resolutions are upstream and must not have been affected.
                    for key, value in values.items():
                        np.testing.assert_allclose(probe.contexts[level][key], value, rtol=0, atol=0)
                    actual_after = next(torch.from_numpy(d) for r, stage, d in probe.stages
                                        if r == level and stage == 'after_propagation')
                    torch.testing.assert_close(local_depth, actual_after, rtol=1e-4, atol=1e-3)
                # Relative errors against intended accepted measurements expose
                # remaining dilution even where GT is unknown. Stage values are
                # unclamped; exact anchoring need not remain exact after decode.
                intended = torch.where((local_labels == 1) & (inputs[3] > 0), clean_anchor_gt, inputs[2]) \
                    if policy.get('measurements') else inputs[2]
                anchor_mask = (inputs[3] > 0) & (intended > 0)
                emit('separation_result', sample=sample, seed=seed, actual_seed=actual_seed,
                     level=level, policy=name, coverage=coverage,
                     metrics=error_metrics(prediction, target, valid),
                     local_metrics=error_metrics(local_depth, local_target, local_valid),
                     anchor_retention=error_metrics(local_depth, intended, anchor_mask),
                     losses=probe.losses,
                     groups={k: error_metrics(prediction, target, valid & m) for k,m in level_groups.items()},
                     local_groups={k: error_metrics(local_depth, local_target, local_valid & m) for k,m in local_groups.items()},
                     stages=stage_report(probe, target, valid, worst))
                if name in ('baseline', 'injection_only', 'G_injection_only', 'bypass', 'G_bypass'):
                    local_outputs[name], final_outputs[name] = local_depth, prediction
                if save_maps and seed_index == 0:
                    prefix = 'r{}_{}'.format(level, name)
                    maps[prefix + '_final'] = prediction.numpy()
                    maps[prefix + '_after_propagation'] = local_depth.numpy()
                if name != 'baseline':
                    del probe
            if args.separation_protocol == 'extended':
                # Reparameterization without exchange should cancel exactly
                # up to roundoff, including downstream recomputation. Log, do
                # not assert final equality: later heads may amplify roundoff.
                for name, reference in [('G_injection_only', 'injection_only'), ('G_bypass', 'bypass')]:
                    emit('separation_null_control', sample=sample, seed=seed, level=level,
                         policy=name, reference=reference,
                         local_max_difference_m=float((local_outputs[name] - local_outputs[reference]).abs().max()),
                         final_max_difference_m=float((final_outputs[name] - final_outputs[reference]).abs().max()))
            if save_maps and seed_index == 0:
                maps.update({'baseline_r{}_{}'.format(level, k): v for k,v in values.items()})
                maps['r{}_grid_gt'.format(level)] = local_target.numpy()
                maps['r{}_anchor_gt'.format(level)] = anchor_target.numpy()
                maps['r{}_observable_graph'.format(level)] = incident.numpy()
            print('separation sample {}, seed {}, r{} complete'.format(sample, seed, level), flush=True)
        if save_maps and seed_index == 0:
            np.savez_compressed(str(directory / 'separation_sample_{}_seed_{}.npz'.format(sample, seed)), **maps)


def summarize_level(rows):
    names = list(dict.fromkeys(r['policy'] for r in rows))
    result = dict(policies={}, comparisons={})
    for name in names:
        selected = [r for r in rows if r['policy'] == name]
        result['policies'][name] = dict(
            metrics=pool_metrics([r['metrics'] for r in selected]),
            local_metrics=pool_metrics([r['local_metrics'] for r in selected]),
            groups={k: pool_metrics([r['groups'][k] for r in selected]) for k in selected[0]['groups']},
            local_groups={k: pool_metrics([r['local_groups'][k] for r in selected]) for k in selected[0]['local_groups']},
            anchor_retention=pool_metrics([r['anchor_retention'] for r in selected]),
            by_seed={str(seed): pool_metrics([r['metrics'] for r in selected if r['seed'] == seed])
                     for seed in sorted({r['seed'] for r in selected})},
            coverage_totals={k: sum(r['coverage'][k] for r in selected) for k in selected[0]['coverage']})
    # Compare each factor both alone and in the presence of the other factors.
    pairs = [(n, 'baseline') for n in names if n != 'baseline']
    for factor in 'AGC':
        for name in policies():
            if name == 'injection_only' or factor in name:
                continue
            other = '' if name == 'baseline' else name
            added = ''.join(k for k in 'AGC' if k in other + factor)
            if (added, name) not in pairs:
                pairs.append((added, name))
    pairs += [('G_injection_only', 'injection_only'), ('G_bypass', 'bypass'),
              ('G', 'G_injection_only'), ('baseline', 'injection_only'),
              ('G_anchor', 'G'), ('G_nonanchor', 'G'), ('AE', 'A'), ('AE', 'E'),
              ('matched_G', 'matched'), ('matched_C', 'matched'),
              ('matched_GC', 'matched_G'), ('matched_GC', 'matched_C')]
    pairs = list(dict.fromkeys((p, ref) for p, ref in pairs if p in names and ref in names))
    for metric_key in ['metrics', 'local_metrics', 'matched_final', 'matched_local']:
        selected = [dict(r, metrics=r['groups']['observable_graph'] if metric_key == 'matched_final' else
                        r['local_groups']['observable_graph'] if metric_key == 'matched_local' else r[metric_key])
                    for r in rows]
        result['comparisons'][metric_key] = [paired_comparison(selected, p, ref) for p,ref in pairs]
    return result


def write_summary(directory, records):
    rows = [r for r in records if r['type'] == 'separation_result']
    result = dict(scope='One level per intervention; all rejection decisions frozen; native-strength baseline',
                  levels={})
    for level in sorted({r['level'] for r in rows}):
        result['levels']['r{}'.format(level)] = summarize_level([r for r in rows if r['level'] == level])
    # Retain the previous r0 summary entry points for existing analysis scripts.
    if 'r0' in result['levels']:
        result.update(result['levels']['r0'])
    result['null_controls'] = [r for r in records if r['type'] == 'separation_null_control']
    (directory / 'separation_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Propagation cause separation', '',
             'A: correct accepted GT-inconsistent measurements inside propagation.',
             'G: GT-relative propagation geometry, preserving the incoming metric depth.',
             'C: block observed GT-discontinuity edges; unknown edges remain.',
             'E: exact accepted anchors after every exchange (diagnostic only).',
             'G_anchor / G_nonanchor: correct only accepted-source / other geometry.',
             'G_injection_only and G_bypass: geometry without exchange / without exchange or injection.',
             'matched*: same known-endpoint graph for control, G and C; compare to matched.',
             'One level changes per trial; finer downstream stages recompute, gates remain frozen.', '']
    def number(value):
        return 'unavailable' if value is None else '{:.2f}'.format(1000 * value)
    for level, summary in result['levels'].items():
        lines += ['## Intervention at {}'.format(level), '',
                  '| Policy | Local sampled-GT RMSE mm | Final RMSE mm | Final MAE mm | Anchor RMSE mm |',
                  '|---|---:|---:|---:|---:|']
        for name,r in summary['policies'].items():
            lines.append('| {} | {} | {} | {} | {} |'.format(name,
                number(r['local_metrics'].get('rmse_m')), number(r['metrics'].get('rmse_m')),
                number(r['metrics'].get('mae_m')), number(r['anchor_retention'].get('rmse_m'))))
        lines.append('')
    lines += ['', 'Inspect paired image-bootstrap comparisons in separation_summary.json.',
              'Compare local and final gains: downstream refinement may preserve or undo them.',
              'GT coverage limits every oracle. G uses GT shape and is a diagnostic, not a deployable fix or guaranteed performance bound.',
              'C uses depth jumps as a surface proxy; no gain does not rule out inappropriate connections.',
              'Factor effects interact and must not be interpreted as additive percentages of total RMSE.',
              'Post-refinement still receives original geometry and sparse observations, including those corrected only inside A.',
              'Coarse GT is sampled at grid points without filling/averaging. Measurement correction uses GT at the selected ORIGINAL sparse coordinate.',
              'Coarse G_anchor uses that measurement coordinate; full G uses grid GT. They are not an exact source/destination partition at coarse levels.',
              'Coarse cells may span multiple surfaces. These sparse point oracles do not provide dense or surface-resolved coarse GT.',
              'Matched tests restrict all exchange to known-endpoint edges (plus self). Read observable_graph metrics and compare to matched, not only baseline.',
              'A zero/null control checks reparameterization itself. A large final difference despite tiny local difference indicates downstream sensitivity.',
              'Compare E to baseline and AE to A: remaining dilution vs preserving wrong measurements. Exactness applies immediately after propagation, not final refinement.',
              'Repeat the SAME indices/seeds/protocol on a later checkpoint; each ZIP records the checkpoint hash and step.',
              'No retraining or changes to upstream levels. Single-level effects do not establish the full hierarchy joint effect.',
              'Native baseline uses the configured strength and its own sampled rejection decisions. Share this ZIP.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
