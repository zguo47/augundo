'''Geometry supervision experiments; alignment is used only for the loss.

Affine objective follows the supplied MoGe affine_invariant_global_loss with
defaults beta=0, trunc=1, sparsity_aware=False, improve_weighting=False.
Point maps come from the SAME calibrated camera rays for prediction and GT.
The original point alignment solver is retained, with chunked anchor search
and torch 1.10 operations. No GT alignment enters inference/propagation.
'''

import math
import torch
from .geometry_alignment import align


def scale_aligned_l1(src, tgt, mask):
    '''Fit one positive scale per image minimizing valid-pixel absolute error.

    The optimum is the weighted median of G/Z, with weights Z. Select the
    median without gradients, then reconstruct its ratio with gradients. This
    is the exact L1 scale fit (not least squares or median(G)/median(Z)).
    Returns loss, per-image scale. Empty images contribute zero.
    '''
    losses, scales = [], []
    for z, g, valid in zip(src, tgt, mask.bool()):
        z, g = z[valid].clamp_min(1e-7), g[valid]
        if not z.numel():
            losses.append(src.sum() * 0)
            scales.append(src.new_tensor(1.))
            continue
        with torch.no_grad():
            order = (g / z).argsort()
            cumulative = z[order].cumsum(0)
            index = order[torch.searchsorted(cumulative, cumulative[-1] * .5)]
        scale = g[index] / z[index]
        losses.append((scale * z - g).abs().mean())
        scales.append(scale)
    return torch.stack(losses).mean(), torch.stack(scales)


def camera_points(depth, intrinsics):
    '''B x 1 x H x W depth -> B x H x W x 3 calibrated XYZ points.'''
    batch, _, height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                          torch.arange(width, device=depth.device, dtype=depth.dtype))
    pixels = torch.stack([x, y, torch.ones_like(x)]).reshape(1, 3, -1).expand(batch, -1, -1)
    rays = torch.linalg.solve(intrinsics.to(depth), pixels)
    return (rays * depth.flatten(2)).transpose(1, 2).reshape(batch, height, width, 3)


def alignment_samples(points, gt, mask, resolution=64):
    '''Nearest valid source pixel in each output pixel's footprint; no averaging.

    Equivalent local-window nearest selection to mask-aware nearest resize.
    Missing footprints remain missing. Avoid unfolding the whole full-res map.
    '''
    batch, height, width = mask.shape
    fh, fw = max(1., height / resolution), max(1., width / resolution)
    cy = (torch.arange(resolution, device=mask.device) + .5) * height / resolution
    cx = (torch.arange(resolution, device=mask.device) + .5) * width / resolution
    yy, xx = torch.meshgrid(cy, cx)
    dy, dx = torch.meshgrid(torch.arange(math.ceil(fh), device=mask.device),
                           torch.arange(math.ceil(fw), device=mask.device))
    y = (yy - fh / 2).round().long()[..., None] + dy.flatten()
    x = (xx - fw / 2).round().long()[..., None] + dx.flatten()
    inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    index = y.clamp(0, height-1) * width + x.clamp(0, width-1)
    valid = mask.flatten(1)[:, index] & inside
    distance = (y - yy[..., None]).square() + (x - xx[..., None]).square()
    nearest = torch.where(valid, distance, distance.new_tensor(float('inf'))).argmin(-1)
    selected = index.expand(batch, -1, -1, -1).gather(-1, nearest[..., None]).squeeze(-1)
    def gather(value):
        return value.flatten(1, 2).gather(1, selected.flatten(1)[..., None].expand(-1, -1, 3))
    return gather(points), gather(gt), valid.any(-1).flatten(1)


def fit_affine_points(src, tgt, weight, trunc=1.):
    '''Original scale+z-shift anchor enumeration, streamed to bound memory.'''
    n = src.shape[0]
    best_loss, best_anchor, best_coordinate = float('inf'), 0, 0
    with torch.no_grad():
        for anchors in (weight > 0).nonzero().flatten().split(max(1, (2**18) // n)):
            origin_src = torch.zeros(len(anchors), 3, device=src.device, dtype=src.dtype)
            origin_tgt = torch.zeros_like(origin_src)
            origin_src[:, 2], origin_tgt[:, 2] = src[anchors, 2], tgt[anchors, 2]
            _, losses, coordinates = align(
                (src[None] - origin_src[:, None]).flatten(1),
                (tgt[None] - origin_tgt[:, None]).flatten(1),
                weight[None, :, None].expand(len(anchors), -1, 3).flatten(1), trunc)
            winner = losses.argmin()
            if float(losses[winner]) < best_loss:
                best_loss = float(losses[winner])
                best_anchor, best_coordinate = int(anchors[winner]), int(coordinates[winner])
    point, axis = divmod(best_coordinate, 3)
    source_origin = src[best_anchor, 2] if axis == 2 else src.new_zeros(())
    target_origin = tgt[best_anchor, 2] if axis == 2 else tgt.new_zeros(())
    denominator = src[point, axis] - source_origin
    scale = (tgt[point, axis] - target_origin) / torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    shift = tgt[best_anchor, 2] - scale * src[best_anchor, 2]
    return scale, shift


def affine_invariant_loss(src, tgt, mask, intrinsics, resolution=64):
    '''Supplied global affine XYZ objective, including learned z-offset fit.

    Normalize by the full spatial area as in the pasted loss, NOT by valid
    count. Inverse GT-depth weights are capped at 10x their valid mean.
    Alignment rejects nonpositive fitted scales. The loss retains gradients
    through fitted scale and shift; returned diagnostics are detached later.
    '''
    if intrinsics is None:
        raise ValueError('Affine XYZ loss requires camera intrinsics.')
    valid = mask[:, 0].bool()
    safe_gt = torch.where(mask.bool(), tgt, torch.ones_like(tgt))
    points, gt = camera_points(src, intrinsics), camera_points(safe_gt, intrinsics)
    low, low_gt, low_valid = alignment_samples(points, gt, valid, resolution)
    losses, scales, shifts = [], [], []
    for p, g, m, lp, lg, lm in zip(points, gt, valid, low, low_gt, low_valid):
        if not lm.any():
            losses.append(p.sum() * 0)
            scales.append(p.new_zeros(())); shifts.append(p.new_zeros(()))
            continue
        a, b = fit_affine_points(lp, lg, lm.to(p) / lg[:, 2].clamp_min(1e-2))
        active = a > 0
        a, b = torch.where(active, a, torch.zeros_like(a)), torch.where(active, b, torch.zeros_like(b))
        shift = torch.stack([b * 0, b * 0, b])
        weight = (m & active).to(p) / g[..., 2].clamp_min(1e-5)
        weight = weight.clamp_max(10 * (weight * m).sum() / (m.sum() + 1e-7))
        losses.append(((a * p + shift - g).abs() * weight[..., None]).mean())
        scales.append(a); shifts.append(b)
    return torch.stack(losses).mean(), torch.stack(scales), torch.stack(shifts)
