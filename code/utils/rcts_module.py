import math

import torch
import torch.nn.functional as F

from utils.rccr_module import compute_jsd_agreement


REGION_NAMES = ("bg_inner", "bg_boundary", "fg_boundary", "fg_inner")


def compute_normalized_entropy(probabilities, num_classes, eps=1e-6):
    """Return normalized prediction entropy [B, W, H, D] in [0, 1]."""
    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)
    return (entropy / math.log(num_classes)).clamp(min=0.0, max=1.0)


def weak_intensity_perturb(input_tensor, noise_std=0.03, intensity_scale=0.05, intensity_shift=0.05):
    """Apply a weak intensity perturbation used only for teacher-stability routing."""
    patch_std = input_tensor.detach().std().clamp(min=1e-6)
    scale = 1.0 + (torch.rand(1, device=input_tensor.device) * 2.0 - 1.0) * float(intensity_scale)
    shift = (torch.rand(1, device=input_tensor.device) * 2.0 - 1.0) * float(intensity_shift) * patch_std
    noise = torch.randn_like(input_tensor) * float(noise_std) * patch_std
    return input_tensor * scale + shift + noise


def compute_binary_region_map(pseudo_labels, boundary_width=1, spatial_size=None):
    """Build four source-pseudo-label regions independent of the HR class.

    Region ids:
        0 bg_inner, 1 bg_boundary, 2 fg_boundary, 3 fg_inner.
    """
    if pseudo_labels.dim() in (4, 5) and pseudo_labels.shape[1] == 1:
        pseudo_labels = pseudo_labels[:, 0]
    if spatial_size is None:
        spatial_size = pseudo_labels.shape[1:]
    spatial_dims = len(tuple(spatial_size))
    if spatial_dims not in (2, 3):
        raise ValueError("compute_binary_region_map expects 2D or 3D spatial size.")

    if tuple(pseudo_labels.shape[-spatial_dims:]) != tuple(spatial_size):
        pseudo_labels = F.interpolate(
            pseudo_labels.unsqueeze(1).float(),
            size=spatial_size,
            mode="nearest",
        ).squeeze(1).long()
    else:
        pseudo_labels = pseudo_labels.long()

    fg_mask = (pseudo_labels == 1).float().unsqueeze(1)
    boundary_width = max(int(boundary_width), 1)
    kernel_size = 2 * boundary_width + 1
    padding = boundary_width

    if spatial_dims == 2:
        fg_dilated = F.max_pool2d(fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        fg_eroded = 1.0 - F.max_pool2d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
    else:
        fg_dilated = F.max_pool3d(fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        fg_eroded = 1.0 - F.max_pool3d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
    fg_eroded = fg_eroded * fg_mask

    fg_inner = fg_eroded[:, 0].bool()
    fg_boundary = (fg_mask[:, 0] - fg_eroded[:, 0]).clamp(min=0.0, max=1.0).bool()
    bg_boundary = (fg_dilated[:, 0] - fg_mask[:, 0]).clamp(min=0.0, max=1.0).bool()

    region_map = torch.zeros(
        pseudo_labels.shape,
        device=pseudo_labels.device,
        dtype=torch.long,
    )
    region_map[bg_boundary] = 1
    region_map[fg_boundary] = 2
    region_map[fg_inner] = 3
    return region_map


def build_rcts_evidence(
    source_prob,
    target_prob,
    source_hr,
    target_hr,
    source_entropy,
    target_entropy,
    num_classes,
    eps=1e-6,
):
    """Build [relative certainty, HR advantage, agreement] for source->target."""
    agreement = compute_jsd_agreement(source_prob, target_prob, num_classes=num_classes, eps=eps)
    evidence = torch.stack(
        [
            (target_entropy.detach() - source_entropy.detach()).float().clamp(min=-1.0, max=1.0),
            (source_hr.detach() - target_hr.detach()).float().clamp(min=-1.0, max=1.0),
            agreement.detach().float().clamp(min=0.0, max=1.0),
        ],
        dim=-1,
    )
    return evidence, agreement.detach()


def _select_region_value(region_map, inner_value, boundary_value):
    values = torch.full(
        region_map.shape,
        float(inner_value),
        device=region_map.device,
        dtype=torch.float32,
    )
    boundary_mask = (region_map == 1) | (region_map == 2)
    values[boundary_mask] = float(boundary_value)
    return values


def build_fixed_rcts_route_weight(
    wr_type,
    source_prob,
    target_prob,
    source_hr,
    target_hr,
    source_entropy,
    target_entropy,
    source_region_map,
    num_classes,
    entropy_margin_inner=0.0,
    entropy_margin_boundary=0.01,
    agreement_tau_inner=0.0,
    agreement_tau_boundary=0.6,
    hr_margin_inner=0.0,
    hr_margin_boundary=0.0,
    hr_weight=1.0,
    eps=1e-6,
):
    """Build fixed-rule RCTS-v0/v1 route weights.

    v0: original entropy WR plus region-conditioned agreement/advantage
        rejection. It does not use HR advantage.
    v1: uses the same rejection idea, but the teacher score additionally
        includes source-vs-target Region-HR advantage.

    The returned route remains a teacher-selection gate rather than a
    reliability weight. RCCR/HR still decides how much the accepted pseudo-label
    should contribute.
    """
    if wr_type not in ("rcts_v0", "rcts_v1"):
        raise ValueError("Unsupported fixed RCTS type: {}".format(wr_type))

    agreement = compute_jsd_agreement(
        source_prob,
        target_prob,
        num_classes=num_classes,
        eps=eps,
    ).detach()
    entropy_advantage = (target_entropy.detach() - source_entropy.detach()).float()
    hr_advantage = (source_hr.detach() - target_hr.detach()).float()

    entropy_margin = _select_region_value(
        source_region_map,
        entropy_margin_inner,
        entropy_margin_boundary,
    )
    agreement_tau = _select_region_value(
        source_region_map,
        agreement_tau_inner,
        agreement_tau_boundary,
    )

    if wr_type == "rcts_v0":
        teacher_score = entropy_advantage
        score_margin = entropy_margin
    else:
        hr_margin = _select_region_value(
            source_region_map,
            hr_margin_inner,
            hr_margin_boundary,
        )
        teacher_score = entropy_advantage + float(hr_weight) * hr_advantage
        score_margin = entropy_margin + float(hr_weight) * hr_margin

    route_weight = ((teacher_score > score_margin) & (agreement >= agreement_tau)).float()
    stats = {
        "agreement": agreement,
        "entropy_advantage": entropy_advantage,
        "hr_advantage": hr_advantage,
        "teacher_score": teacher_score,
    }
    return route_weight, stats


def build_ps_wr_route_weight(
    source_stability,
    target_stability,
    stability_margin=0.0,
):
    """Build PS-WR teacher routing from branch self-stability.

    The route chooses source->target only when the source branch is more stable
    than the target branch under a weak intensity perturbation. This keeps WR
    as a teacher-selection mechanism and does not use HR/RCCR reliability.
    """
    stability_advantage = (source_stability.detach() - target_stability.detach()).float()
    route_weight = (stability_advantage > float(stability_margin)).float()
    stats = {
        "source_stability": source_stability.detach(),
        "target_stability": target_stability.detach(),
        "stability_advantage": stability_advantage,
        "teacher_score": stability_advantage,
    }
    return route_weight, stats


# Backward-compatible alias for old experiment commands/log notes.
build_dtsr_stability_route_weight = build_ps_wr_route_weight
