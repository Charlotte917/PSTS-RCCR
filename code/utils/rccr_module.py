import math

import torch
import torch.nn as nn


def compute_jsd_agreement(prob_a, prob_b, num_classes, eps=1e-6):
    """Compute normalized JSD agreement between two probability maps.

    Args:
        prob_a: Tensor [B, C, W, H, D].
        prob_b: Tensor [B, C, W, H, D].
        num_classes: Number of classes C.

    Returns:
        Tensor [B, W, H, D] in [0, 1], where larger means more agreement.
    """
    mean_prob = 0.5 * (prob_a + prob_b)
    kl_a = (prob_a * (torch.log(prob_a + eps) - torch.log(mean_prob + eps))).sum(dim=1)
    kl_b = (prob_b * (torch.log(prob_b + eps) - torch.log(mean_prob + eps))).sum(dim=1)
    jsd = 0.5 * (kl_a + kl_b)
    agreement = 1.0 - jsd / math.log(num_classes)
    return agreement.clamp(min=0.0, max=1.0)


def build_rccr_weight(
    rccr_type,
    source_prob,
    target_prob,
    num_classes,
    agreement_floor=0.0,
    agreement_gamma=1.0,
):
    """Build the extra RCCR weighting term for the current pseudo-label path.

    B2 uses only cross-view agreement:
        weight = floor + (1 - floor) * agreement ** gamma

    The returned tensor should be detached by the caller before it affects loss.
    """
    if rccr_type == "none":
        return None
    if rccr_type != "agreement":
        raise ValueError("Unsupported rccr_type: {}".format(rccr_type))

    agreement = compute_jsd_agreement(source_prob, target_prob, num_classes=num_classes)
    if agreement_gamma != 1.0:
        agreement = agreement.clamp(min=0.0, max=1.0).pow(agreement_gamma)
    agreement_floor = float(agreement_floor)
    if agreement_floor < 0.0 or agreement_floor > 1.0:
        raise ValueError("agreement_floor must be in [0, 1]. Got {}.".format(agreement_floor))
    return agreement_floor + (1.0 - agreement_floor) * agreement


def summarize_weight(weight):
    if weight is None or weight.numel() == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    values = weight.detach().float()
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "std": float(values.std(unbiased=False).item()),
    }


class GlobalRCCRCalibrator(nn.Module):
    """B3 global lightweight calibrator.

    The linear layer is zero-initialized. For B3-v2, its raw output is used as
    a residual delta in B2 logit space:
        W = sigmoid(logit(h * agreement) + delta)

    The input features must be detached before being passed here when the
    calibrator is trained from correctness labels.
    """

    def __init__(self, input_dim=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward_delta(self, evidence):
        return self.linear(evidence).squeeze(-1)

    def forward(self, evidence):
        return torch.sigmoid(self.forward_delta(evidence))


class RegionConditionedRCCRCalibrator(nn.Module):
    """B4 region-conditioned residual calibrator.

    B3-v2 uses one global linear residual head. B4 keeps the same logit-space
    residual formula, but routes each voxel to one of four region-specific
    heads according to the source pseudo-label region:
        0 bg_inner, 1 bg_boundary, 2 fg_boundary, 3 fg_inner.

    All heads are zero-initialized, so B4 exactly starts from the B2 base
    weight before the residual heads learn anything.
    """

    REGION_NAMES = ("bg_inner", "bg_boundary", "fg_boundary", "fg_inner")

    def __init__(self, input_dim=3, num_regions=4):
        super().__init__()
        if num_regions != len(self.REGION_NAMES):
            raise ValueError("RegionConditionedRCCRCalibrator currently expects 4 regions.")
        self.input_dim = input_dim
        self.num_regions = num_regions
        self.heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_regions)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_delta(self, evidence, region_ids):
        original_shape = evidence.shape[:-1]
        evidence_flat = evidence.reshape(-1, evidence.shape[-1])
        region_flat = region_ids.reshape(-1).long()
        if evidence_flat.shape[0] != region_flat.shape[0]:
            raise ValueError(
                "Evidence and region_ids must have the same number of voxels. "
                "Got {} and {}.".format(evidence_flat.shape[0], region_flat.shape[0])
            )
        region_flat = region_flat.clamp(min=0, max=self.num_regions - 1)
        weight = torch.stack([head.weight.squeeze(0) for head in self.heads], dim=0)
        bias = torch.stack([head.bias.squeeze(0) for head in self.heads], dim=0)
        selected_weight = weight.index_select(0, region_flat)
        selected_bias = bias.index_select(0, region_flat)
        delta = (evidence_flat * selected_weight).sum(dim=-1) + selected_bias
        return delta.view(*original_shape)

    def forward(self, evidence, region_ids):
        return torch.sigmoid(self.forward_delta(evidence, region_ids))

    def head_parameters(self):
        params = []
        for region_id, head in enumerate(self.heads):
            params.append({
                "region_id": region_id,
                "region_name": self.REGION_NAMES[region_id],
                "weight": head.weight.detach().view(-1).clone(),
                "bias": head.bias.detach()[0].clone(),
            })
        return params


def safe_logit(weight, eps=1e-4):
    weight = weight.clamp(min=eps, max=1.0 - eps)
    return torch.log(weight) - torch.log1p(-weight)


def apply_logit_residual(base_weight, delta, residual_lambda=1.0, eps=1e-4):
    """Apply B3-v2 residual calibration around the B2 base weight.

    When delta is zero or residual_lambda is zero, the result is exactly the
    clipped B2 base weight.
    """
    return torch.sigmoid(safe_logit(base_weight, eps=eps) + residual_lambda * delta)


def compute_region_relative_delta(raw_score, region_ids, max_delta=0.5, num_regions=4):
    """Convert raw region-head scores into conservative relative residuals.

    B4-RC should not learn a constant negative offset for an entire region.
    This function centers scores inside each sample and source-pseudo-label
    region, then bounds the applied residual with tanh:
        delta = max_delta * tanh(raw_score - stopgrad(region_mean)).

    Args:
        raw_score: Tensor shaped [B, W, H, D] or [N].
        region_ids: Tensor with the same shape, values in 0..num_regions-1.
        max_delta: Absolute bound for the residual applied in logit space.
        num_regions: Number of region ids.

    Returns:
        relative_score: Centered raw score.
        applied_delta: Bounded residual in [-max_delta, max_delta].
    """
    if tuple(raw_score.shape) != tuple(region_ids.shape):
        raise ValueError(
            "raw_score and region_ids must have the same shape. Got {} and {}.".format(
                tuple(raw_score.shape),
                tuple(region_ids.shape),
            )
        )

    max_delta = float(max_delta)
    if max_delta < 0.0:
        raise ValueError("max_delta must be non-negative. Got {}.".format(max_delta))

    region_ids = region_ids.long()
    relative_score = raw_score.clone()

    if raw_score.dim() <= 1:
        for region_id in range(num_regions):
            mask = region_ids == region_id
            if bool(mask.any().item()):
                region_mean = raw_score[mask].mean().detach()
                relative_score[mask] = raw_score[mask] - region_mean
    else:
        batch_size = raw_score.shape[0]
        for batch_idx in range(batch_size):
            for region_id in range(num_regions):
                mask = region_ids[batch_idx] == region_id
                if bool(mask.any().item()):
                    region_mean = raw_score[batch_idx][mask].mean().detach()
                    relative_score[batch_idx][mask] = raw_score[batch_idx][mask] - region_mean

    applied_delta = max_delta * torch.tanh(relative_score)
    return relative_score, applied_delta


def compute_region_selective_suppress_delta(
    raw_score,
    region_ids,
    suppress_scale=0.3,
    tail_threshold=0.5,
    min_voxels=16,
    eps=1e-6,
    num_regions=4,
):
    """Build B4-RC2 one-sided, region-relative suppressive residuals.

    The raw region head still learns a correctness logit. For applying the
    unsupervised weight, B4-RC2 standardizes scores inside each sample and
    source region, then only suppresses the low-score tail:
        z = (score - stopgrad(region_mean)) / (stopgrad(region_std) + eps)
        delta = -scale * tanh(relu(-(z + tail_threshold))).

    Therefore delta is always in [-scale, 0], so the final RCCR weight can
    never exceed the B2 base weight when used in logit space.
    """
    if tuple(raw_score.shape) != tuple(region_ids.shape):
        raise ValueError(
            "raw_score and region_ids must have the same shape. Got {} and {}.".format(
                tuple(raw_score.shape),
                tuple(region_ids.shape),
            )
        )

    suppress_scale = float(suppress_scale)
    if suppress_scale < 0.0:
        raise ValueError("suppress_scale must be non-negative. Got {}.".format(suppress_scale))
    min_voxels = max(int(min_voxels), 1)

    region_ids = region_ids.long()
    standardized_score = raw_score.new_zeros(raw_score.shape)
    applied_delta = raw_score.new_zeros(raw_score.shape)

    def fill_for_mask(mask):
        if int(mask.sum().item()) < min_voxels:
            return None, None
        values = raw_score[mask]
        region_std = values.std(unbiased=False).detach()
        if float(region_std.item()) <= float(eps):
            return None, None
        region_mean = values.mean().detach()
        z_score = (values - region_mean) / (region_std + eps)
        low_score_strength = torch.relu(-(z_score + float(tail_threshold)))
        delta = -suppress_scale * torch.tanh(low_score_strength)
        return z_score, delta

    if raw_score.dim() <= 1:
        for region_id in range(num_regions):
            mask = region_ids == region_id
            z_score, delta = fill_for_mask(mask)
            if z_score is not None:
                standardized_score[mask] = z_score
                applied_delta[mask] = delta
    else:
        batch_size = raw_score.shape[0]
        for batch_idx in range(batch_size):
            sample_scores = raw_score[batch_idx]
            sample_regions = region_ids[batch_idx]
            for region_id in range(num_regions):
                mask = sample_regions == region_id
                if int(mask.sum().item()) < min_voxels:
                    continue
                values = sample_scores[mask]
                region_std = values.std(unbiased=False).detach()
                if float(region_std.item()) <= float(eps):
                    continue
                region_mean = values.mean().detach()
                z_score = (values - region_mean) / (region_std + eps)
                low_score_strength = torch.relu(-(z_score + float(tail_threshold)))
                delta = -suppress_scale * torch.tanh(low_score_strength)
                standardized_score[batch_idx][mask] = z_score
                applied_delta[batch_idx][mask] = delta

    return standardized_score, applied_delta


def calibration_lambda(iter_num, start_iter=500, rampup_iter=500):
    """Linear transition from B2 reliability to learned B3 reliability."""
    if iter_num <= start_iter:
        return 0.0
    if rampup_iter <= 0:
        return 1.0
    if iter_num >= start_iter + rampup_iter:
        return 1.0
    return float(iter_num - start_iter) / float(rampup_iter)


def build_rccr_evidence(source_prob, target_prob, hr_weight, num_classes, eps=1e-6):
    """Build [h, agreement, 1 - entropy] evidence for one source branch."""
    agreement = compute_jsd_agreement(source_prob, target_prob, num_classes=num_classes, eps=eps)
    entropy = -(source_prob * torch.log(source_prob + eps)).sum(dim=1) / math.log(num_classes)
    uncertainty_reliability = (1.0 - entropy).clamp(min=0.0, max=1.0)
    evidence = torch.stack([
        hr_weight.detach().float().clamp(min=0.0, max=1.0),
        agreement.detach().float().clamp(min=0.0, max=1.0),
        uncertainty_reliability.detach().float().clamp(min=0.0, max=1.0),
    ], dim=-1)
    return evidence, agreement.detach(), uncertainty_reliability.detach()


def balanced_calibration_batch(
    evidence,
    pseudo_labels,
    gt_labels,
    region_map,
    base_weight=None,
    samples_per_group=128,
):
    """Sample a balanced labeled batch over region x correctness groups.

    Args:
        evidence: [B, W, H, D, 3].
        pseudo_labels: [B, W, H, D].
        gt_labels: [B, W, H, D].
        region_map: [B, W, H, D] with ids 0..3.
        base_weight: Optional [B, W, H, D] B2 weight sampled with evidence.

    Returns:
        features: [N, 3], targets: [N], sample_weights: [N],
        base_weights: [N] or None, region_ids: [N], stats: dict.
    """
    correctness = (pseudo_labels == gt_labels).long()
    feature_chunks = []
    target_chunks = []
    weight_chunks = []
    base_weight_chunks = []
    region_id_chunks = []
    stats = {
        "sampled": 0,
        "correct": 0,
        "wrong": 0,
        "groups": 0,
        "candidate": 0,
    }

    samples_per_group = max(int(samples_per_group), 1)
    for region_id in range(4):
        region_mask = region_map == region_id
        for correct_value in (0, 1):
            group_mask = region_mask & (correctness == correct_value)
            indices = torch.nonzero(group_mask, as_tuple=False)
            if indices.numel() == 0:
                continue

            total = indices.shape[0]
            sample_count = min(total, samples_per_group)
            if total > sample_count:
                selected = torch.randperm(total, device=indices.device)[:sample_count]
                indices = indices[selected]

            index_tuple = tuple(indices.t())
            sampled_features = evidence[index_tuple]
            sampled_targets = torch.full(
                (indices.shape[0],),
                float(correct_value),
                device=evidence.device,
                dtype=torch.float32,
            )
            sampled_weights = torch.full(
                (indices.shape[0],),
                float(total) / float(max(sample_count, 1)),
                device=evidence.device,
                dtype=torch.float32,
            )
            feature_chunks.append(sampled_features)
            target_chunks.append(sampled_targets)
            weight_chunks.append(sampled_weights)
            if base_weight is not None:
                base_weight_chunks.append(base_weight[index_tuple])
            region_id_chunks.append(torch.full(
                (indices.shape[0],),
                region_id,
                device=evidence.device,
                dtype=torch.long,
            ))
            stats["candidate"] += int(total)
            stats["sampled"] += int(indices.shape[0])
            stats["correct"] += int(indices.shape[0]) if correct_value == 1 else 0
            stats["wrong"] += int(indices.shape[0]) if correct_value == 0 else 0
            stats["groups"] += 1

    if not feature_chunks:
        return None, None, None, None, None, stats
    sampled_base_weights = torch.cat(base_weight_chunks, dim=0) if base_weight_chunks else None
    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        torch.cat(weight_chunks, dim=0),
        sampled_base_weights,
        torch.cat(region_id_chunks, dim=0),
        stats,
    )
