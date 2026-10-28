import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalHR(nn.Module):
    """Original HR reliability weighting from the baseline paper."""

    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, features, probabilities, pseudo_labels, branch_id=None):
        batch_o, c_o = probabilities.shape[:2]
        batch_f, c_f = features.shape[:2]
        feature_spatial = features.shape[2:]
        prob_spatial = probabilities.shape[2:]

        if pseudo_labels.dim() == features.dim() - 1:
            pseudo_labels = pseudo_labels.unsqueeze(1)

        teacher_f = features.reshape(batch_f, c_f, -1)
        prototype_bank = torch.zeros(batch_f, self.num_classes, c_f, device=features.device)
        spatial_reduce_dims = tuple(range(1, features[0].dim()))

        for ba in range(batch_f):
            for n_class in range(self.num_classes):
                mask_temp = (pseudo_labels[ba] == n_class).float()
                top_fea = features[ba] * mask_temp
                prototype_bank[ba, n_class] = top_fea.sum(dim=spatial_reduce_dims) / (mask_temp.sum() + self.eps)

        prototype_bank = F.normalize(prototype_bank, dim=-1)
        prototype_similarity = torch.zeros(
            batch_f,
            self.num_classes,
            *feature_spatial,
            device=features.device,
            dtype=probabilities.dtype,
        )
        for ba in range(batch_o):
            for n_class in range(self.num_classes):
                class_prototype = prototype_bank[ba, n_class]
                prototype_similarity[ba, n_class] = F.cosine_similarity(
                    teacher_f[ba],
                    class_prototype.unsqueeze(1),
                    dim=0,
                ).view(*feature_spatial)

        if tuple(prototype_similarity.shape[2:]) != tuple(prob_spatial):
            mode = "bilinear" if len(prob_spatial) == 2 else "trilinear"
            prototype_similarity = F.interpolate(
                prototype_similarity,
                size=prob_spatial,
                mode=mode,
                align_corners=False,
            )

        return (1 - F.mse_loss(prototype_similarity, probabilities, reduction="none")).mean(1)

    def update_memory(self):
        pass


class RegionHR(OriginalHR):
    """Region-conditioned HR with interior and boundary prototypes."""

    REGION_NAMES = ("bg_inner", "bg_boundary", "fg_boundary", "fg_inner")

    def __init__(self, num_classes, boundary_width=1, min_voxels=16, eps=1e-6):
        super().__init__(num_classes=num_classes, eps=eps)
        self.boundary_width = boundary_width
        self.min_voxels = min_voxels

    def _match_label_size(self, pseudo_labels, spatial_size):
        spatial_dims = len(tuple(spatial_size))
        if pseudo_labels.dim() == spatial_dims + 1:
            pseudo_labels = pseudo_labels.unsqueeze(1)
        if tuple(pseudo_labels.shape[-spatial_dims:]) == tuple(spatial_size):
            return pseudo_labels.long()
        return F.interpolate(pseudo_labels.float(), size=spatial_size, mode="nearest").long()

    def _binary_region_masks(self, pseudo_labels):
        fg_mask = (pseudo_labels == 1).float()
        kernel_size = 2 * self.boundary_width + 1
        padding = self.boundary_width

        spatial_dims = pseudo_labels.dim() - 2
        if spatial_dims == 2:
            fg_dilated = F.max_pool2d(fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
            fg_eroded = 1.0 - F.max_pool2d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        elif spatial_dims == 3:
            fg_dilated = F.max_pool3d(fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
            fg_eroded = 1.0 - F.max_pool3d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        else:
            raise ValueError("RegionHR expects 2D or 3D pseudo labels.")
        fg_eroded = fg_eroded * fg_mask

        fg_inner = fg_eroded
        fg_boundary = (fg_mask - fg_inner).clamp(min=0.0, max=1.0)
        bg_boundary = (fg_dilated - fg_mask).clamp(min=0.0, max=1.0)
        bg_inner = (1.0 - fg_dilated).clamp(min=0.0, max=1.0)

        class_region_masks = torch.zeros(
            pseudo_labels.shape[0],
            self.num_classes,
            2,
            *pseudo_labels.shape[2:],
            device=pseudo_labels.device,
            dtype=torch.float32,
        )
        class_region_masks[:, 0, 0] = bg_inner[:, 0]
        class_region_masks[:, 0, 1] = bg_boundary[:, 0]
        class_region_masks[:, 1, 0] = fg_inner[:, 0]
        class_region_masks[:, 1, 1] = fg_boundary[:, 0]

        boundary_selector = ((fg_boundary + bg_boundary) > 0).float()
        return class_region_masks, boundary_selector

    def compute_region_map(self, pseudo_labels, spatial_size=None):
        """Return source-pseudo-label region ids for RCCR diagnostics.

        Region ids are:
            0: bg_inner
            1: bg_boundary
            2: fg_boundary
            3: fg_inner
        """
        if self.num_classes != 2:
            raise ValueError("compute_region_map currently supports binary segmentation only.")

        if spatial_size is None:
            if pseudo_labels.dim() in (4, 5) and pseudo_labels.shape[1] == 1:
                spatial_size = pseudo_labels.shape[2:]
            else:
                spatial_size = pseudo_labels.shape[1:]

        pseudo_labels = self._match_label_size(pseudo_labels, spatial_size)
        class_region_masks, _ = self._binary_region_masks(pseudo_labels)

        region_map = torch.zeros(
            pseudo_labels.shape[0],
            *pseudo_labels.shape[2:],
            device=pseudo_labels.device,
            dtype=torch.long,
        )
        region_map[class_region_masks[:, 0, 1].bool()] = 1
        region_map[class_region_masks[:, 1, 1].bool()] = 2
        region_map[class_region_masks[:, 1, 0].bool()] = 3
        return region_map

    def compute_global_class_prototypes(self, features, pseudo_labels):
        batch_size, feat_channels = features.shape[:2]
        features_flat = features.reshape(batch_size, feat_channels, -1)
        labels_flat = pseudo_labels[:, 0].reshape(batch_size, -1).long()
        class_masks = F.one_hot(labels_flat.clamp(min=0, max=self.num_classes - 1), self.num_classes)
        class_masks = class_masks.permute(0, 2, 1).to(dtype=features.dtype)

        counts = class_masks.sum(dim=-1)
        prototype_sum = torch.einsum("bfn,bcn->bcf", features_flat, class_masks)
        global_prototypes = prototype_sum / counts.clamp_min(self.eps).unsqueeze(-1)

        sample_mean = features_flat.mean(dim=-1).unsqueeze(1)
        valid = counts >= self.min_voxels
        global_prototypes = torch.where(valid.unsqueeze(-1), global_prototypes, sample_mean)
        return F.normalize(global_prototypes, dim=-1)

    def compute_region_prototypes(self, features, pseudo_labels, class_region_masks):
        batch_size, feat_channels = features.shape[:2]
        global_prototypes = self.compute_global_class_prototypes(features, pseudo_labels)
        features_flat = features.reshape(batch_size, feat_channels, -1)
        region_masks = class_region_masks.reshape(batch_size, self.num_classes, 2, -1).to(dtype=features.dtype)
        counts = region_masks.sum(dim=-1)

        prototype_sum = torch.einsum("bfn,bcrn->bcrf", features_flat, region_masks)
        prototypes = prototype_sum / counts.clamp_min(self.eps).unsqueeze(-1)
        valid_mask = counts >= self.min_voxels
        prototypes = torch.where(valid_mask.unsqueeze(-1), prototypes, global_prototypes.unsqueeze(2))

        return F.normalize(prototypes, dim=-1), valid_mask, global_prototypes

    def compute_hr_from_prototypes(self, features, probabilities, prototypes, boundary_selector):
        batch_f, c_f = features.shape[:2]
        feature_spatial = features.shape[2:]
        teacher_f = F.normalize(features.reshape(batch_f, c_f, -1), dim=1)
        boundary_flat = boundary_selector.reshape(batch_f, -1)
        prototypes = F.normalize(prototypes, dim=-1)
        prototype_similarity_all = torch.einsum("bfn,bcrf->bcrn", teacher_f, prototypes)
        prototype_similarity = (
            prototype_similarity_all[:, :, 0] * (1.0 - boundary_flat).unsqueeze(1)
            + prototype_similarity_all[:, :, 1] * boundary_flat.unsqueeze(1)
        ).reshape(batch_f, self.num_classes, *feature_spatial)
        prototype_similarity = prototype_similarity.to(dtype=probabilities.dtype)

        if tuple(prototype_similarity.shape[2:]) != tuple(probabilities.shape[2:]):
            mode = "bilinear" if probabilities.dim() == 4 else "trilinear"
            prototype_similarity = F.interpolate(
                prototype_similarity,
                size=probabilities.shape[2:],
                mode=mode,
                align_corners=False,
            )

        return (1 - F.mse_loss(prototype_similarity, probabilities, reduction="none")).mean(1)

    def forward(self, features, probabilities, pseudo_labels, branch_id=None):
        if self.num_classes != 2:
            return super().forward(features, probabilities, pseudo_labels, branch_id=branch_id)

        pseudo_labels = self._match_label_size(pseudo_labels, features.shape[2:])
        class_region_masks, boundary_selector = self._binary_region_masks(pseudo_labels)
        prototypes, valid_mask, global_prototypes = self.compute_region_prototypes(
            features,
            pseudo_labels,
            class_region_masks,
        )
        return self.compute_hr_from_prototypes(features, probabilities, prototypes, boundary_selector)


class RegionMemoryHR(RegionHR):
    """E2 Region-Memory-HR with independent branch memories.

    It adds fixed current/history prototype fusion and EMA memory update on top
    of RegionHR. No reliability gate or adaptive fusion is used in E2.
    """

    def __init__(
        self,
        num_classes,
        boundary_width=1,
        min_voxels=16,
        memory_weight=0.2,
        memory_momentum=0.1,
        num_branches=2,
        eps=1e-6,
    ):
        super().__init__(num_classes=num_classes, boundary_width=boundary_width, min_voxels=min_voxels, eps=eps)
        self.memory_weight = memory_weight
        self.memory_momentum = memory_momentum
        self.num_branches = num_branches
        self.register_buffer("memory_prototypes", torch.empty(0))
        self.register_buffer("memory_valid", torch.empty(0, dtype=torch.bool))
        self._pending_updates = []

    def _ensure_memory(self, feat_channels, device, dtype):
        expected_shape = (self.num_branches, self.num_classes, 2, feat_channels)
        if tuple(self.memory_prototypes.shape) != expected_shape:
            self.memory_prototypes = torch.zeros(expected_shape, device=device, dtype=dtype)
            self.memory_valid = torch.zeros(
                self.num_branches,
                self.num_classes,
                2,
                device=device,
                dtype=torch.bool,
            )
        elif self.memory_prototypes.device != device or self.memory_prototypes.dtype != dtype:
            self.memory_prototypes = self.memory_prototypes.to(device=device, dtype=dtype)
            self.memory_valid = self.memory_valid.to(device=device)

    def build_effective_prototypes(self, current_prototypes, valid_mask, branch_id):
        memory = self.memory_prototypes[branch_id].unsqueeze(0)
        memory_valid = self.memory_valid[branch_id].unsqueeze(0)
        use_memory = valid_mask & memory_valid
        current_prototypes = F.normalize(current_prototypes, p=2, dim=-1)
        memory = F.normalize(memory, p=2, dim=-1)
        fused = F.normalize(
            self.memory_weight * memory + (1.0 - self.memory_weight) * current_prototypes,
            p=2,
            dim=-1,
        )
        return torch.where(use_memory.unsqueeze(-1), fused, current_prototypes)

    def _store_pending_update(self, current_prototypes, valid_mask, branch_id):
        with torch.no_grad():
            self._pending_updates.append((
                int(branch_id),
                current_prototypes.detach(),
                valid_mask.detach(),
            ))

    def update_memory(self):
        if not self._pending_updates or self.memory_prototypes.numel() == 0:
            self._pending_updates = []
            return

        with torch.no_grad():
            for branch_id, current_prototypes, valid_mask in self._pending_updates:
                for n_class in range(self.num_classes):
                    for region_idx in range(2):
                        valid_samples = valid_mask[:, n_class, region_idx]
                        if not bool(valid_samples.any()):
                            continue

                        current_proto = current_prototypes[valid_samples, n_class, region_idx].mean(dim=0)
                        current_proto = F.normalize(current_proto, dim=0)

                        if not bool(self.memory_valid[branch_id, n_class, region_idx]):
                            self.memory_prototypes[branch_id, n_class, region_idx] = current_proto
                            self.memory_valid[branch_id, n_class, region_idx] = True
                        else:
                            previous = self.memory_prototypes[branch_id, n_class, region_idx]
                            updated = (1.0 - self.memory_momentum) * previous + self.memory_momentum * current_proto
                            self.memory_prototypes[branch_id, n_class, region_idx] = F.normalize(updated, dim=0)

        self._pending_updates = []

    def forward(self, features, probabilities, pseudo_labels, branch_id=None):
        if self.num_classes != 2:
            return super().forward(features, probabilities, pseudo_labels, branch_id=branch_id)
        if branch_id is None:
            raise ValueError("RegionMemoryHR requires branch_id to keep decoder memories independent.")
        if branch_id < 0 or branch_id >= self.num_branches:
            raise ValueError("branch_id must be in [0, {}). Got {}.".format(self.num_branches, branch_id))

        batch_f, c_f = features.shape[:2]
        self._ensure_memory(c_f, features.device, features.dtype)

        pseudo_labels = self._match_label_size(pseudo_labels, features.shape[-3:])
        class_region_masks, boundary_selector = self._binary_region_masks(pseudo_labels)
        current_prototypes, valid_mask, global_prototypes = self.compute_region_prototypes(
            features,
            pseudo_labels,
            class_region_masks,
        )
        effective_prototypes = self.build_effective_prototypes(current_prototypes, valid_mask, branch_id)
        hr_weight = self.compute_hr_from_prototypes(features, probabilities, effective_prototypes, boundary_selector)
        self._store_pending_update(current_prototypes, valid_mask, branch_id)
        return hr_weight



class GatedRegionMemoryHR(RegionMemoryHR):
    """E3 = E2 + gated memory update.

    Current HR computation and fixed current/history prototype fusion are
    unchanged from E2. The only added mechanism is the agreement-confidence
    gate that decides which current region prototypes can update memory.
    """

    def __init__(
        self,
        num_classes,
        boundary_width=1,
        min_voxels=16,
        memory_weight=0.2,
        memory_momentum=0.1,
        gate_tau=0.7,
        memory_start_iter=0,
        num_branches=2,
        eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            boundary_width=boundary_width,
            min_voxels=min_voxels,
            memory_weight=memory_weight,
            memory_momentum=memory_momentum,
            num_branches=num_branches,
            eps=eps,
        )
        self.gate_tau = gate_tau
        self._pending_gate_updates = []
        self._last_gate_stats = None

    def _store_pending_update(self, current_prototypes, valid_mask, branch_id):
        pass

    def _store_pending_gate_update(self, features, class_region_masks, branch_id):
        with torch.no_grad():
            self._pending_gate_updates.append((
                int(branch_id),
                features.detach(),
                class_region_masks.detach(),
            ))

    def build_reliable_update_mask(self, probabilities_all, spatial_size):
        if probabilities_all is None:
            raise ValueError("GatedRegionMemoryHR.update_memory requires probabilities_all.")
        if probabilities_all.dim() != 6 or probabilities_all.shape[0] < 2:
            raise ValueError("probabilities_all must have shape [branches, B, C, W, H, D].")

        probs = probabilities_all.detach()
        pseudo = torch.argmax(probs, dim=2)
        conf = torch.max(probs, dim=2)[0]
        agreement_mask = pseudo[0] == pseudo[1]
        confidence_mask = (conf[0] >= self.gate_tau) & (conf[1] >= self.gate_tau)
        reliable_mask = agreement_mask & confidence_mask

        conf_mask_a = conf[0] >= self.gate_tau
        conf_mask_b = conf[1] >= self.gate_tau

        if tuple(reliable_mask.shape[-3:]) != tuple(spatial_size):
            reliable_mask = F.interpolate(
                reliable_mask.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1).bool()
            agreement_mask = F.interpolate(
                agreement_mask.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1).bool()
            confidence_mask = F.interpolate(
                confidence_mask.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1).bool()
            conf_mask_a = F.interpolate(
                conf_mask_a.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1).bool()
            conf_mask_b = F.interpolate(
                conf_mask_b.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1).bool()
            conf_a_for_stats = F.interpolate(
                conf[0].unsqueeze(1),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1)
            conf_b_for_stats = F.interpolate(
                conf[1].unsqueeze(1),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1)
        else:
            conf_a_for_stats = conf[0]
            conf_b_for_stats = conf[1]

        stats = {
            "agreement_ratio": float(agreement_mask.float().mean().item()),
            "confidence_ratio": float(confidence_mask.float().mean().item()),
            "conf_pass_ratio_a": float(conf_mask_a.float().mean().item()),
            "conf_pass_ratio_b": float(conf_mask_b.float().mean().item()),
            "gate_pass_ratio": float(reliable_mask.float().mean().item()),
            "conf_a_min": float(conf_a_for_stats.min().item()),
            "conf_a_mean": float(conf_a_for_stats.mean().item()),
            "conf_a_max": float(conf_a_for_stats.max().item()),
            "conf_b_min": float(conf_b_for_stats.min().item()),
            "conf_b_mean": float(conf_b_for_stats.mean().item()),
            "conf_b_max": float(conf_b_for_stats.max().item()),
        }
        return reliable_mask, stats

    def compute_update_prototypes(self, features, class_region_masks, reliable_update_mask):
        batch_size, feat_channels = features.shape[:2]
        update_prototypes = torch.zeros(
            batch_size,
            self.num_classes,
            2,
            feat_channels,
            device=features.device,
            dtype=features.dtype,
        )
        update_valid = torch.zeros(
            batch_size,
            self.num_classes,
            2,
            device=features.device,
            dtype=torch.bool,
        )
        raw_counts = torch.zeros(batch_size, self.num_classes, 2, device=features.device)
        gated_counts = torch.zeros(batch_size, self.num_classes, 2, device=features.device)
        region_valid = torch.zeros(batch_size, self.num_classes, 2, device=features.device, dtype=torch.bool)
        update_rejected = torch.zeros(batch_size, self.num_classes, 2, device=features.device, dtype=torch.bool)

        for ba in range(batch_size):
            sample_features = features[ba]
            reliable_mask = reliable_update_mask[ba].unsqueeze(0).float()
            for n_class in range(self.num_classes):
                for region_idx in range(2):
                    region_mask = class_region_masks[ba, n_class, region_idx].unsqueeze(0)
                    update_mask = region_mask * reliable_mask
                    raw_count = region_mask.sum()
                    gated_count = update_mask.sum()
                    raw_counts[ba, n_class, region_idx] = raw_count
                    gated_counts[ba, n_class, region_idx] = gated_count
                    if raw_count >= self.min_voxels:
                        region_valid[ba, n_class, region_idx] = True
                    if gated_count >= self.min_voxels:
                        update_prototypes[ba, n_class, region_idx] = (
                            sample_features * update_mask
                        ).sum(dim=(1, 2, 3)) / (gated_count + self.eps)
                        update_valid[ba, n_class, region_idx] = True
                    elif raw_count >= self.min_voxels:
                        update_rejected[ba, n_class, region_idx] = True

        return F.normalize(update_prototypes, dim=-1), update_valid, raw_counts, gated_counts, region_valid, update_rejected

    def update_memory_with_gate(self, update_prototypes, update_valid, branch_id):
        update_count = 0
        reject_count = 0
        for n_class in range(self.num_classes):
            for region_idx in range(2):
                valid_samples = update_valid[:, n_class, region_idx]
                if not bool(valid_samples.any()):
                    reject_count += 1
                    continue

                update_proto = update_prototypes[valid_samples, n_class, region_idx].mean(dim=0)
                update_proto = F.normalize(update_proto, dim=0)

                if not bool(self.memory_valid[branch_id, n_class, region_idx]):
                    self.memory_prototypes[branch_id, n_class, region_idx] = update_proto
                    self.memory_valid[branch_id, n_class, region_idx] = True
                else:
                    previous = self.memory_prototypes[branch_id, n_class, region_idx]
                    updated = (1.0 - self.memory_momentum) * previous + self.memory_momentum * update_proto
                    self.memory_prototypes[branch_id, n_class, region_idx] = F.normalize(updated, dim=0)
                update_count += 1

        return update_count, reject_count

    def update_memory(self, probabilities_all=None, label_batch=None, labeled_bs=0, iter_num=None):
        if not self._pending_gate_updates or self.memory_prototypes.numel() == 0:
            self._pending_gate_updates = []
            self._last_gate_stats = None
            return

        with torch.no_grad():
            spatial_size = self._pending_gate_updates[0][1].shape[-3:]
            reliable_mask, stats = self.build_reliable_update_mask(probabilities_all, spatial_size)
            total_updates = 0
            total_rejects = 0
            raw_total = 0.0
            gated_total = 0.0

            region_names = {
                (1, 0): "fg_inner",
                (1, 1): "fg_boundary",
                (0, 1): "bg_boundary",
                (0, 0): "bg_inner",
            }
            branch_region_stats = {}

            for branch_id, features, class_region_masks in self._pending_gate_updates:
                update_prototypes, update_valid, raw_counts, gated_counts, region_valid, update_rejected = self.compute_update_prototypes(
                    features,
                    class_region_masks,
                    reliable_mask,
                )
                update_count, reject_count = self.update_memory_with_gate(
                    update_prototypes,
                    update_valid,
                    branch_id,
                )
                total_updates += update_count
                total_rejects += reject_count
                raw_total += float(raw_counts.sum().item())
                gated_total += float(gated_counts.sum().item())

                for (n_class, region_idx), region_name in region_names.items():
                    key = "branch{}_{}".format(branch_id, region_name)
                    region_count = float(raw_counts[:, n_class, region_idx].sum().item())
                    gated_count = float(gated_counts[:, n_class, region_idx].sum().item())
                    branch_region_stats[key] = {
                        "region_count": region_count,
                        "gated_count": gated_count,
                        "gate_ratio": gated_count / max(region_count, 1.0),
                        "region_valid": int(region_valid[:, n_class, region_idx].sum().item()),
                        "update_allowed": int(update_valid[:, n_class, region_idx].sum().item()),
                        "update_rejected": int(update_rejected[:, n_class, region_idx].sum().item()),
                        "memory_valid": int(bool(self.memory_valid[branch_id, n_class, region_idx])),
                    }

            stats["memory_update_count"] = total_updates
            stats["memory_reject_count"] = total_rejects
            stats["raw_region_voxels"] = raw_total
            stats["gated_region_voxels"] = gated_total
            stats["region_stats"] = branch_region_stats
            self._last_gate_stats = stats

        self._pending_gate_updates = []

    def pop_update_summary(self):
        stats = self._last_gate_stats
        self._last_gate_stats = None
        return stats

    def forward(self, features, probabilities, pseudo_labels, branch_id=None):
        if self.num_classes != 2:
            return super().forward(features, probabilities, pseudo_labels, branch_id=branch_id)
        if branch_id is None:
            raise ValueError("GatedRegionMemoryHR requires branch_id to keep decoder memories independent.")
        if branch_id < 0 or branch_id >= self.num_branches:
            raise ValueError("branch_id must be in [0, {}). Got {}.".format(self.num_branches, branch_id))

        batch_f, c_f = features.shape[:2]
        self._ensure_memory(c_f, features.device, features.dtype)

        pseudo_labels = self._match_label_size(pseudo_labels, features.shape[-3:])
        class_region_masks, boundary_selector = self._binary_region_masks(pseudo_labels)
        current_prototypes, valid_mask, global_prototypes = self.compute_region_prototypes(
            features,
            pseudo_labels,
            class_region_masks,
        )
        effective_prototypes = self.build_effective_prototypes(current_prototypes, valid_mask, branch_id)
        hr_weight = self.compute_hr_from_prototypes(features, probabilities, effective_prototypes, boundary_selector)
        self._store_pending_gate_update(features, class_region_masks, branch_id)
        return hr_weight


class AdaptiveGatedRegionMemoryHR(GatedRegionMemoryHR):
    """E4 = E3 + adaptive current/history prototype fusion.

    The gated memory writing strategy is inherited from E3. This class only
    changes how current and historical region prototypes are fused for the
    current HR computation.
    """

    def __init__(
        self,
        num_classes,
        boundary_width=1,
        min_voxels=16,
        max_memory_weight=0.2,
        memory_momentum=0.1,
        gate_tau=0.7,
        memory_start_iter=0,
        num_branches=2,
        eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            boundary_width=boundary_width,
            min_voxels=min_voxels,
            memory_weight=max_memory_weight,
            memory_momentum=memory_momentum,
            gate_tau=gate_tau,
            memory_start_iter=memory_start_iter,
            num_branches=num_branches,
            eps=eps,
        )
        self.max_memory_weight = float(max_memory_weight)
        self._pending_adaptive_stats = {}

    def _summarize_values(self, values):
        if values.numel() == 0:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "zero_ratio": 0.0,
                "count": 0,
            }
        values = values.detach().float()
        return {
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "zero_ratio": float((values <= self.eps).float().mean().item()),
            "count": int(values.numel()),
        }

    def build_effective_prototypes(self, current_prototypes, valid_mask, branch_id):
        memory = self.memory_prototypes[branch_id].unsqueeze(0)
        memory_valid = self.memory_valid[branch_id].unsqueeze(0)
        use_memory = valid_mask & memory_valid

        current_prototypes = F.normalize(current_prototypes, p=2, dim=-1)
        memory = F.normalize(memory, p=2, dim=-1)
        similarity = (current_prototypes * memory).sum(dim=-1).clamp(min=0.0, max=1.0)
        adaptive_weight = self.max_memory_weight * similarity

        fused = F.normalize(
            adaptive_weight.unsqueeze(-1) * memory
            + (1.0 - adaptive_weight).unsqueeze(-1) * current_prototypes,
            p=2,
            dim=-1,
        )

        region_names = {
            (1, 0): "fg_inner",
            (1, 1): "fg_boundary",
            (0, 1): "bg_boundary",
            (0, 0): "bg_inner",
        }
        with torch.no_grad():
            for (n_class, region_idx), region_name in region_names.items():
                region_use = use_memory[:, n_class, region_idx]
                key = "branch{}_{}".format(branch_id, region_name)
                self._pending_adaptive_stats[key] = {
                    "similarity": self._summarize_values(similarity[:, n_class, region_idx][region_use]),
                    "weight": self._summarize_values(adaptive_weight[:, n_class, region_idx][region_use]),
                }

        return torch.where(use_memory.unsqueeze(-1), fused, current_prototypes)

    def update_memory(self, probabilities_all=None, label_batch=None, labeled_bs=0, iter_num=None):
        super().update_memory(
            probabilities_all=probabilities_all,
        )
        if self._last_gate_stats is not None:
            self._last_gate_stats["adaptive_stats"] = self._pending_adaptive_stats
        self._pending_adaptive_stats = {}


def build_hr_module(
    hr_type,
    num_classes,
    boundary_width=1,
    min_voxels=16,
    memory_weight=0.2,
    max_memory_weight=0.2,
    memory_momentum=0.1,
    num_branches=2,
    gate_tau=0.7,
    memory_start_iter=0,
):
    if hr_type == "original":
        return OriginalHR(num_classes=num_classes)
    if hr_type == "region":
        return RegionHR(num_classes=num_classes, boundary_width=boundary_width, min_voxels=min_voxels)
    if hr_type == "region_memory":
        return RegionMemoryHR(
            num_classes=num_classes,
            boundary_width=boundary_width,
            min_voxels=min_voxels,
            memory_weight=memory_weight,
            memory_momentum=memory_momentum,
            num_branches=num_branches,
        )
    if hr_type == "gated_region_memory":
        return GatedRegionMemoryHR(
            num_classes=num_classes,
            boundary_width=boundary_width,
            min_voxels=min_voxels,
            memory_weight=memory_weight,
            memory_momentum=memory_momentum,
            gate_tau=gate_tau,
            memory_start_iter=memory_start_iter,
            num_branches=num_branches,
        )
    if hr_type == "adaptive_gated_region_memory":
        return AdaptiveGatedRegionMemoryHR(
            num_classes=num_classes,
            boundary_width=boundary_width,
            min_voxels=min_voxels,
            max_memory_weight=max_memory_weight,
            memory_momentum=memory_momentum,
            gate_tau=gate_tau,
            memory_start_iter=memory_start_iter,
            num_branches=num_branches,
        )
    raise ValueError("Unsupported hr_type: {}".format(hr_type))
