import csv
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


class RCCRDiagnosticRecorder:
    """Collect labeled-batch evidence for RCCR without changing training loss.

    The recorder samples voxels in a balanced way over:
        region x correctness

    It writes three CSV files under the experiment snapshot directory:
        rccr_diag_samples.csv
        rccr_diag_counts.csv
        rccr_diag_summary.csv
    """

    REGION_NAMES = ("bg_inner", "bg_boundary", "fg_boundary", "fg_inner")

    def __init__(
        self,
        snapshot_path,
        num_classes,
        interval=50,
        samples_per_group=128,
        eps=1e-6,
    ):
        self.snapshot_path = snapshot_path
        self.num_classes = num_classes
        self.interval = max(int(interval), 1)
        self.samples_per_group = max(int(samples_per_group), 1)
        self.eps = eps
        self.metric_rows = []

        os.makedirs(snapshot_path, exist_ok=True)
        self.samples_path = os.path.join(snapshot_path, "rccr_diag_samples.csv")
        self.counts_path = os.path.join(snapshot_path, "rccr_diag_counts.csv")
        self.summary_path = os.path.join(snapshot_path, "rccr_diag_summary.csv")
        self.last_iter = None

        self.samples_file = open(self.samples_path, "w", newline="")
        self.counts_file = open(self.counts_path, "w", newline="")
        self.samples_writer = csv.writer(self.samples_file)
        self.counts_writer = csv.writer(self.counts_file)

        self.samples_writer.writerow([
            "iter",
            "branch",
            "region_id",
            "region_name",
            "correct",
            "h_region_hr",
            "agreement_jsd",
            "uncertainty_entropy",
            "uncertainty_reliability",
            "joint_mean_score",
            "pseudo_label",
            "gt_label",
        ])
        self.counts_writer.writerow([
            "iter",
            "branch",
            "region_id",
            "region_name",
            "correct",
            "total_voxels",
            "sampled_voxels",
        ])
        self.samples_file.flush()
        self.counts_file.flush()

    def close(self):
        self.write_summary(iter_num=self.last_iter)
        self.samples_file.close()
        self.counts_file.close()

    def should_record(self, iter_num):
        return iter_num == 1 or iter_num % self.interval == 0

    def normalized_entropy(self, probabilities):
        entropy = -(probabilities * torch.log(probabilities + self.eps)).sum(dim=1)
        return entropy / math.log(self.num_classes)

    def jsd_agreement(self, prob_a, prob_b):
        mean_prob = 0.5 * (prob_a + prob_b)
        kl_a = (prob_a * (torch.log(prob_a + self.eps) - torch.log(mean_prob + self.eps))).sum(dim=1)
        kl_b = (prob_b * (torch.log(prob_b + self.eps) - torch.log(mean_prob + self.eps))).sum(dim=1)
        jsd = 0.5 * (kl_a + kl_b)
        agreement = 1.0 - jsd / math.log(self.num_classes)
        return agreement.clamp(min=0.0, max=1.0)

    def _match_spatial(self, tensor, spatial_size, mode):
        if tuple(tensor.shape[-3:]) == tuple(spatial_size):
            return tensor
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        if mode == "nearest":
            out = F.interpolate(tensor.float(), size=spatial_size, mode=mode)
        else:
            out = F.interpolate(tensor.float(), size=spatial_size, mode=mode, align_corners=False)
        return out.squeeze(1) if squeeze else out

    def _sample_indices(self, mask):
        indices = torch.nonzero(mask, as_tuple=False)
        total = indices.shape[0]
        if total == 0:
            return indices, total
        sample_count = min(total, self.samples_per_group)
        if total > sample_count:
            selected = torch.randperm(total, device=indices.device)[:sample_count]
            indices = indices[selected]
        return indices, total

    def _safe_metric(self, values, default=float("nan")):
        try:
            return float(values)
        except Exception:
            return default

    def _auroc(self, scores, labels):
        labels = labels.astype(np.int64)
        scores = scores.astype(np.float64)
        pos = int(labels.sum())
        neg = int(labels.shape[0] - pos)
        if pos == 0 or neg == 0:
            return float("nan")
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        tps = np.cumsum(sorted_labels)
        fps = np.cumsum(1 - sorted_labels)
        tpr = np.concatenate([[0.0], tps / pos, [1.0]])
        fpr = np.concatenate([[0.0], fps / neg, [1.0]])
        return float(np.trapz(tpr, fpr))

    def _auprc(self, scores, labels):
        labels = labels.astype(np.int64)
        scores = scores.astype(np.float64)
        pos = int(labels.sum())
        if pos == 0:
            return float("nan")
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        tps = np.cumsum(sorted_labels)
        precision = tps / (np.arange(sorted_labels.shape[0]) + 1)
        return float((precision * sorted_labels).sum() / pos)

    def _brier(self, scores, labels):
        scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
        labels = labels.astype(np.float64)
        return float(np.mean((scores - labels) ** 2))

    def _metric_record(self, iter_num, branch_name, scope_name, evidence_name, scores, labels):
        return [
            iter_num,
            branch_name,
            scope_name,
            evidence_name,
            int(labels.shape[0]),
            int(labels.sum()),
            int(labels.shape[0] - labels.sum()),
            self._auroc(scores, labels),
            self._auprc(scores, labels),
            self._brier(scores, labels),
        ]

    def write_summary(self, iter_num=None):
        header = [
            "iter",
            "branch",
            "scope",
            "evidence",
            "n",
            "correct_n",
            "wrong_n",
            "auroc",
            "auprc",
            "brier",
        ]
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            if not self.metric_rows:
                return

            data = np.asarray(self.metric_rows, dtype=np.float64)
            branches = [None, 0, 1]
            branch_names = {None: "both", 0: "A", 1: "B"}
            scopes = {
                "all_sampled": np.ones(data.shape[0], dtype=bool),
                "foreground_related": np.isin(data[:, 1], [1, 2, 3]),
                "boundary_related": np.isin(data[:, 1], [1, 2]),
                "bg_inner": data[:, 1] == 0,
                "bg_boundary": data[:, 1] == 1,
                "fg_boundary": data[:, 1] == 2,
                "fg_inner": data[:, 1] == 3,
            }
            evidences = {
                "h_region_hr": data[:, 3],
                "agreement_jsd": data[:, 4],
                "uncertainty_reliability": 1.0 - data[:, 5],
                "joint_mean": (data[:, 3] + data[:, 4] + (1.0 - data[:, 5])) / 3.0,
            }

            summary_iter = int(iter_num) if iter_num is not None else -1
            for branch in branches:
                branch_mask = np.ones(data.shape[0], dtype=bool) if branch is None else data[:, 0] == branch
                for scope_name, scope_mask in scopes.items():
                    mask = branch_mask & scope_mask
                    if not mask.any():
                        continue
                    labels = data[mask, 2].astype(np.int64)
                    for evidence_name, scores in evidences.items():
                        writer.writerow(self._metric_record(
                            summary_iter,
                            branch_names[branch],
                            scope_name,
                            evidence_name,
                            scores[mask],
                            labels,
                        ))

    def _log_main_summary(self, iter_num):
        if not self.metric_rows:
            return
        data = np.asarray(self.metric_rows, dtype=np.float64)
        boundary_mask = np.isin(data[:, 1], [1, 2])
        if not boundary_mask.any():
            return
        labels = data[boundary_mask, 2].astype(np.int64)
        h_score = data[boundary_mask, 3]
        agreement = data[boundary_mask, 4]
        uncertainty_reliability = 1.0 - data[boundary_mask, 5]
        joint_score = (h_score + agreement + uncertainty_reliability) / 3.0
        logging.info(
            "rccr diag iter %d : boundary AUROC h %.6f, agreement %.6f, uncertainty %.6f, joint %.6f | samples %d" % (
                iter_num,
                self._auroc(h_score, labels),
                self._auroc(agreement, labels),
                self._auroc(uncertainty_reliability, labels),
                self._auroc(joint_score, labels),
                int(labels.shape[0]),
            )
        )

    def record(self, iter_num, probabilities_all, hr_weights, label_batch, labeled_bs, hr_module):
        if not self.should_record(iter_num):
            return
        if probabilities_all.shape[0] < 2 or labeled_bs <= 0:
            return
        if not hasattr(hr_module, "compute_region_map"):
            logging.warning("RCCR diagnosis requires RegionHR-compatible hr_module; skipped at iter %d.", iter_num)
            return

        with torch.no_grad():
            probs = probabilities_all[:2, :labeled_bs].detach().float()
            spatial_size = probs.shape[-3:]
            labels = label_batch[:labeled_bs].detach().long()
            labels = self._match_spatial(labels, spatial_size, mode="nearest").long()

            agreement = self.jsd_agreement(probs[0], probs[1])

            for branch_id in range(2):
                branch_probs = probs[branch_id]
                pseudo = torch.argmax(branch_probs, dim=1)
                uncertainty = self.normalized_entropy(branch_probs)
                correctness = (pseudo == labels).long()
                region_map = hr_module.compute_region_map(pseudo.unsqueeze(1), spatial_size=spatial_size)

                h_weight = hr_weights[branch_id]
                if h_weight is None:
                    continue
                h_weight = h_weight[:labeled_bs].detach().float()
                h_weight = self._match_spatial(h_weight, spatial_size, mode="trilinear").clamp(0.0, 1.0)

                for region_id, region_name in enumerate(self.REGION_NAMES):
                    region_mask = region_map == region_id
                    for correct_value in (0, 1):
                        group_mask = region_mask & (correctness == correct_value)
                        indices, total_count = self._sample_indices(group_mask)
                        sampled_count = int(indices.shape[0])
                        self.counts_writer.writerow([
                            iter_num,
                            branch_id,
                            region_id,
                            region_name,
                            correct_value,
                            int(total_count),
                            sampled_count,
                        ])
                        if sampled_count == 0:
                            continue

                        b_idx = indices[:, 0]
                        x_idx = indices[:, 1]
                        y_idx = indices[:, 2]
                        z_idx = indices[:, 3]

                        h_vals = h_weight[b_idx, x_idx, y_idx, z_idx].detach().cpu().numpy()
                        a_vals = agreement[b_idx, x_idx, y_idx, z_idx].detach().cpu().numpy()
                        u_vals = uncertainty[b_idx, x_idx, y_idx, z_idx].detach().cpu().numpy()
                        pseudo_vals = pseudo[b_idx, x_idx, y_idx, z_idx].detach().cpu().numpy()
                        gt_vals = labels[b_idx, x_idx, y_idx, z_idx].detach().cpu().numpy()

                        reliability_u = 1.0 - u_vals
                        joint_vals = (h_vals + a_vals + reliability_u) / 3.0

                        for row_idx in range(sampled_count):
                            row = [
                                iter_num,
                                branch_id,
                                region_id,
                                region_name,
                                correct_value,
                                float(h_vals[row_idx]),
                                float(a_vals[row_idx]),
                                float(u_vals[row_idx]),
                                float(reliability_u[row_idx]),
                                float(joint_vals[row_idx]),
                                int(pseudo_vals[row_idx]),
                                int(gt_vals[row_idx]),
                            ]
                            self.samples_writer.writerow(row)
                            self.metric_rows.append((
                                float(branch_id),
                                float(region_id),
                                float(correct_value),
                                float(h_vals[row_idx]),
                                float(a_vals[row_idx]),
                                float(u_vals[row_idx]),
                            ))

        self.samples_file.flush()
        self.counts_file.flush()
        self.last_iter = iter_num
        self.write_summary(iter_num=iter_num)
        self._log_main_summary(iter_num)
