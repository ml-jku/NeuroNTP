"""
+----------------------------------------------------------------------------------------------------------------------+
segtype_metrics.py
Segment-type evaluation metrics (ictal/non-ictal or preictal/interictal) for NeuroNTP

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve


@dataclass
class SegTypeMetricsConfig:
    threshold: float = 0.5
    compute_by_ctxlen_in_test: bool = True
    log_wandb_artifacts: bool = True
    fusion_level: str = "channel"
    eval_ts_last_k: int = 0
    eval_ts_last_k_agg: str = "mean"


class SegTypeMetrics:
    """
    Unifies segtype-like metrics for:
      - pretrained head (seg_types)
      - downstream head(s) (dws logits)
    Input: lists of tensors collected on each rank.
    Output: dict of scalar metrics (rank0 only; other ranks return {}).
    """

    def __init__(self, cfg: SegTypeMetricsConfig):
        self.cfg = cfg

    @staticmethod
    def _ddp_concat_numpy(y_true_local: np.ndarray, y_prob_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if dist.is_available() and dist.is_initialized():
            y_true_list = [None] * dist.get_world_size()
            y_prob_list = [None] * dist.get_world_size()
            dist.all_gather_object(y_true_list, y_true_local)
            dist.all_gather_object(y_prob_list, y_prob_local)
            y_true = np.concatenate([x for x in y_true_list if x is not None], axis=0)
            y_prob = np.concatenate([x for x in y_prob_list if x is not None], axis=0)
        else:
            y_true, y_prob = y_true_local, y_prob_local
        return y_true, y_prob

    @staticmethod
    def _agg_last_k_in_ts(yp: torch.Tensor, ep: torch.Tensor, k: int, mode: str = "max") -> torch.Tensor:
        """
        yp: [..., L] token probabilities, ep: [..., L] bool EoS mask (same shape, time on last dim).
        Returns [..., L] where each position holds the max/mean over itself and up to k-1
        preceding tokens *of the same timestep*. Only the values at EoS positions are meaningful.
        """
        ep = ep.to(device=yp.device).bool()
        # timestep segment id per token; the EoS token belongs to its own timestep
        seg = torch.cumsum(ep.long(), dim=-1) - ep.long()

        # sliding windows over [t-k+1, ..., t]
        yp_w = F.pad(yp.float(), (k - 1, 0), value=0.0).unfold(-1, k, 1)  # [..., L, k]
        seg_w = F.pad(seg, (k - 1, 0), value=-1).unfold(-1, k, 1)  # [..., L, k]
        valid = seg_w == seg.unsqueeze(-1)  # same timestep, no padding

        if mode == "max":
            return yp_w.masked_fill(~valid, float("-inf")).amax(dim=-1)
        elif mode == "mean":
            return (yp_w * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        raise ValueError(f"Unknown eval_ts_last_k_agg: {mode}")

    def compute(
        self,
        y_true_list: List[torch.Tensor],  # list of (N, L) int/long CPU tensors
        y_prob_list: List[torch.Tensor],  # list of (N, L) float CPU tensors (already sigmoid)
        *,
        regime: str,
        prefix: str,
        eval_ctx_len: Optional[int] = -1,
        logger_experiment=None,  # e.g. self.logger.experiment (wandb)
        is_global_zero: bool = True,
        eos_pos: List[torch.Tensor] = None,
        ctx_len: int = 30
    ) -> Dict[str, float]:
        """
        Returns dict of metrics. Clears lists is the caller's job.
        """
        if not y_true_list or not y_prob_list:
            return {}

        # concat on this rank
        if self.cfg.fusion_level == "timestep":
            y_true_local = torch.cat(y_true_list, dim=0).detach().cpu().numpy()
            y_prob_local = torch.cat(y_prob_list, dim=0).detach().cpu().numpy()
        else:
            y_true_local = torch.cat([yt.reshape(-1) for yt in y_true_list]).detach().cpu().numpy()
            y_prob_local = torch.cat([yp.reshape(-1) for yp in y_prob_list]).detach().cpu().numpy()
            if eos_pos is not None:
                mask = torch.cat([ep.reshape(-1) for ep in eos_pos]).detach().cpu().numpy().astype(bool)
                ctx_true = y_true_local[mask]

                k = getattr(self.cfg, "eval_ts_last_k", None)
                if k:  # None or 0 -> default behaviour
                    agg_mode = getattr(self.cfg, "eval_ts_last_k_agg", "max")
                    y_prob_agg = torch.cat([
                        self._agg_last_k_in_ts(yp.reshape(ep.shape), ep, int(k), agg_mode).reshape(-1)
                        for yp, ep in zip(y_prob_list, eos_pos)
                    ]).detach().cpu().numpy()
                    ctx_prob = y_prob_agg[mask]
                else:
                    ctx_prob = y_prob_local[mask]

        # DDP gather
        y_true, y_prob = self._ddp_concat_numpy(y_true_local, y_prob_local)

        # rank0 only
        if not is_global_zero:
            return {}

        metrics: Dict[str, float] = {}

        # --- aggregate over all timesteps ---
        yt_all = y_true.reshape(-1)
        yp_all = y_prob.reshape(-1)

        # handle edge case: all labels same
        uniq = np.unique(yt_all)
        if uniq.size == 1:
            prevalence = float(yt_all.mean())
            preds = (yp_all > self.cfg.threshold).astype(int)
            bacc = float(balanced_accuracy_score(yt_all, preds))  # will be 1.0 in degenerate cases
            # AP undefined when only one class; set AP = prevalence for interpretability
            ap = float(prevalence)
            auc = float("nan")
        else:
            ap = float(average_precision_score(yt_all, yp_all))
            preds = (yp_all > self.cfg.threshold).astype(int)
            bacc = float(balanced_accuracy_score(yt_all, preds))
            try:
                auc = float(roc_auc_score(yt_all, yp_all))
            except ValueError:
                auc = float("nan")
            prevalence = float(yt_all.mean())

        davgp = float(ap - prevalence)

        metrics[f"{regime}/{prefix}_ap"] = ap
        metrics[f"{regime}/{prefix}_davgp"] = davgp
        metrics[f"{regime}/{prefix}_bacc"] = bacc
        if not np.isnan(auc):
            metrics[f"{regime}/{prefix}_auroc"] = auc
        metrics[f"{regime}/{prefix}_prevalence"] = prevalence

        if self.cfg.fusion_level != "timestep":
            if eos_pos is None:
                return metrics
            y_true = ctx_true.reshape(-1, ctx_len)
            y_prob = ctx_prob.reshape(-1, ctx_len)

        if regime == "test" and self.cfg.compute_by_ctxlen_in_test:
            L = y_true.shape[1]
            ap_t = np.full(L, np.nan, dtype=float)
            auc_t = np.full(L, np.nan, dtype=float)
            bacc_t = np.full(L, np.nan, dtype=float)
            prev_t = y_true.mean(axis=0)
            best_thr_t = np.full(L, np.nan, dtype=float)
            bacc_opt_t = np.full(L, np.nan, dtype=float)

            for t in range(L):
                yt = y_true[:, t]
                yp = y_prob[:, t]
                uniq_t = np.unique(yt)
                if uniq_t.size == 1:
                    # keep as NaN or set to baseline ...
                    continue
                ap_t[t] = average_precision_score(yt, yp)
                auc_t[t] = roc_auc_score(yt, yp)
                bacc_t[t] = balanced_accuracy_score(yt, (yp > self.cfg.threshold).astype(int))

                # val.opt. bacc
                fpr_t, tpr_t, thr_t = roc_curve(yt, yp)
                j = tpr_t - fpr_t
                idx = np.argmax(j)
                best_thr_t[t] = thr_t[idx]
                ypred = (yp >= best_thr_t[t]).astype(int)
                bacc_opt_t[t] = balanced_accuracy_score(yt, ypred)

            delta_ap_t = ap_t - prev_t

            # log "ctx index" scalars if provided
            if eval_ctx_len is not None and -1 <= int(eval_ctx_len) < L:
                i = int(eval_ctx_len)
                if not np.isnan(ap_t[i]):
                    metrics[f"{regime}/{prefix}_ctx_ap"] = float(ap_t[i])
                    metrics[f"{regime}/{prefix}_ctx_davgp"] = float(delta_ap_t[i])
                    metrics[f"{regime}/{prefix}_ctx_bacc"] = float(bacc_t[i])
                    metrics[f"{regime}/{prefix}_ctx_auroc"] = float(auc_t[i]) if not np.isnan(auc_t[i]) else float("nan")
                    metrics[f"{regime}/{prefix}_ctx_bacc_opt"] = float(bacc_opt_t[i])
                    metrics[f"{regime}/{prefix}_ctx_thr_opt"] = float(best_thr_t[i])

        # additional metrics and details
        yt = y_true[:, int(eval_ctx_len)]
        yp = y_prob[:, int(eval_ctx_len)]
        thr = self.cfg.threshold
        pred = (yp > thr).astype(np.int32)

        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())

        metrics[f"{regime}/{prefix}_ctx_tp"] = tp
        metrics[f"{regime}/{prefix}_ctx_fp"] = fp
        metrics[f"{regime}/{prefix}_ctx_fn"] = fn
        metrics[f"{regime}/{prefix}_ctx_tn"] = tn

        # derived
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fpr = fp / (fp + tn + eps)
        pred_pos_rate = float(pred.mean())

        metrics[f"{regime}/{prefix}_ctx_precision@{thr}"] = float(precision)
        metrics[f"{regime}/{prefix}_ctx_recall@{thr}"] = float(recall)
        metrics[f"{regime}/{prefix}_ctx_fpr@{thr}"] = float(fpr)
        metrics[f"{regime}/{prefix}_ctx_pred_pos_rate@{thr}"] = pred_pos_rate

        return metrics
