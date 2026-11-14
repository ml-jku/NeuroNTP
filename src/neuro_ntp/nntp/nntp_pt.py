"""
+----------------------------------------------------------------------------------------------------------------------+
nntp_pt.py
Lightning module and pretrain() entry point for NeuroNTP pretraining and evaluation

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Implements the NeuroNTP pretraining and evaluation pipeline with PyTorch Lightning, used for the pretrain,
    linprobe and evaluate modes. The module works with the NeuroNTP data module (nntp_dm.py) and expects the
    mini-batches in the self-supervised sample format: (signals, chunks, marker_embs, notes_embs, metadata), each
    sample covering a contiguous recording segment of T chunks of length L (e.g. 30 x 10 s = 5 min). The lightning
    module is also used by fine-tuning (nntp_ft.py). For the sample structure, see the data module and the
    self-supervised dataset view classes of the individual datasets.
+----------------------------------------------------------------------------------------------------------------------+
"""
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.datapipes.gen_pyi import mapDP_method_to_special_output_type
from transformers import get_cosine_schedule_with_warmup

from neuro_ntp.model.ntp.nntp_misc import QuantileLoss, FocalLoss, find_closest_markers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from neuro_ntp.model.ntp.nntp_misc import pretrain_collate_fn as collate
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from neuro_ntp.model.mmt.core.token_type import TokenType, token_type_id, parse_channel_base_and_sub, SUBTYPE_VOCABS
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy
import os
from neuro_ntp.model.ntp.neuro_ntp import NeuroNTP
from neuro_ntp.nntp.nntp_dm import PretrainingDataModule
import neuro_ntp.misc.nntp_utils as nnutils
from neuro_ntp.misc.segtype_metrics import SegTypeMetrics, SegTypeMetricsConfig

import logging
logger = logging.getLogger(__name__)


class PretrainingLightningModel(pl.LightningModule):

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.config = args.config
        self.num_devices = (len(self.args.device.split(",")) if self.args.device != "cpu" else 1)

        # loss coefficients
        self.alfa = self.config['pt_loss_alfa']
        self.beta = self.config['pt_loss_beta']
        self.gamma = self.config['pt_loss_gamma']
        self.delta = self.config['pt_loss_delta']
        self.epsilon = self.config['pt_loss_epsilon']
        self.zeta = self.config['pt_loss_zeta']

        # finetuning/pretrain coefficient
        self.tau = self.config['finetuning_coeff']

        # ictal onset loss weighting (onset_weighting, not used in the released configurations)
        self.use_onset_w = bool(self.config.get("onset_weighting", False))
        if self.use_onset_w:
            self.max_ts = float(self.config.get("max_ts", 10 ** 9))
            self.onset_chi_s = float(self.config.get("onset_chi_s", 30.0))
            self.max_w = float(self.config.get("onset_max_w", 4.0))

        # init output and metric collections
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_embeddings = []
        self.st_ap_logits = []
        self.st_ap_targets = []
        self.st_eos_pos = []
        self.mmt_ap_logits = []
        self.mmt_ap_targets = []

        # downstream metric collections (fine-tuning with task-specific heads)
        if args.mode == "finetune" and self.model.config.get("downstream_tasks", {}).get("enabled", False):
            task_names = list(self.model.dws_heads.keys())
            self.dws_logits = {t: [] for t in task_names}
            self.dws_targets = {t: [] for t in task_names}
            self.dws_eos_pos = {t: [] for t in task_names}
            self.dws_floss_fn = FocalLoss(alpha=self.config['pt_segtype_focal_alpha'],
                                         gamma=self.config['pt_segtype_focal_gamma'])

        # NTP loss
        ntp_loss_type = self.config.get("ntp_loss", "WQL")
        if ntp_loss_type == "WQL":
            # NTP multi-quantile loss: sort by q for stable pairing (fn, weight)
            q_items = sorted(self.config['pt_quantiles'].items(), key=lambda kv: float(kv[0]))
            q_keys, q_weights = zip(*q_items) if q_items else ([], [])
            self.ntp_qloss_fns = nn.ModuleList([QuantileLoss(float(q)) for q in q_keys])
            self.ntp_qloss_weights = list(q_weights)
        elif ntp_loss_type == "MSE":
            # MSE loss for next-token regression
            self.ntp_loss_fn = nn.MSELoss()
        else:
            # L1 loss for next-token regression
            self.ntp_loss_fn = nn.L1Loss()

        # token-type loss
        if self.config['predict_token_type']:
            class_weights = torch.ones(TokenType.max_id() + 1, dtype=torch.float32)
            if self.config['pt_toktype_weights']:
                for key, value in self.config['pt_toktype_weights'].items():
                    class_weights[token_type_id(key)] *= float(value)
            self.register_buffer("tt_class_weights", class_weights)

        # segment-type loss
        if self.config['predict_segtype']:
            self.st_floss_fn = FocalLoss(alpha=self.config['pt_segtype_focal_alpha'], gamma=self.config['pt_segtype_focal_gamma'])

        # TPS smooth L1 loss
        if self.config['predict_tps']:
            self.tps_loss_fn = nn.SmoothL1Loss(beta=self.config['pt_tps_s1loss_beta'])

        # TTS smooth L1 loss
        if self.config['predict_tts']:
            self.tts_loss_fn = nn.SmoothL1Loss(beta=self.config['pt_tts_s1loss_beta'])

        # marker reconstruction: quantile loss
        if self.config.get('decode_markers', False):
            m_items = sorted(self.config['pt_marker_quantiles'].items(), key=lambda kv: float(kv[0]))
            mq_keys, mq_weights = zip(*m_items) if m_items else ([], [])
            self.mkr_qloss_fns = nn.ModuleList([QuantileLoss(float(q)) for q in mq_keys])
            self.mkr_qloss_weights = list(mq_weights)
            self.mkr_results_list = []

        self.total_predicted_markers = 0
        self.total_expected_markers = 0
        self.token_masking = args.token_masking
        self.metrics_logging_active = True

        # segment type metrics handler
        self.seg_eval = SegTypeMetrics(SegTypeMetricsConfig(
            threshold=float(getattr(self.args, "threshold", 0.5)),
            compute_by_ctxlen_in_test=True,
            log_wandb_artifacts=True,
            fusion_level=self.config.get('fusion_level', 'timestep'),
            eval_ts_last_k=int(getattr(self.args, "eval_ts_last_k", 0)),
            eval_ts_last_k_agg=str(getattr(self.args, "eval_ts_last_k_agg", "mean"))
        ))
        self._seizure_logits_source = "unset"

        # stateful inference cache for recurrent backbones
        self._ntp_state = None
        self._stream_key = None  # detects a new session or patient to reset the cache
        self.use_stateful = bool(getattr(self.args, "stateful_inference", False))

        # scheduled masking
        self.mask_sch_enabled = self.config.get("mask_scheduler_enabled", False)
        if self.mask_sch_enabled:
            self.mask_sch_p1 = self.config.get("mask_sch_p1", 1.0)
            self.mask_sch_warmup = self.config.get("mask_sch_warmup", 10)
            self.mask_sch_target = self.config.get("mask_sch_target", 80)
            self.mask_keys = ["segtype_mask_prob", "tps_mask_prob", "markers_mask_prob", "notes_mask_prob",]
            self.mask_probs = {k: float(self.config.get(k, 0.0)) for k in self.mask_keys}

    def common_step(self, batch, batch_idx, regime):
        """ Common step implementation for train/val/test regimes. """
        device = next(self.parameters()).device                               # infer device
        signals, chunks, marker_embs, notes_embs, metadata = batch            # self-supervised batch
        inp = (signals.to(device), chunks, marker_embs, notes_embs, metadata) # move signal to device

        # predicted token sequence (NTP), tokenizer output (MMT), downstream head predictions
        ntp_seq, mmt_seq, dws_pred = self.model(inp)

        # next token prediction loss - primary objective
        ntp_loss_type = self.config.get("ntp_loss", "WQL")
        if ntp_loss_type == "WQL":
            ntp_loss = sum(w * l(ntp_seq.tokens, mmt_seq.target_tokens[:, 1:, :])
                           for l, w in zip(self.ntp_qloss_fns, self.ntp_qloss_weights))
        else:
            ntp_loss = self.ntp_loss_fn(ntp_seq.tokens, mmt_seq.target_tokens[:, 1:, :])
        self.log_metrics(f"{regime}/ntp_loss", ntp_loss)

        # auxiliary losses
        ttype_clf_loss, tps_loss, tts_loss, st_loss, mkr_rec_loss = self.multitask_losses(
            ntp_seq, mmt_seq, device, regime, batch_idx, marker_embs, chunks)

        # calculate the overall compound loss
        loss = (self.alfa * ntp_loss
                + self.beta * ttype_clf_loss
                + self.gamma * tps_loss
                + self.delta * tts_loss
                + self.epsilon * st_loss
                + self.zeta * mkr_rec_loss)

        fusion_level = self.config.get('fusion_level', 'timestep')

        # fine-tuning: downstream task losses
        if self.args.mode == "finetune":
            dws_total = torch.tensor(0.0, device=device)
            tasks = list(dws_pred.keys())
            for task in tasks:
                if task not in dws_pred or dws_pred[task] is None:
                    continue

                st_logits = dws_pred[task]  # expect (B, L) or (B, L, 1)
                # normalize shape -> (B, L)
                if st_logits.ndim == 3 and st_logits.size(-1) == 1:
                    st_logits = st_logits[..., 0]
                bin_targets = self._get_dws_targets(task, mmt_seq, device)  # (B, L)

                if (getattr(self.args, "dws_loss_scope", "dense") == "eos"
                    and fusion_level == 'channel'):
                    eos_mask = (mmt_seq.token_types[:, 1:] == int(TokenType.EOS))  # (B, L) bool
                    st_sel = st_logits[eos_mask]  # (B*seq_len,)
                    y_sel = bin_targets[eos_mask]  # (B*seq_len,)
                    task_loss = self.dws_floss_fn(st_sel, y_sel)
                elif (getattr(self.args, "dws_loss_scope", "dense") == "last_eos"
                          and fusion_level == 'channel'):
                    eos_mask = (mmt_seq.token_types[:, 1:] == int(TokenType.EOS))  # (B, L)
                    B, L = eos_mask.shape
                    idx = torch.arange(L, device=eos_mask.device).unsqueeze(0).expand(B, L)
                    last_idx = torch.where(eos_mask, idx, torch.full_like(idx, -1)).amax(dim=1)  # (B,)
                    st_sel = st_logits.gather(1, last_idx[:, None]).squeeze(1)  # (B,)
                    y_sel = bin_targets.gather(1, last_idx[:, None]).squeeze(1)  # (B,)
                    task_loss = self.dws_floss_fn(st_sel, y_sel)
                else:
                    # dense: all tokens (default)
                    task_loss = self.dws_floss_fn(st_logits.reshape(-1), bin_targets.reshape(-1))

                weight = 1.0
                if hasattr(self.model, "dws_specs") and task in self.model.dws_specs:
                    weight = float(self.model.dws_specs[task].get("weight", 1.0))
                dws_total = dws_total + weight * task_loss
                # log per-task
                self.log_metrics(f"{regime}/{task}_loss", task_loss)
                if regime != "train":
                    with torch.no_grad():
                        self.dws_logits[task].append(torch.sigmoid(st_logits).detach().cpu())
                        self.dws_targets[task].append(bin_targets.long().detach().cpu())
                        if fusion_level == 'channel':
                            self.dws_eos_pos[task].append((mmt_seq.token_types[:, 1:] == int(TokenType.EOS)).detach().cpu())

            # weighted combination: (1 - tau) * pretraining losses + tau * downstream losses
            loss = loss * (1 - self.tau) + dws_total * self.tau
            self.log_metrics(f"{regime}/dws_loss", dws_total)

        # log the overall loss
        self.log_metrics(f"{regime}/loss", loss)

        ########################################################
        # monitoring: gradient norm and embedding statistics
        ########################################################
        # monitor gradient norm (only in training regime)
        if regime == 'train':
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.log_metrics(f"{regime}/grad_norm", grad_norm)

        if regime != 'test':
            # log token embedding (MMT) variance over sequence
            emb_var = torch.var(mmt_seq.tokens, dim=1).mean()
            self.log_metrics(f"{regime}/mmt_emb_var", emb_var)

            # log predicted token (NTP) variance over sequence
            pred_var = torch.var(ntp_seq.tokens, dim=1).mean()
            self.log_metrics(f"{regime}/ntp_pred_var", pred_var)

        # log cosine similarity between target tokens (MMT) and predicted tokens (NTP)
        avg_cosine_sim = nn.functional.cosine_similarity(ntp_seq.tokens, mmt_seq.target_tokens[:, 1:, :], dim=-1).mean()
        self.log_metrics(f"{regime}/ntp_cos_sim", avg_cosine_sim)

        return loss

    def stateful_inference_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        signals, chunks, marker_embs, notes_embs, metadata = batch
        inp = (signals.to(device), chunks, marker_embs, notes_embs, metadata)

        # reset cache on new stream
        stream_key = self._get_stream_key(metadata)
        if stream_key != self._stream_key:
            logger.debug(">>> RESET STREAM KEY <<<")
            self._stream_key = stream_key
            self._ntp_state = None

        fusion_level = self.config.get("fusion_level", "timestep")

        with (torch.inference_mode()):
            if fusion_level == "channel":
                ntp_seq, mmt_seq, dws_pred, self._ntp_state = self.model.forward_stateful_channel(inp, self._ntp_state)
            else:
                ntp_seq, mmt_seq, dws_pred, self._ntp_state = self.model.forward_stateful(inp, self._ntp_state)

        # ---------------- AUX segtype metrics ----------------
        if ntp_seq.seg_types is not None and mmt_seq.seg_types is not None:
            st_logits = ntp_seq.seg_types
            if st_logits.ndim == 3 and st_logits.size(-1) == 1:
                st_logits = st_logits[..., 0]  # (B, Lpred)

            # prob of the last predicted token of the returned block
            st_prob_head = torch.sigmoid(st_logits[:, -1])  # (B,)

            st_targets = mmt_seq.seg_types[:, 1:].to(device)  # (B, S-1)
            st_true_head = (st_targets[:, -1] == 1).long()

            with torch.no_grad():
                self.st_ap_logits.append(st_prob_head.detach().cpu())  # (B,)
                self.st_ap_targets.append(st_true_head.detach().cpu())  # (B,)

        return torch.tensor(0.0, device=device)

    def _get_dws_targets(self, task: str, mmt_seq, device):
        st_targets = mmt_seq.seg_types[:, 1:].to(device)  # (B, L)
        bin_targets = (st_targets == 1).float()  # ictal=1
        return bin_targets

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        if self.use_stateful:
            out = self.stateful_inference_step(batch, batch_idx)
        else:
            out = self.common_step(batch, batch_idx, 'test')
        return out

    def multitask_losses(self, ntp_seq, mmt_seq, device, regime, batch_idx, marker_embs, chunks):
        fusion_level = self.config.get('fusion_level', 'timestep')
        # token type prediction loss (channel level) / marker detection loss (in timestep-fused mode)
        ttype_clf_loss = 0.0
        if self.config.get('predict_token_type', False) and (ntp_seq.token_types is not None):
            if fusion_level == 'channel':
                tt_logits = ntp_seq.token_types
                C = tt_logits.size(-1)
                tt_targets = mmt_seq.token_types[:, 1:].to(device)  # (B, S-1)
                ttype_clf_loss = F.cross_entropy(tt_logits.reshape(-1, C), tt_targets.reshape(-1),
                                                 weight=getattr(self, "tt_class_weights", None))
                self.log_metrics(f"{regime}/token_type_loss", ttype_clf_loss)
            else:
                mp_logits = ntp_seq.token_types
                mp_targets = (mmt_seq.token_types[:, 1:] == int(TokenType.Marker)).float().to(device)  # (B, S-1)
                ttype_clf_loss = self.st_floss_fn(mp_logits.reshape(-1), mp_targets.reshape(-1))
                self.log_metrics(f"{regime}/mkr_detection_loss", ttype_clf_loss)

        # TPS loss
        tps_loss = torch.tensor(0.0, device=device)
        if self.config.get('predict_tps', False) and (ntp_seq.tps is not None) and (mmt_seq.tps is not None):
            tps_loss = self.tps_loss_fn(ntp_seq.tps.reshape(-1), mmt_seq.tps[:, 1:].to(device).reshape(-1))
            self.log_metrics(f"{regime}/tps_loss", tps_loss)

        # TTS loss
        tts_loss = torch.tensor(0.0, device=device)
        if self.config.get('predict_tts', False) and (ntp_seq.tts is not None) and (mmt_seq.tts is not None):
            tts_loss = self.tts_loss_fn(ntp_seq.tts.reshape(-1), mmt_seq.tts[:, 1:].to(device).reshape(-1))
            self.log_metrics(f"{regime}/tts_loss", tts_loss)

        # segment type loss
        st_loss = 0.0
        if self.config.get('predict_segtype', False) and (mmt_seq.seg_types is not None):
            st_logits = ntp_seq.seg_types
            if st_logits.ndim == 3 and st_logits.size(-1) == 1:
                st_logits = st_logits[..., 0]
            st_targets = mmt_seq.seg_types[:, 1:].to(device)
            bin_targets = (st_targets == 1).float()  # ictal is 1

            # onset weighting: higher weight for the first ictal chunks (onset_weighting, not used in the released configurations)
            if self.use_onset_w:
                # chunks: object array (B, L, K); column 5 holds the time past seizure
                tps_np = chunks[..., 5].astype(np.float32)
                tps_s = torch.from_numpy(tps_np).to(device)[:, 1:]
                w = self.onset_weights_from_tps(tps_s, bin_targets)

                # elementwise BCE
                elem = F.binary_cross_entropy_with_logits(st_logits, bin_targets, reduction="none")
                st_loss = (elem * w).mean()

                self.log_metrics(f"{regime}/segtype_onset_w_mean", w.mean())
                self.log_metrics(f"{regime}/segtype_onset_w_max", w.max())
            else:
                st_loss = self.st_floss_fn(st_logits.reshape(-1), bin_targets.reshape(-1))

            # collect predictions and targets for the segment-type metrics (val/test)
            if regime != "train":
                with torch.no_grad():
                    self.st_ap_logits.append(torch.sigmoid(st_logits).detach().cpu())
                    self.st_ap_targets.append(bin_targets.long().detach().cpu())
                    if fusion_level == 'channel':
                        self.st_eos_pos.append((mmt_seq.token_types[:, 1:] == int(TokenType.EOS)).detach().cpu())
            self.log_metrics(f"{regime}/seg_type_loss", st_loss)

        # number of expected markers (channel-level and timestep-fused)
        expected_markers = 0
        mp = getattr(mmt_seq, "marker_present", None)  # (B, T) bool: marker present at the timestep
        if mp is not None:
            expected_markers = int(mp[:, 1:].sum().item())  # next-step positions
        elif getattr(mmt_seq, "marker_pos", None) is not None:
            expected_markers = len(mmt_seq.marker_pos)
        self.total_expected_markers += expected_markers

        # marker detection diagnostics (channel-level and timestep-fused)
        if mp is not None and (ntp_seq.token_types is not None):
            # align with head: next-step / causal offset
            expected_next = torch.as_tensor(mp, device=device)[:, 1:]

            if fusion_level == "timestep":
                predicted_next = (ntp_seq.token_types.squeeze(-1).sigmoid() > 0.5)
            else:
                predicted_next = (ntp_seq.token_types.argmax(dim=-1) == int(TokenType.Marker))

            tp = (expected_next & predicted_next).sum().item()
            pred = predicted_next.sum().item()
            exp = expected_next.sum().item()

            # log counts (in train/val/test)
            self.log_metrics(f"{regime}/mkr_expected", exp)
            self.log_metrics(f"{regime}/mkr_predicted", pred)
            self.log_metrics(f"{regime}/mkr_true_pos", tp)

            # marker recall and precision
            if exp > 0:
                self.log_metrics(f"{regime}/mkr_recall", tp / exp)
            if pred > 0:
                self.log_metrics(f"{regime}/mkr_precision", tp / pred)

        # marker reconstruction loss
        mkr_rec_loss = torch.tensor(0.0, device=device)

        if self.config.get('decode_markers', False) and (ntp_seq.markers is not None):
            mkr_rec_loss = self.marker_reconstruction(mmt_seq, ntp_seq, marker_embs, mp, device, regime, batch_idx, chunks)

        return ttype_clf_loss, tps_loss, tts_loss, st_loss, mkr_rec_loss

    def onset_weights_from_tps(self, tps_s: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """
        tps_s: (B, L) seconds since seizure onset (>=0), may include max_ts sentinel
        pos_mask: (B, L) 1 where ictal target is positive, 0 elsewhere
        Returns weights (B, L) in [1, max_w] applied only on positives.
        """
        tps = tps_s.float()

        # ignore invalid/sentinel values
        valid = (tps >= 0) & (tps < self.max_ts)

        # w = 1 + (max_w-1)*exp(-tps/tau), only on positives and valid
        w = torch.ones_like(tps)
        w_pos = 1.0 + (self.max_w - 1.0) * torch.exp(-tps / max(self.onset_chi_s, 1e-6))
        w = torch.where(valid & (pos_mask > 0.5), w_pos, w)
        return w

    def marker_reconstruction(self, mmt_seq, ntp_seq, marker_embs, mp, device, regime, batch_idx, chunks):
        # marker positions provided by the model, (K, 2) rows of [batch index, token position]
        idx = getattr(ntp_seq, "marker_pos", None)
        if idx is None:
            # otherwise, positions from the dense marker-presence mask
            if (ntp_seq.token_types is not None) and (mp is not None):
                expected_next = torch.as_tensor(mp, device=device)[:, 1:]  # (B, T-1)
                # predicted_next = (ntp_seq.token_types.argmax(dim=-1) == int(TokenType.Marker))  # (B, T-1)
                mask = expected_next  # & predicted_next
                if mask.any():
                    idx = mask.nonzero(as_tuple=False)  # (K, 2)
            elif getattr(mmt_seq, "marker_pos", None) is not None:
                # explicit marker positions of the tokenizer; the prediction targets the next step
                idx = torch.stack([
                    torch.tensor([b, t - 1], device=device) for (b, t) in mmt_seq.marker_pos if t > 0
                ], dim=0) if len(mmt_seq.marker_pos) > 0 else None

        if (idx is None) or (idx.numel() == 0) or (marker_embs is None):
            return 0.0

        # compact predictions (K, Dm)
        pred_mkr_emb = ntp_seq.markers.to(device)  # (K, Dm)

        # targets: marker embeddings of the next timestep, marker_embs (B, T, Dm)
        idx = idx.to(device)
        T = int(marker_embs.shape[1])

        fusion_level = getattr(self.model, "config", {}).get("fusion_level", None)
        if fusion_level is None:
            # fall back to the tokenizer configuration
            fusion_level = self.model.mm_tokenizer.config.get("fusion_level", "timestep")

        b = idx[:, 0].long()
        s_minus1 = idx[:, 1].long()

        if fusion_level == "channel":
            # channel fusion: flattened sequence length S = T * C
            S = int(mmt_seq.tokens.shape[1])
            if S % T != 0:
                raise RuntimeError(f"Cannot infer C in channel-fusion: S={S} not divisible by T={T}")
            C = S // T
            t = (s_minus1 // C) + 1  # next timestep target
        else:
            # timestep-fusion: s_minus1 is already timestep-1
            t = s_minus1 + 1

        # keep positions with a valid target timestep
        keep = (b >= 0) & (b < int(marker_embs.shape[0])) & (t >= 0) & (t < T)
        if not bool(keep.any()):
            return 0.0

        b = b[keep]
        t = t[keep]
        pred_mkr_emb = pred_mkr_emb[keep]

        b_cpu = b.detach().cpu().tolist()
        t_cpu = t.detach().cpu().tolist()

        tgt_text = [str(chunks[bi, ti, 3]).strip() for bi, ti in zip(b_cpu, t_cpu)]
        nonempty = torch.tensor([s != "" for s in tgt_text], device=device)

        if not bool(nonempty.any()):
            return 0.0

        ####################
        # apply filter
        b = b[nonempty]
        t = t[nonempty]
        pred_mkr_emb = pred_mkr_emb[nonempty]
        tgt_text = [s for s, ok in zip(tgt_text, nonempty.detach().cpu().tolist()) if ok]

        # keep positions with a non-empty target marker
        tgt_mkr = marker_embs[b, t].to(device)
        ####################
        #tgt_mkr = marker_embs[b, t].to(device)  # (K_kept, Dm)

        # reconstruction loss and similarity metrics
        cos_sim = F.cosine_similarity(pred_mkr_emb, tgt_mkr, dim=-1, eps=1e-8)
        self.log_metrics(f"{regime}/marker_cos_sim", cos_sim.mean())

        mkr_rec_loss = sum(
            w * l(pred_mkr_emb, tgt_mkr) for l, w in zip(self.mkr_qloss_fns, self.mkr_qloss_weights)
        )
        self.log_metrics(f"{regime}/marker_recon_loss", mkr_rec_loss)

        self.total_predicted_markers += int(pred_mkr_emb.size(0))

        if regime != "train":
            pred_text = find_closest_markers(
                pred_mkr_emb,
                self.ds_view.extended_marker_emb_dict if self.args.ext_markers else self.ds_view.marker_emb_dict,
                topk=self.args.topk,
            )
            # targets as nearest dictionary entries, in the same text space as the predictions
            tgt_text = find_closest_markers(tgt_mkr, self.ds_view.marker_emb_dict)

            jaccard = 0.0
            n_pred = len(pred_text)
            for p, ttxt in zip(pred_text, tgt_text):
                self.mkr_results_list.append(
                    {"epoch": self.current_epoch, "batch_idx": batch_idx, "predicted": p, "target": ttxt,
                     "type": "cos-sim"}
                )
                ps = set(map(str.strip, p.split(","))) if p else set()
                ts = set(map(str.strip, ttxt.split(","))) if ttxt else set()
                jaccard += (len(ps & ts) / len(ps | ts)) if (ps or ts) else 1.0

            if n_pred > 0:
                self.log_metrics(f"{regime}/marker_IoU", jaccard / n_pred)

        return mkr_rec_loss

    def log_marker_predictions(self, regime: str):
        # gather from all ranks (list of lists of dicts)
        if dist.is_available() and dist.is_initialized():
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, self.mkr_results_list)
            all_results = []
            for gpu_results in gathered_results:
                if gpu_results:
                    all_results.extend(gpu_results)
        else:
            all_results = list(self.mkr_results_list)

        # clear local buffer on every rank to prevent growth across epochs
        self.mkr_results_list = []

        # only global rank 0 logs tables
        is_rank0 = getattr(self.trainer, "is_global_zero", True)
        if not is_rank0 or len(all_results) == 0:
            return

        # W&B table logging
        if isinstance(self.logger, WandbLogger):
            run = getattr(self.logger, "experiment", None)  # wandb.run
            try:
                import wandb
            except Exception:
                run = None  # silently skip if wandb not installed

            if run is not None:
                import pandas as pd
                df = pd.DataFrame(all_results)
                table = wandb.Table(dataframe=df)
                run.log(
                    {f"{regime}_marker_predictions_epoch_{self.current_epoch}": table},
                    step=self.global_step,
                    commit=True
                )
                logger.info(f"Logged {len(all_results)} {regime} marker predictions (epoch {self.current_epoch})")

    def log_segtype_metrics(self, regime: str):
        # nothing collected
        if not self.st_ap_targets:
            return

        use_stateful = bool(getattr(self, "use_stateful", False))

        # compute only on rank0
        if not getattr(self.trainer, "is_global_zero", True):
            self.st_ap_logits.clear()
            self.st_ap_targets.clear()
            if hasattr(self, "st_eos_pos"):
                self.st_eos_pos.clear()
            return

        if use_stateful:
            # stateful inference: one prediction per block (last token), no EoS mask or context slicing
            metrics = self.seg_eval.compute(
                self.st_ap_targets,  # list of tensors
                self.st_ap_logits,  # list of tensors
                regime=regime,
                prefix="seg_type",
                eval_ctx_len=None,  # no per-position evaluation: one prediction per block
                eos_pos=None,       # no EoS selection: predictions are already per block
                ctx_len=None,
                logger_experiment=getattr(self.logger, "experiment", None),
            )
        else:
            # normal non-stateful evaluation (timestep or channel mode)
            metrics = self.seg_eval.compute(
                self.st_ap_targets,
                self.st_ap_logits,
                regime=regime,
                prefix="seg_type",
                eval_ctx_len=getattr(self.args, "eval_ctx_len", None),
                logger_experiment=getattr(self.logger, "experiment", None),
                eos_pos=self.st_eos_pos,
                ctx_len=self.ds_view.cfg["seq_len"],
            )

        # clear buffers
        self.st_ap_logits.clear()
        self.st_ap_targets.clear()
        if hasattr(self, "st_eos_pos"):
            self.st_eos_pos.clear()

        for k, v in metrics.items():
            self.log(k, v, sync_dist=False)

    def log_dws_metrics(self, regime: str):
        tasks = sorted(set(self.dws_targets.keys()) & set(self.dws_logits.keys()))
        for task in tasks:
            if not self.dws_targets[task] or not self.dws_logits[task]:
                continue

            # compute is called on all ranks, since it gathers the predictions across ranks
            metrics = self.seg_eval.compute(
                self.dws_targets[task],
                self.dws_logits[task],
                regime=regime,
                prefix=task,
                eval_ctx_len=getattr(self.args, "eval_ctx_len", None),
                logger_experiment=getattr(self.logger, "experiment", None),
                is_global_zero=getattr(self.trainer, "is_global_zero", True),
                eos_pos=self.dws_eos_pos[task],
                ctx_len=self.ds_view.cfg['seq_len']
            )

            # clear the buffers on all ranks
            self.dws_logits[task].clear()
            self.dws_targets[task].clear()
            self.dws_eos_pos[task].clear()

            # metrics are only returned on rank 0 ({} on the other ranks)
            for k, v in metrics.items():
                self.log(k, v, sync_dist=False)

    def log_MCR(self, regime: str):
        """Log the marker coverage ratio: reconstructed markers / expected markers, summed over all ranks."""
        exp = torch.tensor(float(self.total_expected_markers), device=self.device if hasattr(self, "device") else "cpu")
        pred = torch.tensor(float(self.total_predicted_markers), device=exp.device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exp, op=dist.ReduceOp.SUM)
            dist.all_reduce(pred, op=dist.ReduceOp.SUM)

        # reset local counters
        self.total_predicted_markers = 0
        self.total_expected_markers = 0

        is_rank0 = getattr(self.trainer, "is_global_zero", True)
        if not is_rank0:
            return

        mcr = (pred / exp).item() if exp.item() > 0 else 0.0
        self.log(f"{regime}/marker_coverage_ratio", mcr, batch_size=1, sync_dist=False)

    def log_metrics(self, label, value, prog_bar: bool = False):
        """Log a scalar metric (converted to a tensor on the model device), synchronised across ranks."""
        if not self.metrics_logging_active:
            return

        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        else:
            value = value.to(self.device)

        self.log(
            label, value,
            prog_bar=prog_bar,
            sync_dist=True,
            batch_size=getattr(self.args, "batch_size", None)
        )

    def on_fit_start(self):
        logger.debug("<<< Lightning - on_fit_start >>>")
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "ds_view", None) is not None:
            self.ds_view = dm.ds_view

        # build a small batch without DataLoader workers
        ds = dm.train_ds if hasattr(dm, "train_ds") else None
        assert ds is not None, "DataModule must set self.train_ds in setup('fit')"

        # take a few samples and collate with the same function the loader uses
        samples = [ds[i] for i in range(min(2, len(ds)))]
        batch = collate(samples)

        # move tensors to device
        def _to(x):
            return x.to(self.device) if isinstance(x, torch.Tensor) else x

        batch = torch.utils._pytree.tree_map_only(torch.Tensor, _to, batch)

        # forward pass to materialize lazily created modules
        self.eval()
        with torch.no_grad():
            _ = self.model(batch)
        self.train()

        # set runtime flags (again)
        self.set_runtime_flags()

        self._built = True

        self.metrics_logging_active = True

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_optimizer_params(self):
        if not self.trainer.optimizers:
            return 0
        opt = self.trainer.optimizers[0]
        return sum(p.numel() for group in opt.param_groups for p in group["params"])

    def on_test_start(self):
        logger.debug("<<< Lightning - on_test_start >>>")

        if getattr(self, "ds_view", None) is None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and getattr(dm, "ds_view", None) is not None:
                self.ds_view = dm.ds_view

        self.metrics_logging_active = True

        if self.config.get("sigenc_stem_adnorm", False):
            self.model.mm_tokenizer.encoders.get("signals").set_adaptive_bn()

    def on_train_epoch_start(self):
        logger.debug("<<< Lightning - on_train_epoch_start >>>")
        dm = self.trainer.datamodule
        if hasattr(dm, "train_key_sampler") and hasattr(dm.train_key_sampler, "set_epoch"):
            dm.train_key_sampler.set_epoch(int(self.current_epoch))

        # priority sampling: only if enabled and the view exposes the method
        if self.args.priority_sampling:
            dv = getattr(dm, "ds_view", None) if self.ds_view is None else self.ds_view
            if hasattr(dv, "priority_sample"):
                dv.priority_sample()

        # random subsampling: only if enabled and the view exposes the method
        if self.args.random_subsampling:
            dv = getattr(dm, "ds_view", None) if self.ds_view is None else self.ds_view
            if hasattr(dv, "random_subsample"):
                dv.random_subsample()

        # apply masking schedule
        if self.mask_sch_enabled:
            epoch = int(self.current_epoch)
            mmt = self.model.mm_tokenizer
            for k in self.mask_keys:
                p = self._mask_prob_epoch(epoch, self.mask_probs[k])
                p = max(0.0, min(1.0, p))
                mmt.config[k] = float(p)
                logger.info(f"[on_train_epoch_start] masking probability {k}={p:.2f}")

    def on_validation_epoch_end(self):
        logger.debug("<<< Lightning - on_validation_epoch_end >>>")
        if self.config.get('decode_markers', False):
            self.log_marker_predictions("val")
        self.log_segtype_metrics("val")
        self.log_MCR("val")
        if self.args.mode == "finetune":
            self.log_dws_metrics("val")

        if (self.args.dump_every > 0) and (self.current_epoch + 1) % self.args.dump_every == 0 and getattr(self.trainer, "is_global_zero", True):
            ckpt_dir = f"{self.args.save_path}/checkpoints/{self.args.run_name}_{self.args.run_uuid}"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_file = f"{ckpt_dir}/epoch_{int(self.current_epoch + 1)}.ckpt"
            self.trainer.save_checkpoint(ckpt_file)
            logger.info(f"saved validation checkpoint for epoch {int(self.current_epoch)} -> {ckpt_file}")

    def on_test_epoch_end(self):
        logger.debug("<<< Lightning - on_test_epoch_end >>>")
        if self.config.get('decode_markers', False):
            self.log_marker_predictions("test")

        if self.args.mode == "finetune":
            self.log_dws_metrics("test")
        else:
            self.log_segtype_metrics("test")

        self.log_MCR("test")

    def on_train_start(self):
        logger.debug("<<< Lightning - on_train_start >>>")
        self.model.set_frozen(self.args.freeze_all)
        if not self.args.freeze_all:
            self.model.mm_tokenizer.set_frozen(self.args.freeze_mmt)
            self.model.ntp.set_frozen(self.args.freeze_ntp)
            self.model.set_downstream_heads_frozen(self.args.freeze_head)
            if hasattr(self.model, "multitask_head"):
                self.model.multitask_head.set_frozen(self.args.freeze_head)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        logger.debug("<<< Lightning - on_save_checkpoint >>>")
        # 1) export all encoders (families & ids) in a nested block
        enc = getattr(self.model, "encoders", None) or getattr(self.model, "mm_tokenizer",
                                                               None) and self.model.mm_tokenizer.encoders
        if enc is not None:
            # parameters per family/id
            checkpoint["mmt_registry_states"] = enc.export_states()  # {family: {id: state_dict}}

            # known encoder ids per family (informational)
            checkpoint.setdefault("meta", {})["mmt_registry_known_ids"] = {
                fam: enc.family(fam).known_ids() for fam in enc.families()
            }

        # 2) model metadata and registry format version
        checkpoint.setdefault("meta", {}).update({
            "neuroNTP": getattr(self.model, "export_meta", lambda: {})(),
            "mmt_registry_format_version": 1,
        })

    def set_runtime_flags(self):
        if hasattr(self.model, "mm_tokenizer"):
            self.model.mm_tokenizer.token_masking = self.args.token_masking
        if hasattr(self.model, "set_frozen"):
            self.model.set_frozen(self.args.freeze_all)

        if not self.args.freeze_all:
            if hasattr(self.model, "mm_tokenizer"):
                self.model.mm_tokenizer.set_frozen(self.args.freeze_mmt)
            if hasattr(self.model, "ntp"):
                self.model.ntp.set_frozen(self.args.freeze_ntp)
            if hasattr(self.model, "set_downstream_heads_frozen"):
                self.model.set_downstream_heads_frozen(self.args.freeze_head)
            if hasattr(self.model, "multitask_head"):
                self.model.multitask_head.set_frozen(self.args.freeze_head)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        logger.debug("<<< Lightning - on_load_checkpoint >>>")
        # 1) rebuild registry structure first (so modules get consistent names/slots)
        enc = getattr(self.model, "encoders", None) or getattr(self.model, "mm_tokenizer",
                                                               None) and self.model.mm_tokenizer.encoders
        if enc is not None:
            # 2) import nested states; create missing modules lazily
            nested = checkpoint.get("mmt_registry_states", {})  # {family: {id: state_dict}}
            if nested:
                enc.import_states(nested, merge=True)  # keep existing, add new, overwrite overlaps

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx == 0 and not hasattr(self, "_report_done"):
            self._report_done = True
            target = getattr(self, "model", self)
            nnutils.print_param_tree(target, include_children_totals=True)
            nnutils.print_top_modules_by_params(target, topk=30)
            nnutils.find_unregistered_tensors(target)

    def setup(self, stage=None):
        logger.debug("<<< Lightning - setup >>>")
        self.metrics_logging_active = False
        dm = self.trainer.datamodule
        self.enc_warmup(stage, dm)

    def enc_warmup(self, stage: str, datamodule) -> None:
        logger.debug("<<< Lightning - enc_warmup >>>")
        if stage != "fit":
            return

        # fetch the unified registry
        regs = self.model.registries() if hasattr(self.model, "registries") else {}
        enc_reg = regs.get("mmt.encoders") or regs.get("encoders")
        if enc_reg is None:
            return

        # model configuration (from the model or the tokenizer)
        cfg = getattr(self.model, "config", None)
        if cfg is None and hasattr(self.model, "mm_tokenizer"):
            cfg = getattr(self.model.mm_tokenizer, "config", {})

        # gather all channel ids from the DataModule
        channel_ids = list(datamodule.ds_view.norm_signal_channels)

        # 1) warm up per-channel signal encoders (one instance per channel id)
        if channel_ids:
            if cfg.get("sigenc_stem_warmup", True):
                enc_reg.warmup("signals", ["signals"])
            enc_reg.warmup("signal_channels", channel_ids)

        # 2) warm up subtype tables for bases present in the training channels
        #    (e.g. axis or band sub-vocabularies of ACC, PPG, EOG, EMG)
        present_bases = sorted({
            parse_channel_base_and_sub(ch)[0].name.lower()
            for ch in channel_ids
            if parse_channel_base_and_sub(ch)[0] in SUBTYPE_VOCABS
        })
        for b in present_bases:
            _ = enc_reg.get(f"ch_subtype_{b}")  # creates on the right device

        # 3) singletons / per-family encoders gated by config
        if cfg:
            if cfg.get("add_eeg_location", False):
                _ = enc_reg.get("eeg_location")

            if cfg.get("add_markers", False) or cfg.get("decode_markers", False):
                _ = enc_reg.get("markers")

            if cfg.get("add_notes", False) or cfg.get("decode_notes", False):
                _ = enc_reg.get("notes")

            if cfg.get("add_gender", False):
                _ = enc_reg.get("gender")

            if cfg.get("add_age", False):
                _ = enc_reg.get("age")

            if cfg.get("add_hod", False):
                _ = enc_reg.get("hod")

            if cfg.get("add_segtype", False) or cfg.get("predict_segtype", False):
                _ = enc_reg.get("segtype")

            if cfg.get("add_tps", False) or cfg.get("predict_tps", False):
                _ = enc_reg.get("tps")

            if cfg.get("predict_tts", False):
                _ = enc_reg.get("tts")

            if cfg.get("add_end_of_segment", False):
                _ = enc_reg.get("eos")

            # 4) mask tokens for families actually used
            used_families = {"signals"}
            if cfg.get("add_markers", False) or cfg.get("decode_markers", False):
                used_families.add("markers")
            if cfg.get("add_notes", False) or cfg.get("decode_notes", False):
                used_families.add("notes")
            if cfg.get("add_tps", False):
                used_families.add("tps")
            if cfg.get("add_segtype", False):
                used_families.add("segtype")
            if cfg.get("add_gender", False):
                used_families.add("gender")
            if cfg.get("add_age", False):
                used_families.add("age")
            if cfg.get("add_hod", False):
                used_families.add("hod")

            # warmup family "mask_tokens" with ids = family prefixes
            enc_reg.warmup("mask_tokens", sorted(used_families))

    # non-strict loading: checkpoints may differ in the encoder registry (channels, sensors)
    def load_state_dict(self, state_dict, strict: bool = False, **kwargs):  # default to non-strict
        return super().load_state_dict(state_dict, strict=False)

    def _mask_prob_epoch(self, epoch: int, p0: float) -> float:
        """Masking probability of an epoch: p0 until the warmup epoch, then linear to mask_sch_p1 at mask_sch_target."""
        p1 = self.mask_sch_p1
        w = int(self.mask_sch_warmup)
        e_end = int(self.mask_sch_target)

        if epoch < w:
            return p0

        # ramp from epoch=w to epoch=e_end
        ramp_len = max(1, e_end - w)
        t = min(1.0, (epoch - w) / ramp_len)
        return float(p0 + t * (p1 - p0))

    def configure_optimizers(self):
        logger.info("optimizer: lr=%s, weight_decay=%s, scheduler=%s",
                    self.args.lr, self.args.weight_decay, self.args.lr_scheduler)
        self.set_runtime_flags()

        if self.args.lr_scheduler is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            return optimizer

        elif self.args.lr_scheduler == 'cos_warmup':
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(self.args.lr_warmup * total_steps)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

            # cosine LR scheduler with warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            logger.info("cosine schedule: %d warmup steps of %d", warmup_steps, total_steps)

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        else:
            raise AttributeError(f"Scheduler {self.args.lr_scheduler} not supported!")

    def _pick_seizure_logits(self, ntp_seq, mmt_seq, dws_pred):
        targets = mmt_seq.seg_types[:, 1:]

        # prefer downstream head if present (finetune)
        if isinstance(dws_pred, dict) and dws_pred:
            # take the first (and only) downstream task
            task, logits = next(iter(dws_pred.items()))
            if logits is not None:
                self._seizure_logits_source = f"dws:{task}"
                return logits, targets

        # else use pretrained segtype head if enabled
        if getattr(ntp_seq, "seg_types", None) is not None:
            self._seizure_logits_source = "aux"
            return ntp_seq.seg_types, targets

        self._seizure_logits_source = "none"
        return None, targets

    def _get_stream_key(self, metadata_list):
        # batch_size=1 in continuous mode -> metadata_list[0]
        md = metadata_list[0]
        return (md.get("patient"), md.get("filename"))


def pretrain(args, test_only=False):
    # console message logger to use
    logger = logging.getLogger()

    # create datamodule
    pretraining_datamodule = PretrainingDataModule(args)

    # instantiate the Neuro-NTP model
    model = NeuroNTP(config=args.config)
    model.mm_tokenizer.token_masking = args.token_masking

    # lightning model
    lightning_model = PretrainingLightningModel(args, model)

    # logger and callbacks
    version = f"{args.run_name}-{args.dataset}-{args.lr}-{args.batch_size}"
    if args.wandb:
        tr_logger = WandbLogger(
            save_dir=f"{args.save_path}/logs/{args.run_name}_{args.run_uuid}",
            # version=version,
            name=version + f"_{np.random.randint(1000)}",
            settings=dict(_disable_stats=True),
        )
        tr_logger.log_hyperparams(vars(args))
    else:
        tr_logger = False

    modsel_mon, modsel_mode = args.model_select.split("|")
    logger.info(f"...model selection criterion: ${modsel_mode}(${modsel_mon})")
    checkpoint_cb = ModelCheckpoint(
        monitor=modsel_mon,
        mode=modsel_mode,
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}",
    ) if not test_only else None

    progress_cb = TQDMProgressBar(refresh_rate=10)
    trainer = pl.Trainer(
        devices=[int(d) for d in args.device.split(",")] if args.device != "cpu" else 1,
        accelerator="gpu" if args.device != "cpu" else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed" if args.fp16 and torch.cuda.is_bf16_supported() else ("16-mixed" if args.fp16 else 32),
        benchmark=True,
        enable_checkpointing=not test_only,
        logger=tr_logger,
        log_every_n_steps=50,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, progress_cb] if not test_only else [progress_cb],
        gradient_clip_val=0.75,
        sync_batchnorm=True,
        accumulate_grad_batches=2,
        reload_dataloaders_every_n_epochs=1
    )

    # restoring a saved model
    if args.ckpt_path:
        nnutils.restore_from_ckpt(args.ckpt_path, pretraining_datamodule, lightning_model)

    # train the model
    if not test_only:
        logger.info("Starting pre-training...")
        try:
            trainer.fit(lightning_model, datamodule=pretraining_datamodule)
            logger.info("Finished pre-training...")
        except KeyboardInterrupt:
            logger.info("Aborting pre-training - keyboard interrupt...")
    else:
        logger.info("Start evaluation on test set...")
        trainer.test(model=lightning_model, ckpt_path=None, datamodule=pretraining_datamodule)[0]
        logger.info(f'seizure logits source used in test: {getattr(lightning_model, "_seizure_logits_source", "unknown")}')
        logger.info("Finished evaluation on test set...")


