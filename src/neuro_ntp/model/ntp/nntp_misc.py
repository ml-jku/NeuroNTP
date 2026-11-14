"""
+----------------------------------------------------------------------------------------------------------------------+
nntp_misc.py
Loss functions, batch collation and marker retrieval for NeuroNTP

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
        QuantileLoss           pinball loss for a single quantile (next-token and marker embedding regression)
        FocalLoss              binary focal loss on logits (segment type, downstream heads)
        pretrain_collate_fn    collates EpiSelfSupervisedSample items into a self-supervised batch
        find_closest_markers   nearest clinical markers of predicted marker embeddings (cosine similarity)
+----------------------------------------------------------------------------------------------------------------------+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.maximum(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def pretrain_collate_fn(batch):
    signal_list = []
    chunks_list = []
    metadata_list = []
    marker_embs_list = []
    notes_embs_list = []
    has_marker = False
    has_notes = False

    for sample in batch:
        signal, chunks, marker_embs, notes_embs, metadata = sample.as_tuple()
        # signal: (C, T, L)
        signal_list.append(signal)
        chunks_list.append(chunks)
        metadata_list.append(metadata)

        if marker_embs is not None:
            has_marker = True
        marker_embs_list.append(marker_embs)

        if notes_embs is not None:
            has_notes = True
        notes_embs_list.append(notes_embs)

    # -> (B, C, T, L)
    signals = torch.stack(signal_list, dim=0)
    # -> (B, T, K)
    chunks = np.stack(chunks_list, axis=0)

    if has_marker:
        marker_embs = torch.stack(
            [m if m is not None else torch.zeros_like(marker_embs_list[0]) for m in marker_embs_list],
            dim=0,
        )
    else:
        marker_embs = None

    if has_notes:
        notes_embs = torch.stack(
            [n if n is not None else torch.zeros_like(notes_embs_list[0]) for n in notes_embs_list],
            dim=0,
        )
    else:
        notes_embs = None

    return signals, chunks, marker_embs, notes_embs, metadata_list


def find_closest_markers(embedding, marker_emb_dict, topk=None):
    # convert dictionary to tensor for batch computation
    marker_words = list(marker_emb_dict.keys())
    marker_embeddings = torch.stack(list(marker_emb_dict.values()))
    ntp_seq_markers = F.normalize(embedding, p=2, dim=-1)
    marker_embeddings = F.normalize(marker_embeddings, p=2, dim=-1)
    similarity_matrix = torch.mm(ntp_seq_markers.to('cpu'), marker_embeddings.T)

    if topk is None or topk == 1:
        closest_indices = similarity_matrix.argmax(dim=-1)
        closest_markers = [marker_words[i] for i in closest_indices.tolist()]
    else:
        topk_vals, topk_indices = similarity_matrix.topk(topk, dim=-1)
        closest_markers = [", ".join(list(set(map(str.strip, (','.join([marker_words[idx] for idx in row])).split(","))))) for row in topk_indices.tolist()]
    return closest_markers
