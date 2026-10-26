"""Contrastive loss for the identity embeddings (consistent training).

Adapted from CTVIS (ct_cl_plugin.py, multi_ref_cl_plugin.py), which follows the
loss formulation of IDOL:

  loss_reid:     log(1 + sum_{pos, neg} exp(<a, neg> - <a, pos>)), computed with
                 logsumexp over all (negative - positive) margins of an anchor.
                 Operates on raw dot products (no temperature).

  loss_aux_reid: squared error of the cosine similarity of each
                 anchor-candidate pair against 1 (positive) or 0 (negative).

Construction of the items (per clip of T frames):
  1. Hungarian matching of the segmentation loss assigns each labeled thing
     instance to one query in each frame.
  2. A ``TrainMemoryBank`` is reset and updated frame by frame over the clip
     with the matched query embeddings; after each frame the memory embedding
     of every instance is snapshotted.
  3. Every frame j takes a turn as anchor frame. For each matched thing
     instance at j:
       - anchor = embedding of its matched query at j
       - for each reference frame r != j where the instance is matched:
           * positive = memory-embedding snapshot at r (with probability
             ``positive_momentum_prob``) or the raw embedding at r (otherwise)
           * negatives = ``num_negatives`` other queries of frame r
       - for each reference frame r != j where the instance is absent:
           * ``num_negatives`` queries of r become negatives (with the default
             199 and Q = 200, one query is dropped at random)
       - one contrastive item per (anchor frame, instance), bundling all
         positives and all negatives of the reference frames.

Each item is weighted by the class weight of its anchor (both loss terms), and
each term is divided by the sum of the weights.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from panoctvps.tracker.memory_bank_train import TrainMemoryBank


def contrastive_loss(
    contrastive_items: List[Dict[str, torch.Tensor]],
    dummy_tensor: torch.Tensor,
    class_weights: Dict[int, float],
) -> Dict[str, torch.Tensor]:
    """Compute the contrastive losses from the contrastive items.

      * ``loss_reid``: logsumexp over ``(negative - positive)`` margin pairs.
      * ``loss_aux_reid``: squared error of the cosine similarity against the
        binary labels.

    Args:
        contrastive_items: list of dicts from
            :func:`build_multi_ref_contrastive_items`.
        dummy_tensor: any tensor on the correct device (for the zero loss
            when ``contrastive_items`` is empty).
        class_weights: ``{class_id: weight}``. Each item is multiplied by
            ``class_weights.get(item["anchor_class_id"], 1.0)`` and the
            denominator is the sum of the weights.

    Returns:
        Dict with ``loss_reid`` and ``loss_aux_reid`` scalar tensors.
    """
    if len(contrastive_items) == 0:
        zero = dummy_tensor.sum() * 0
        return {"loss_reid": zero, "loss_aux_reid": zero}

    total_contras = dummy_tensor.new_tensor(0.0)
    total_aux = dummy_tensor.new_tensor(0.0)
    total_weight = dummy_tensor.new_tensor(0.0)

    for item in contrastive_items:
        w = float(class_weights.get(item["anchor_class_id"], 1.0))
        if w <= 0.0:
            continue

        pred = item["dot_product"].permute(1, 0)  # (1, 1+num_neg)
        label = item["label"].unsqueeze(0)  # (1, 1+num_neg)

        pos_inds = label == 1
        neg_inds = label == 0

        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()

        # Mask out with inf to exclude from logsumexp
        pred_pos[neg_inds] = pred_pos[neg_inds] + float("inf")
        pred_neg[pos_inds] = pred_neg[pos_inds] + float("-inf")

        # Pairwise margin: (neg - pos) for all (neg, pos) pairs
        N = pred.shape[1]
        _pos_expand = torch.repeat_interleave(pred_pos, N, dim=1)
        _neg_expand = pred_neg.repeat(1, N)
        x = F.pad(_neg_expand - _pos_expand, (0, 1), "constant", 0)
        total_contras = total_contras + w * torch.logsumexp(x, dim=1).sum()

        # Auxiliary cosine similarity loss
        aux_pred = item["cosine_similarity"].permute(1, 0)  # (1, 1+num_neg)
        aux_label = item["label"].unsqueeze(0).float()
        total_aux = total_aux + w * (torch.abs(aux_pred - aux_label) ** 2).mean()

        total_weight = total_weight + w

    denom = total_weight.clamp(min=1e-8)
    return {
        "loss_reid": total_contras / denom,
        "loss_aux_reid": total_aux / denom,
    }


def _sample_negatives(
    Q: int,
    exclude_qi: Optional[int],
    num_negatives: int,
    rng: random.Random,
) -> List[int]:
    """Pick negative query indices from a frame.

    Args:
        Q: total query count for the frame.
        exclude_qi: a query index to exclude (the positive's matched query at
            this reference frame). If ``None``, all ``Q`` queries are eligible.
        num_negatives: requested count. When ``>=`` the candidate pool size,
            all candidates are returned in sorted order (no sampling).
        rng: per-call ``random.Random`` instance for reproducibility.
    """
    pool = [i for i in range(Q) if i != exclude_qi]
    if len(pool) <= num_negatives:
        return pool
    return sorted(rng.sample(pool, num_negatives))


def build_multi_ref_contrastive_items(
    reid_embeddings: torch.Tensor,
    matching_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    global_instance_ids: List[torch.Tensor],
    class_labels: List[torch.Tensor],
    clip_length: int,
    batch_size: int,
    train_memory_bank: TrainMemoryBank,
    num_negatives: int = 199,
    positive_momentum_prob: float = 0.5,
    rng_seed: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Build the multi-reference contrastive items with memory-bank positives.

    For each clip in the batch:

      1. ``train_memory_bank`` is ``reset()`` and then updated temporally: for
         each frame ``t``, every matched thing instance contributes its raw
         embedding to the bank. After each frame's updates, the memory
         embedding of every instance is snapshotted.
      2. For each anchor frame ``j`` (every frame of the clip) and each
         matched thing instance at ``j``, a single contrastive item is built.
         The item bundles:

           - ``num_positive`` positives, one per reference frame ``r != j``
             where the instance is matched (memory-embedding snapshot with
             probability ``positive_momentum_prob``, raw embedding otherwise);
           - ``num_negatives`` negatives per such reference frame, plus
             ``num_negatives`` queries from each reference frame where the
             instance is absent (199 of the 200 at the defaults; one dropped at
             random).

      3. Items with zero positives are dropped.

    Args:
        reid_embeddings: ``(B*T, Q, C)`` embeddings of the embedding head.
        matching_indices: ``B*T`` tuples ``(src, tgt)`` from the Hungarian
            matching of the segmentation loss.
        global_instance_ids: ``B*T`` long tensors ``(N_t,)`` with global IDs
            (``-1`` for stuff segments).
        class_labels: ``B*T`` long tensors ``(N_t,)`` with class IDs per GT
            target; stored on each item for the class weighting.
        clip_length: ``T``.
        batch_size: ``B``.
        train_memory_bank: the ``TrainMemoryBank``, reset per clip.
        num_negatives: per-reference-frame negative count. Default 199
            (all non-anchor queries at ``Q=200``).
        positive_momentum_prob: probability of using the memory embedding
            instead of the raw embedding as the positive of an
            ``(anchor, ref)`` pair. Default 0.5.
        rng_seed: optional seed for the ``random.Random`` used for sampling.
            Default ``None`` (non-deterministic).

    Returns:
        List of contrastive items, each containing:

            ``dot_product``: ``(num_positive + num_negatives_total, 1)``
                raw dot products of anchor vs (pos / neg) embeddings.
            ``cosine_similarity``: ``(num_positive + num_negatives_total, 1)``
                normalised similarities.
            ``label``: ``(num_positive + num_negatives_total,)`` long, with
                the first ``num_positive`` entries set to 1.
            ``anchor_class_id``: ``int`` — class id of the anchor's GT target.
            ``num_positive``: ``int``.
    """
    Q = reid_embeddings.shape[1]
    rng = random.Random(rng_seed)
    contrastive_items: List[Dict[str, torch.Tensor]] = []

    for b in range(batch_size):
        # ------------------------------------------------------------------
        # 1. Per-frame gid → query mapping; bank build-up + sg snapshots.
        # ------------------------------------------------------------------
        frame_gid_to_qi: List[Dict[int, int]] = []
        frame_gid_to_class: List[Dict[int, int]] = []
        sg_snapshots: List[Dict[int, torch.Tensor]] = []

        train_memory_bank.reset()

        for t in range(clip_length):
            f_idx = b * clip_length + t
            src, tgt = matching_indices[f_idx]
            gids_t = global_instance_ids[f_idx]
            classes_t = class_labels[f_idx]
            embeds_t = reid_embeddings[f_idx]  # (Q, C)

            gid_to_qi: Dict[int, int] = {}
            gid_to_cls: Dict[int, int] = {}
            for k in range(len(tgt)):
                gid = gids_t[tgt[k]].item()
                if gid <= 0:
                    continue
                qi = src[k].item()
                gid_to_qi[gid] = qi
                gid_to_cls[gid] = int(classes_t[tgt[k]].item())
            frame_gid_to_qi.append(gid_to_qi)
            frame_gid_to_class.append(gid_to_cls)

            # Bank update + snapshot after this frame.
            snap: Dict[int, torch.Tensor] = {}
            for gid, qi in gid_to_qi.items():
                train_memory_bank.update(gid, embeds_t[qi])
            # Snapshot the current memory embedding of every seeded instance.
            for gid in train_memory_bank.gids:
                sg = train_memory_bank.get_sg_embed(gid)
                if sg is not None:
                    snap[gid] = sg.detach().clone()
            sg_snapshots.append(snap)

        # ------------------------------------------------------------------
        # 2. Generate contrastive items per anchor frame, per instance.
        # ------------------------------------------------------------------
        for j in range(clip_length):
            anchor_embeds_j = reid_embeddings[b * clip_length + j]  # (Q, C)
            for gid, anchor_qi in frame_gid_to_qi[j].items():
                anchor_embed = anchor_embeds_j[anchor_qi:anchor_qi + 1]  # (1, C)
                anchor_cls = frame_gid_to_class[j][gid]

                positive_list: List[torch.Tensor] = []
                negative_list: List[torch.Tensor] = []

                for r in range(clip_length):
                    if r == j:
                        continue
                    ref_embeds_r = reid_embeddings[b * clip_length + r]  # (Q, C)

                    if gid in frame_gid_to_qi[r]:
                        pos_qi = frame_gid_to_qi[r][gid]
                        # Pick positive source.
                        use_sg = (
                            gid in sg_snapshots[r]
                            and rng.random() < positive_momentum_prob
                        )
                        if use_sg:
                            pos = sg_snapshots[r][gid].to(
                                dtype=anchor_embed.dtype,
                                device=anchor_embed.device,
                            )
                        else:
                            pos = ref_embeds_r[pos_qi:pos_qi + 1]
                        positive_list.append(pos)

                        neg_idx = _sample_negatives(Q, pos_qi, num_negatives, rng)
                        negative_list.append(ref_embeds_r[neg_idx])
                    else:
                        # Instance absent at r → num_negatives queries of r become negatives
                        # (199 of 200 at the defaults: one is dropped at random).
                        neg_idx = _sample_negatives(Q, None, num_negatives, rng)
                        negative_list.append(ref_embeds_r[neg_idx])

                if not positive_list:
                    continue

                pos_neg_embed = torch.cat(positive_list + negative_list, dim=0)
                num_positive = len(positive_list)
                label = pos_neg_embed.new_zeros(
                    pos_neg_embed.shape[0], dtype=torch.int64,
                )
                label[:num_positive] = 1

                dot_product = torch.einsum(
                    "nc,kc->nk", pos_neg_embed, anchor_embed,
                )  # (N, 1)
                norm_pn = F.normalize(pos_neg_embed, dim=1)
                norm_anchor = F.normalize(anchor_embed, dim=1)
                cosine_sim = torch.einsum(
                    "nc,kc->nk", norm_pn, norm_anchor,
                )

                contrastive_items.append({
                    "dot_product": dot_product,
                    "cosine_similarity": cosine_sim,
                    "label": label,
                    "anchor_class_id": anchor_cls,
                    "num_positive": num_positive,
                })

    return contrastive_items
