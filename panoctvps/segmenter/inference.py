"""Panoptic inference with learned identity embeddings.

Prediction path of the segmenter plus embedding head, used by
`scripts/build_cache.py`, `scripts/build_cache_gt.py` and the in-training
evaluation of `scripts/train_tracker.py`.
"""

from typing import Dict, List

import numpy as np
import torch

from panoctvps.data.constants import THING_CLASSES
from panoctvps.segmenter.mask2former import extract_query_embeddings


@torch.no_grad()
def predict_panoptic_with_learned_embeddings(
    model,
    embed_head,
    pixel_values: torch.Tensor,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    return_raw: bool = False,
) -> List[Dict]:
    """Run M2F + embedding head inference, returning panoptic + identity embeddings.

    For each thing instance in the panoptic output, the matching M2F query
    is found by mask IoU and its learned (post-MLP) embedding, class score,
    and class id are returned alongside the segmentation.

    Args:
        model: HF Mask2Former model (frozen at inference time).
        embed_head: EmbeddingHead consuming the decoder query states
            (after the decoder's final LayerNorm, see
            ``extract_query_embeddings``).
        pixel_values: (B, 3, H, W) input batch.
        threshold: M2F panoptic class-probability cutoff (HF default 0.5).
        mask_threshold: mask-sigmoid binarisation cutoff (HF default 0.5).
        overlap_mask_area_threshold: panoptic overlap-resolution gate
            (HF default 0.8).
        return_raw: if True, also include in each result dict:
            frame_queries: (Q, C) float32 ndarray — decoder query states fed
                to the embedding head
            frame_class_logits: (Q, num_classes+1) float32 ndarray
            instance_query_idx: dict[inst_id, int] — which query drove each instance

    Returns:
        Per-image dicts with:
            semantic: (H, W) int32 — class id per pixel
            instance: (H, W) int32 — per-frame instance id (1..N)
            instance_embeddings: dict[int, ndarray] — learned 256-d embedding
            instance_scores: dict[int, float] — M2F class softmax score
            instance_classes: dict[int, int] — class id
            [if return_raw] frame_queries, frame_class_logits, instance_query_idx
    """
    from transformers import Mask2FormerImageProcessor

    model.eval()
    embed_head.eval()

    outputs = model(pixel_values=pixel_values)

    query_embeds = extract_query_embeddings(model, outputs)  # (B, Q, C)
    reid_embeds = embed_head(query_embeds)  # (B, Q, C)

    class_logits = outputs.class_queries_logits  # (B, Q, num_classes+1)
    mask_logits = outputs.masks_queries_logits   # (B, Q, H/4, W/4)

    processor = Mask2FormerImageProcessor(
        ignore_index=0, do_rescale=False, do_normalize=False,
    )
    target_sizes = [(pixel_values.shape[-2], pixel_values.shape[-1])] * pixel_values.shape[0]
    panoptic_results = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        overlap_mask_area_threshold=overlap_mask_area_threshold,
        target_sizes=target_sizes,
    )

    feat_h, feat_w = mask_logits.shape[-2], mask_logits.shape[-1]

    results = []
    for b, pr in enumerate(panoptic_results):
        seg_map = pr["segmentation"].cpu().numpy()
        seg_info = pr["segments_info"]

        semantic = np.zeros_like(seg_map, dtype=np.int32)
        instance = np.zeros_like(seg_map, dtype=np.int32)
        inst_embeddings: Dict[int, np.ndarray] = {}
        inst_scores: Dict[int, float] = {}
        inst_classes: Dict[int, int] = {}
        inst_query_idx: Dict[int, int] = {}
        inst_counter = 1

        q_masks = mask_logits[b].sigmoid().cpu().numpy()
        h, w = seg_map.shape
        row_idx = np.linspace(0, h - 1, feat_h).astype(np.int32)
        col_idx = np.linspace(0, w - 1, feat_w).astype(np.int32)
        seg_map_small = seg_map[np.ix_(row_idx, col_idx)]

        # Per-query class score (drop the no-object channel)
        scores = class_logits[b].softmax(-1)[:, :-1].max(-1)

        # Pre-binarise all query masks once for the IoU search below
        q_flat_all = (q_masks.reshape(q_masks.shape[0], -1) > 0.5).astype(np.float32)
        q_areas = q_flat_all.sum(axis=1)

        for seg in seg_info:
            sid = seg["id"]
            cls = seg["label_id"]
            is_thing = cls in THING_CLASSES

            seg_mask = seg_map == sid
            semantic[seg_mask] = cls

            if is_thing:
                instance[seg_mask] = inst_counter

                # Match segment to best query by mask IoU at downsampled resolution
                seg_mask_small = (seg_map_small == sid).ravel().astype(np.float32)
                seg_area = seg_mask_small.sum()
                if seg_area > 0:
                    inters = q_flat_all @ seg_mask_small
                    unions = seg_area + q_areas - inters
                    ious = inters / (unions + 1e-6)
                    best_qi = int(np.argmax(ious))
                    inst_embeddings[inst_counter] = (
                        reid_embeds[b, best_qi].cpu().numpy()
                    )
                    inst_scores[inst_counter] = scores[0][best_qi].item()
                    inst_classes[inst_counter] = cls
                    inst_query_idx[inst_counter] = best_qi

                inst_counter += 1

        result = {
            "semantic": semantic,
            "instance": instance,
            "instance_embeddings": inst_embeddings,
            "instance_scores": inst_scores,
            "instance_classes": inst_classes,
        }
        if return_raw:
            result["frame_queries"] = query_embeds[b].cpu().numpy().astype(np.float32)
            result["frame_class_logits"] = class_logits[b].cpu().numpy().astype(np.float32)
            result["instance_query_idx"] = inst_query_idx
        results.append(result)

    return results
