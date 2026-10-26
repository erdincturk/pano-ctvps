"""Mask2Former wrapper for WOD PVPS.

Builds HuggingFace ``Mask2FormerForUniversalSegmentation`` from a
panoptic-pretrained checkpoint and replaces its classification head. The
head has ``num_classes + 1`` outputs: the 29 WOD class ids (0-28, where
0 is the undefined class) plus the no-object class.

Usage:
    from panoctvps.segmenter.mask2former import build_model
    model = build_model(backbone="swin-large-mapillary")  # the paper's model
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES

# HuggingFace pretrained checkpoints by backbone name.
#   swin-large-mapillary  Swin-L, Mapillary Vistas panoptic (the paper's model)
#   swin-large            Swin-L, Cityscapes panoptic (pre-training comparison)
#   swin-tiny             Swin-T, Cityscapes panoptic (small model for the tests)
_PRETRAINED = {
    "swin-large-mapillary": "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
    "swin-large": "facebook/mask2former-swin-large-cityscapes-panoptic",
    "swin-tiny":  "facebook/mask2former-swin-tiny-cityscapes-panoptic",
}


def build_model(
    backbone: str = "swin-large-mapillary",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    """Build Mask2Former adapted for WOD PVPS.

    Args:
        backbone:        a key of ``_PRETRAINED``.
        num_classes:     number of class ids (29 for WOD PVPS: 0-28).
        pretrained:      load the pretrained weights of ``backbone``; if False,
                         only its configuration is used (weights are random
                         until a checkpoint is loaded).
        pretrained_path: local checkpoint path; if None, download from HF hub.

    Returns:
        nn.Module — HF Mask2FormerForUniversalSegmentation adapted for WOD.
    """
    from transformers import Mask2FormerForUniversalSegmentation, logging as hf_log
    hf_log.set_verbosity_error()  # suppress verbose loading output

    hub_id = pretrained_path or _PRETRAINED.get(backbone)
    if hub_id is None:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_PRETRAINED)}")

    if pretrained:
        # Load with ignore_mismatched_sizes=True so HF reinitialises the
        # class_predictor head while keeping all backbone/decoder weights.
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            hub_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        from transformers import Mask2FormerConfig
        config = Mask2FormerConfig.from_pretrained(hub_id, num_labels=num_classes)
        model = Mask2FormerForUniversalSegmentation(config)

    return model


def freeze_backbone(model: nn.Module) -> int:
    """Freeze all Swin backbone parameters. Returns count of frozen params."""
    frozen = 0
    for name, param in model.named_parameters():
        if "model.pixel_level_module.encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str = "cpu") -> int:
    """Load a training checkpoint saved by scripts/train_segmenter.py.

    Returns the iteration number saved in the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("iteration", 0)


def extract_query_embeddings(model: nn.Module, outputs) -> torch.Tensor:
    """Return the decoder query states after the decoder's final LayerNorm.

    HuggingFace exposes the decoder output before its final LayerNorm as
    ``transformer_decoder_last_hidden_state``; this function applies
    ``model.model.transformer_module.decoder.layernorm`` to it, giving the
    query state that the Mask2Former class and mask heads consume.

    Args:
        model: HF ``Mask2FormerForUniversalSegmentation`` instance.
        outputs: model forward outputs.

    Returns:
        ``(B, Q, C)`` query-embedding tensor for the embedding head / tracker.
    """
    query_embeds = outputs.transformer_decoder_last_hidden_state
    return model.model.transformer_module.decoder.layernorm(query_embeds)


@torch.no_grad()
def predict_panoptic(
    model: nn.Module,
    pixel_values: torch.Tensor,
    threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> Dict:
    """Run inference and return panoptic predictions.

    Args:
        model:       trained Mask2Former model.
        pixel_values: (B, 3, H, W) float32 tensor, ImageNet normalised.
        threshold:   confidence threshold for keeping a segment.
        overlap_mask_area_threshold: max fraction of a mask that can overlap
                     with a higher-scoring mask before being suppressed.

    Returns:
        List of dicts (one per image in the batch), each with:
            "semantic":  (H, W) int32 — class IDs
            "instance":  (H, W) int32 — instance IDs (0 = stuff or background)
    """
    from transformers import Mask2FormerImageProcessor
    model.eval()
    outputs = model(pixel_values=pixel_values)

    # Use the HF processor's post-processing for panoptic output
    processor = Mask2FormerImageProcessor(
        ignore_index=0,
        do_rescale=False,
        do_normalize=False,
    )
    target_sizes = [(pixel_values.shape[-2], pixel_values.shape[-1])] * pixel_values.shape[0]
    panoptic_results = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=threshold,
        overlap_mask_area_threshold=overlap_mask_area_threshold,
        target_sizes=target_sizes,
    )

    results = []
    for pr in panoptic_results:
        seg_map  = pr["segmentation"].cpu().numpy()  # (H, W) int, segment IDs
        seg_info = pr["segments_info"]

        semantic     = np.zeros_like(seg_map, dtype=np.int32)
        instance     = np.zeros_like(seg_map, dtype=np.int32)
        inst_counter = 1

        for seg in seg_info:
            sid      = seg["id"]
            cls      = seg["label_id"]
            is_thing = cls in THING_CLASSES

            mask = seg_map == sid
            semantic[mask] = cls
            if is_thing:
                instance[mask] = inst_counter
                inst_counter += 1

        results.append({"semantic": semantic, "instance": instance})

    return results
