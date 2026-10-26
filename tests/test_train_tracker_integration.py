"""Integration tests for the tracker training pipeline (scripts/train_tracker.py).

Uses a small model (swin-tiny) and synthetic data. Covers:

  1. Model setup: backbone freezing, param groups
  2. Forward pass: segmentation loss + contrastive loss computation
  3. Hungarian matching interface
  4. Backward pass: gradient flow to decoder + embed_head, not backbone
  5. Checkpoint: save/load round-trip with all state
  6. PQ evaluation: evaluate_pq function
  7. Tracking evaluation: evaluate_tracking with the memory-bank tracker
  8. predict_panoptic_with_learned_embeddings: learned embedding swap
  9. Full loop: multiple iterations with checkpoint + eval JSON
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES, STUFF_CLASSES, IGNORE_LABEL
from panoctvps.segmenter.mask2former import build_model, freeze_backbone
from panoctvps.tracker.embedding_head import EmbeddingHead
from panoctvps.data.class_weights import WOD_THING_ANCHOR_WEIGHTS
from panoctvps.tracker.contrastive_loss import build_multi_ref_contrastive_items, contrastive_loss
from panoctvps.tracker.memory_bank_train import TrainMemoryBank
from panoctvps.data.video_dataset import video_collate_fn


def _cl_items(reid_embeds, indices, batch, clip_length, num_negatives=199):
    return build_multi_ref_contrastive_items(
        reid_embeds, indices, batch["global_instance_ids"], batch["class_labels"],
        clip_length=clip_length, batch_size=1, num_negatives=num_negatives,
        train_memory_bank=TrainMemoryBank(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model_and_head(device):
    """Build swin-tiny model + embed_head once for the whole module."""
    model = build_model(backbone="swin-tiny", num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    freeze_backbone(model)

    embed_head = EmbeddingHead(input_dim=256, hidden_dim=256, output_dim=256)
    embed_head = embed_head.to(device)

    return model, embed_head


@pytest.fixture
def output_dir():
    """Temporary directory for checkpoints, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="tracker_test_")
    yield d
    shutil.rmtree(d)


def _make_synthetic_batch(B=1, T=3, H=256, W=256, device="cpu"):
    """Create a synthetic video batch matching video_collate_fn output.

    Creates B clips of T frames each, with realistic panoptic segments:
    - road (class 20, stuff) covering bottom half
    - sky (class 25, stuff) covering top half
    - 2 car instances (class 2, thing) with global IDs consistent across frames
    """
    pixel_values_list = []
    mask_labels_all = []
    class_labels_all = []
    global_ids_all = []

    for b in range(B):
        frames_pv = []
        for t in range(T):
            # Image: random noise, ImageNet-normalized shape
            pv = torch.randn(3, H, W)
            frames_pv.append(pv)

            # Segments: road + sky + 2 cars
            masks = []
            labels = []
            gids = []

            # Road (stuff, bottom half)
            road_mask = torch.zeros(H, W)
            road_mask[H // 2:, :] = 1.0
            masks.append(road_mask)
            labels.append(20)
            gids.append(-1)

            # Sky (stuff, top quarter)
            sky_mask = torch.zeros(H, W)
            sky_mask[:H // 4, :] = 1.0
            masks.append(sky_mask)
            labels.append(25)
            gids.append(-1)

            # Car 1 (thing, global_id=101)
            car1_mask = torch.zeros(H, W)
            car1_mask[H // 4:H // 2, 10:60] = 1.0
            masks.append(car1_mask)
            labels.append(2)
            gids.append(101)

            # Car 2 (thing, global_id=102)
            car2_mask = torch.zeros(H, W)
            car2_mask[H // 4:H // 2, 100:150] = 1.0
            masks.append(car2_mask)
            labels.append(2)
            gids.append(102)

            mask_labels_all.append(torch.stack(masks))
            class_labels_all.append(torch.tensor(labels, dtype=torch.int64))
            global_ids_all.append(torch.tensor(gids, dtype=torch.int64))

        pixel_values_list.append(torch.stack(frames_pv))

    pixel_values = torch.cat(pixel_values_list, dim=0)  # (B*T, 3, H, W)

    return {
        "pixel_values": pixel_values.to(device),
        "mask_labels": mask_labels_all,
        "class_labels": class_labels_all,
        "global_instance_ids": global_ids_all,
        "clip_length": T,
        "batch_size": B,
        "meta": [{"segment_name": f"test_seg_{b}", "camera_id": 1,
                  "timestamps": list(range(T))} for b in range(B)],
    }


# ---------------------------------------------------------------------------
# Test 1: Model setup
# ---------------------------------------------------------------------------

class TestModelSetup:
    """Verify model construction, freezing, and param groups."""

    def test_backbone_frozen(self, model_and_head):
        model, _ = model_and_head
        for name, p in model.named_parameters():
            if "model.pixel_level_module.encoder" in name:
                assert not p.requires_grad, f"Backbone param {name} should be frozen"

    def test_decoder_trainable(self, model_and_head):
        model, _ = model_and_head
        decoder_trainable = []
        for name, p in model.named_parameters():
            if "model.pixel_level_module.encoder" not in name and p.requires_grad:
                decoder_trainable.append(name)
        assert len(decoder_trainable) > 0, "Decoder should have trainable params"

    def test_embed_head_trainable(self, model_and_head):
        _, embed_head = model_and_head
        for p in embed_head.parameters():
            assert p.requires_grad

    def test_separate_param_groups(self, model_and_head):
        """Optimizer should have separate LR groups for decoder and embed_head."""
        model, embed_head = model_and_head
        from scripts.train_tracker import build_optimizer
        cfg = {
            "training": {"optimizer": {
                "lr_decoder": 1e-5, "lr_embed": 1e-4,
                "weight_decay": 0.05, "betas": [0.9, 0.999],
            }}
        }
        optimizer = build_optimizer(model, embed_head, cfg)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-5  # decoder
        assert optimizer.param_groups[1]["lr"] == 1e-4  # embed_head


# ---------------------------------------------------------------------------
# Test 2: Forward pass + loss
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Test M2F forward + contrastive loss computation."""

    def test_m2f_forward_with_labels(self, model_and_head, device):
        """M2F forward pass returns loss and embeddings."""
        model, embed_head = model_and_head
        model.train()
        embed_head.train()

        batch = _make_synthetic_batch(B=1, T=2, H=256, W=256, device=device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            mask_labels=[m.to(device) for m in batch["mask_labels"]],
            class_labels=[c.to(device) for c in batch["class_labels"]],
        )

        assert outputs.loss is not None
        assert outputs.loss.item() > 0
        assert not torch.isnan(outputs.loss)

        # Embeddings
        assert outputs.transformer_decoder_last_hidden_state is not None
        BT, Q, C = outputs.transformer_decoder_last_hidden_state.shape
        assert BT == 2  # B*T
        assert C == 256

    def test_embed_head_on_m2f_output(self, model_and_head, device):
        """Embedding head processes M2F output correctly."""
        model, embed_head = model_and_head
        model.train()
        embed_head.train()

        batch = _make_synthetic_batch(B=1, T=2, device=device)
        outputs = model(pixel_values=batch["pixel_values"])

        query_embeds = outputs.transformer_decoder_last_hidden_state
        reid_embeds = embed_head(query_embeds)

        assert reid_embeds.shape == query_embeds.shape
        assert reid_embeds.requires_grad

    def test_full_loss_computation(self, model_and_head, device):
        """Full forward: M2F loss + contrastive loss, no errors."""
        model, embed_head = model_and_head
        model.train()
        embed_head.train()

        batch = _make_synthetic_batch(B=1, T=3, device=device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        outputs = model(
            pixel_values=batch["pixel_values"],
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        m2f_loss = outputs.loss

        query_embeds = outputs.transformer_decoder_last_hidden_state
        reid_embeds = embed_head(query_embeds)

        # Hungarian matching
        from scripts.train_tracker import hungarian_indices
        indices = hungarian_indices(model, outputs, mask_labels, class_labels, device)
        assert len(indices) == 3  # B*T = 1*3

        # Contrastive loss
        cl_items = _cl_items(reid_embeds, indices, batch, clip_length=3, num_negatives=10)
        cl_losses = contrastive_loss(cl_items, reid_embeds, WOD_THING_ANCHOR_WEIGHTS)

        total_loss = m2f_loss + cl_losses["loss_reid"] * 2.0 + cl_losses["loss_aux_reid"] * 1.0

        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0


# ---------------------------------------------------------------------------
# Test 3: Hungarian matching
# ---------------------------------------------------------------------------

class TestHungarianIndices:
    """Test the Hungarian matching interface."""

    def test_output_format(self, model_and_head, device):
        """Returns a list of (src, tgt) index tuples."""
        model, _ = model_and_head
        model.train()

        batch = _make_synthetic_batch(B=1, T=2, device=device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        outputs = model(
            pixel_values=batch["pixel_values"],
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        from scripts.train_tracker import hungarian_indices
        indices = hungarian_indices(model, outputs, mask_labels, class_labels, device)

        assert len(indices) == 2  # B*T frames
        for src, tgt in indices:
            assert isinstance(src, torch.Tensor)
            assert isinstance(tgt, torch.Tensor)
            assert src.shape == tgt.shape
            # Number of matches <= min(Q, N_segments)
            assert len(src) <= 4  # We have 4 segments per frame


# ---------------------------------------------------------------------------
# Test 4: Backward pass + gradient flow
# ---------------------------------------------------------------------------

class TestBackwardPass:
    """Verify gradients flow correctly."""

    def test_gradients_flow_to_decoder_and_head(self, model_and_head, device):
        """Gradients should reach decoder params and embed_head, not backbone."""
        model, embed_head = model_and_head
        model.train()
        embed_head.train()
        model.zero_grad()
        embed_head.zero_grad()

        batch = _make_synthetic_batch(B=1, T=2, device=device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        outputs = model(
            pixel_values=batch["pixel_values"],
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        reid_embeds = embed_head(outputs.transformer_decoder_last_hidden_state)

        from scripts.train_tracker import hungarian_indices
        indices = hungarian_indices(model, outputs, mask_labels, class_labels, device)
        cl_items = _cl_items(reid_embeds, indices, batch, clip_length=2)
        cl_losses = contrastive_loss(cl_items, reid_embeds, WOD_THING_ANCHOR_WEIGHTS)

        total = outputs.loss + cl_losses["loss_reid"] + cl_losses["loss_aux_reid"]
        total.backward()

        # Embed head should have gradients
        has_embed_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in embed_head.parameters()
        )
        assert has_embed_grad, "Embed head should receive gradients"

        # Backbone should NOT have gradients
        for name, p in model.named_parameters():
            if "model.pixel_level_module.encoder" in name:
                assert p.grad is None or p.grad.abs().sum() == 0, \
                    f"Backbone param {name} should not receive gradients"


# ---------------------------------------------------------------------------
# Test 5: Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Test checkpoint round-trip."""

    def test_save_load_roundtrip(self, model_and_head, device, output_dir):
        """Checkpoint saves and loads all state correctly."""
        model, embed_head = model_and_head
        from scripts.train_tracker import build_optimizer, _build_state
        from panoctvps.training.engine import save_checkpoint, load_latest_checkpoint

        cfg = {"training": {"optimizer": {
            "lr_decoder": 1e-5, "lr_embed": 1e-4,
            "weight_decay": 0.05, "betas": [0.9, 0.999],
        }}}
        optimizer = build_optimizer(model, embed_head, cfg)
        scaler = torch.amp.GradScaler(device="cuda", enabled=False)

        # Save
        state = _build_state(model, embed_head, optimizer, scaler,
                             iteration=500, best_metric=0.5)
        save_checkpoint(state, output_dir, iteration=500)

        # Load
        loaded = load_latest_checkpoint(output_dir, device)
        assert loaded is not None
        assert loaded["iteration"] == 500
        assert loaded["best_metric"] == 0.5
        assert "model" in loaded
        assert "embed_head" in loaded
        assert "optimizer" in loaded
        assert "scaler" in loaded

        # Verify model weights match
        for key in model.state_dict():
            assert torch.equal(
                loaded["model"][key].to(device),
                model.state_dict()[key],
            ), f"Model weight mismatch at {key}"

        for key in embed_head.state_dict():
            assert torch.equal(
                loaded["embed_head"][key].to(device),
                embed_head.state_dict()[key],
            ), f"Embed head weight mismatch at {key}"


# ---------------------------------------------------------------------------
# Test 6: PQ evaluation
# ---------------------------------------------------------------------------

class TestPQEvaluation:
    """Test evaluate_pq function (requires dataset, skip if unavailable)."""

    @pytest.fixture
    def dataset_root(self):
        root = "../data/waymo_open_dataset_v_2_0_1"
        if not os.path.exists(root):
            pytest.skip("WOD dataset not available")
        return root

    def test_evaluate_pq_runs(self, model_and_head, device, dataset_root):
        """evaluate_pq returns valid metrics on a small sample."""
        model, _ = model_and_head
        from scripts.train_tracker import evaluate_pq
        cfg = {
            "dataset": {"cameras": [1]},
            "augmentation": {"crop_size": [256, 256]},
        }
        metrics = evaluate_pq(model, dataset_root, cfg, device, n_samples=5)

        assert "PQ" in metrics
        assert "mIoU" in metrics
        assert 0.0 <= metrics["PQ"] <= 1.0
        assert 0.0 <= metrics["mIoU"] <= 1.0
        assert metrics["num_images"] == 5


# ---------------------------------------------------------------------------
# Test 7: Tracking evaluation
# ---------------------------------------------------------------------------

class TestTrackingEvaluation:
    """Test evaluate_tracking function."""

    @pytest.fixture
    def dataset_root(self):
        root = "../data/waymo_open_dataset_v_2_0_1"
        if not os.path.exists(root):
            pytest.skip("WOD dataset not available")
        return root

    def test_evaluate_tracking_runs(self, model_and_head, device, dataset_root):
        """evaluate_tracking returns valid wSTQ metrics for 1 sequence."""
        model, embed_head = model_and_head
        from scripts.train_tracker import evaluate_tracking
        cfg = {"dataset": {"cameras": [1]}}
        results = evaluate_tracking(
            model, embed_head, dataset_root, cfg, device, n_sequences=1,
        )

        for key in ("STQ", "AQ", "SQ"):
            assert key in results
            assert 0.0 <= results[key] <= 1.0


# ---------------------------------------------------------------------------
# Test 8: predict_panoptic_with_learned_embeddings
# ---------------------------------------------------------------------------

class TestPredictWithEmbedHead:
    """Test learned embedding injection into panoptic predictions."""

    def test_returns_predictions(self, model_and_head, device):
        """predict_panoptic_with_learned_embeddings returns panoptic preds with embeddings."""
        model, embed_head = model_and_head
        from panoctvps.segmenter.inference import predict_panoptic_with_learned_embeddings

        pv = torch.randn(1, 3, 256, 256).to(device)
        preds = predict_panoptic_with_learned_embeddings(model, embed_head, pv)

        assert len(preds) == 1
        assert "semantic" in preds[0]
        assert "instance" in preds[0]
        assert "instance_embeddings" in preds[0]
        assert preds[0]["semantic"].shape == (256, 256)

    def test_embeddings_are_from_head(self, model_and_head, device):
        """The instance embeddings are outputs of the embedding head: a
        different head gives different embeddings for the same segments."""
        model, embed_head = model_and_head
        from panoctvps.segmenter.inference import predict_panoptic_with_learned_embeddings

        torch.manual_seed(0)
        other_head = EmbeddingHead(input_dim=256, hidden_dim=256, output_dim=256).to(device)
        pv = torch.randn(1, 3, 256, 256).to(device)

        preds_a = predict_panoptic_with_learned_embeddings(model, embed_head, pv)
        preds_b = predict_panoptic_with_learned_embeddings(model, other_head, pv)

        for iid, emb_a in preds_a[0]["instance_embeddings"].items():
            emb_b = preds_b[0]["instance_embeddings"][iid]
            assert not np.allclose(emb_a, emb_b, atol=1e-5), \
                "Embeddings should depend on the embedding head"


# ---------------------------------------------------------------------------
# Test 9: Full training loop
# ---------------------------------------------------------------------------

class TestFullTrainingLoop:
    """End-to-end training loop with synthetic data."""

    def test_multi_iteration_training(self, model_and_head, device, output_dir):
        """Run 5 iterations, verify loss decreases or stays finite."""
        model, embed_head = model_and_head
        model.train()
        embed_head.train()

        from scripts.train_tracker import build_optimizer, hungarian_indices, _build_state
        from panoctvps.training.engine import save_checkpoint

        cfg = {"training": {"optimizer": {
            "lr_decoder": 1e-5, "lr_embed": 1e-4,
            "weight_decay": 0.05, "betas": [0.9, 0.999],
        }}}
        optimizer = build_optimizer(model, embed_head, cfg)

        losses = []
        for i in range(5):
            batch = _make_synthetic_batch(B=1, T=3, H=256, W=256, device=device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            optimizer.zero_grad()

            outputs = model(
                pixel_values=batch["pixel_values"],
                mask_labels=mask_labels,
                class_labels=class_labels,
            )

            reid_embeds = embed_head(outputs.transformer_decoder_last_hidden_state)
            indices = hungarian_indices(model, outputs, mask_labels, class_labels, device)
            cl_items = _cl_items(reid_embeds, indices, batch, clip_length=3)
            cl_losses = contrastive_loss(cl_items, reid_embeds, WOD_THING_ANCHOR_WEIGHTS)

            total = outputs.loss + cl_losses["loss_reid"] * 2 + cl_losses["loss_aux_reid"]
            total.backward()
            optimizer.step()

            losses.append(total.item())

        # All losses should be finite
        assert all(np.isfinite(l) for l in losses), f"Non-finite losses: {losses}"

        # Save checkpoint at the end
        scaler = torch.amp.GradScaler(device="cuda", enabled=False)
        state = _build_state(model, embed_head, optimizer, scaler, 5, 0.0)
        ckpt_path = save_checkpoint(state, output_dir, iteration=5)
        assert os.path.exists(ckpt_path)

    def test_eval_json_output(self, model_and_head, device, output_dir):
        """Verify eval results are saved as JSON with correct structure."""
        # Simulate what the training loop does when saving eval
        eval_results = {
            "iteration": 1000,
            "PQ": 0.5,
            "PQ_thing": 0.5,
            "PQ_stuff": 0.5,
            "mIoU": 0.5,
            "tracking": {"STQ": 0.25, "AQ": 0.125, "SQ": 0.5},
        }
        eval_path = os.path.join(output_dir, "eval_0001000.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Load and verify
        with open(eval_path) as f:
            loaded = json.load(f)

        assert loaded["iteration"] == 1000
        assert loaded["PQ"] == 0.5
        assert "tracking" in loaded
        assert loaded["tracking"]["AQ"] == 0.125


# ---------------------------------------------------------------------------
# Test 10: video_collate_fn
# ---------------------------------------------------------------------------

class TestVideoCollateFn:
    """Test the video clip collate function."""

    def test_collate_single_clip(self):
        """Collate a single clip."""
        clip = {
            "pixel_values": torch.randn(5, 3, 640, 640),
            "mask_labels": [torch.randn(4, 640, 640) for _ in range(5)],
            "class_labels": [torch.randint(0, 29, (4,)) for _ in range(5)],
            "global_instance_ids": [torch.tensor([101, 102, -1, -1]) for _ in range(5)],
            "meta": {"segment_name": "test", "camera_id": 1, "timestamps": [0, 1, 2, 3, 4]},
        }
        batch = video_collate_fn([clip])

        assert batch["pixel_values"].shape == (5, 3, 640, 640)
        assert len(batch["mask_labels"]) == 5
        assert batch["clip_length"] == 5
        assert batch["batch_size"] == 1

    def test_collate_multiple_clips(self):
        """Collate multiple clips into a batch."""
        clips = []
        for _ in range(2):
            clips.append({
                "pixel_values": torch.randn(5, 3, 256, 256),
                "mask_labels": [torch.randn(3, 256, 256) for _ in range(5)],
                "class_labels": [torch.randint(0, 29, (3,)) for _ in range(5)],
                "global_instance_ids": [torch.tensor([1, 2, -1]) for _ in range(5)],
                "meta": {"segment_name": "test", "camera_id": 1, "timestamps": list(range(5))},
            })
        batch = video_collate_fn(clips)

        assert batch["pixel_values"].shape == (10, 3, 256, 256)  # B*T = 2*5
        assert len(batch["mask_labels"]) == 10
        assert batch["clip_length"] == 5
        assert batch["batch_size"] == 2
