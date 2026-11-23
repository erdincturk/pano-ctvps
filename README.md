# Pano-CTVPS

**Panoramic video panoptic segmentation on the Waymo Open Dataset.** Pano-CTVPS labels every pixel of the
five cameras of the Waymo rig with a class and gives every object one identity that holds across the cameras
and over time. It has three parts:

- a **segmenter** (Mask2Former with a Swin-L backbone, one model shared by all cameras) that predicts the
  class, the mask and an embedding of every segment in each image;
- a **tracker** that follows the objects of each camera over time and outputs **tracks**;
- a **linker** that joins the tracks of adjacent cameras into **global identities**, using the camera
  calibration and the embeddings.

![The Pano-CTVPS pipeline](assets/pipeline.png)

This repository holds the code of the article *Pano-CTVPS: A Three-Stage Pipeline for Panoramic Video Panoptic
Segmentation* (Türk, Erdinç and Danışman, Taner; IEEE Access, under review). It contains no Waymo data, no predictions,
no caches and no trained weights.

## Results

WOD 2D Video Panoptic Segmentation benchmark, test split, scored by the benchmark server
([leaderboard](https://waymo.com/open/challenges/2023/2d-video-panoptic-segmentation/)):

| Method | wSTQ | wAQ | wSQ (leaderboard: mIoU) |
|---|---|---|---|
| ViP-DeepLab baseline (Mei et al., 2022) | 0.1750 | 0.1062 | 0.2883 |
| **Pano-CTVPS** | **0.3187** | **0.1781** | **0.5702** |

Deployed settings. Tracker: θ_match 0.7, θ_init 0.01, N_dead 5 frames, motion gate 600 / 800 / 800 / 1000 /
1000 px (Front, Front-Left, Front-Right, Side-Left, Side-Right). Linker: `configs/linker_deployed.json`.

![Validation scene: predicted labels of vehicles passing between cameras at an intersection](assets/preview_intersection.gif)

The preview shows the predicted panoptic labels of the five cameras in rig order, without the camera images. Every
object keeps one color and number across the cameras and over time. Dashed outlines mark the overlap regions of
adjacent cameras, computed from the calibration.

## Install

Python 3.10 or later and a CUDA GPU. Pure PyTorch: no custom CUDA extensions, no Detectron2. The exact versions
are pinned in [`requirements.txt`](requirements.txt) (Python 3.12, PyTorch 2.13.0 with CUDA 13.0,
Transformers 5.15.0, NumPy 2.5.2).

```bash
git clone https://github.com/erdincturk/pano-ctvps.git && cd pano-ctvps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
export TORCH_DISABLE_NATIVE_JIT=1   # PyTorch 2.13: skips a Triton JIT that needs python3-dev
pytest                              # unit tests, incl. an STQ check against the DeepLab2 reference
```

## Data

The code uses the [Waymo Open Dataset](https://waymo.com/open/) v2.0.1, which is not redistributed here. Register
on the Waymo Open Dataset website, accept its [terms of use](https://waymo.com/open/terms), and download the
`camera_image`, `camera_segmentation` and `camera_calibration` components of the training, validation and test
splits (for example with `gcloud storage cp`). The code expects them at `../data/waymo_open_dataset_v_2_0_1` or
at the path given with `--dataset-root`. The scene lists are the 696 / 20 / 40 scenes of the benchmark, in
`configs/splits/`. Check the download with `python scripts/pipeline/verify_dataset.py --data-root <root>`.

This software was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License
Agreement for Non-Commercial Use, available at [waymo.com/open/terms](https://waymo.com/open/terms), and your
access and use of such work are governed by the terms and conditions therein.

## Repository layout

```
panoctvps/               the package: data loading, segmenter, tracker, linker, evaluation (STQ, PQ), submission
scripts/pipeline/        train, cache, track, link and submit: the deployed system (steps 1 to 6 below)
scripts/experiments/     the experiments of the paper: tracker cells, held-out check, linker evaluations,
                         causal linker, runtime
scripts/analysis/        the analyses of the paper: dataset statistics, failure modes, tracker and linker analyses
configs/data, models/    dataset and model settings shared by the training configurations
configs/training/        segmenter and identity training (segmenter_cityscapes.yaml and segmenter_mapillary.yaml:
                         the comparison of the two initializations, Section V-B)
configs/tracker_cells/   the tracker settings of Table 6 and Tables S9 to S14
configs/splits/          the annotated frames of the training, validation and test scenes
configs/linker_*.json    the deployed linker and the steps of Table 7
tests/                   unit tests (pytest)
```

## Run the pipeline

Checkpoints are read from `checkpoints/`, results are written to `outputs/`. Every script documents its options
with `--help`. Evaluation is at native resolution, zero-padded to a multiple of 32.

```bash
# 1. Segmenter, panoptic fine-tuning: 20,000 iterations on the Front camera, then 80,000 on all cameras
python scripts/pipeline/train_segmenter.py --config configs/training/segmenter_mapillary_refined_front.yaml
python scripts/pipeline/train_segmenter.py --config configs/training/segmenter_mapillary_refined.yaml \
    --resume-from outputs/segmenter_mapillary_refined_front/best.pth
python scripts/pipeline/eval_segmenter.py --config configs/training/segmenter_mapillary_refined.yaml \
    --checkpoint checkpoints/segmenter.pth                      # Table 5

# 2. Identity training of the decoder and the embedding head
python scripts/pipeline/train_tracker.py --config configs/training/tracker_ctvis.yaml

# 3. Segmenter outputs, computed once and cached (validation and training with ground-truth ids; test without)
python scripts/pipeline/build_cache_gt.py --checkpoint checkpoints/tracker.pth --split validation \
    --cache-dir outputs/tracker_ctvis_val_cache
python scripts/pipeline/build_cache_gt.py --checkpoint checkpoints/tracker.pth --split training \
    --cache-dir outputs/tracker_ctvis_train_cache
python scripts/pipeline/build_cache.py --checkpoint checkpoints/tracker.pth --split test \
    --cache-dir outputs/tracker_ctvis_test_cache

# 4. Tracker at the deployed settings (CPU, from the cache); its tracks are written into the cache
python scripts/pipeline/eval_tracker_cached.py --cache-dir outputs/tracker_ctvis_val_cache --tracker memorybank \
    --match-threshold 0.7 --init-score-threshold 0.01 --num-dead-frames 5 \
    --motion-gate --motion-gate-px-per-cam "1:600,2:800,3:800,4:1000,5:1000" \
    --save-trajectories outputs/tracker/val_tracks.npz --output outputs/tracker/val.json
python scripts/pipeline/build_cache_trackids.py --cache-dir outputs/tracker_ctvis_val_cache \
    --trajectories outputs/tracker/val_tracks.npz --write
#    the same two commands for the training cache, and for the test cache with --no-eval

# 5. Linker: adapter, candidate scores, adapted cosines, evaluation
python scripts/pipeline/build_pairs.py --cache-dir outputs/tracker_ctvis_train_cache --out-dir outputs/linker/pairs_train
python scripts/pipeline/train_adapter.py --pairs-dir outputs/linker/pairs_train --cache-dir outputs/tracker_ctvis_train_cache \
    --out-dir outputs/linker
python scripts/pipeline/build_scores.py --cache-dir outputs/tracker_ctvis_val_cache --out-dir outputs/linker/scored_candidates
python scripts/pipeline/build_cosines.py --adapter outputs/linker/adapter.pt --cache-dir outputs/tracker_ctvis_val_cache \
    --scores-dir outputs/linker/scored_candidates --out-dir outputs/linker/adapted_cosines
python scripts/pipeline/eval_linker_wstq.py --cache-dir outputs/tracker_ctvis_val_cache \
    --cosines-dir outputs/linker/adapted_cosines --configs configs/linker_deployed.json \
    --output outputs/linker/deployed.json

# 6. Benchmark submission (test split)
python scripts/pipeline/build_scores.py --cache-dir outputs/tracker_ctvis_test_cache --out-dir outputs/linker/scored_candidates_test
python scripts/pipeline/build_cosines.py --adapter outputs/linker/adapter.pt --cache-dir outputs/tracker_ctvis_test_cache \
    --scores-dir outputs/linker/scored_candidates_test --out-dir outputs/linker/adapted_cosines_test
python scripts/pipeline/build_submission.py --split test --account <registered e-mail> \
    --cache-dir outputs/tracker_ctvis_test_cache --cosines-dir outputs/linker/adapted_cosines_test
python scripts/pipeline/package_submission.py --split test
python scripts/pipeline/verify_submission.py --split test     # with --split validation: recomputes wSTQ 0.3228
```

## Reproduce the experiments of the paper

All steps after the caches run on the CPU and are deterministic given the caches. Each tracker evaluation needs
about 18 GB of memory; `--workers` sets how many scenes run in parallel.

### Caches and overlap tables

Table 6 compares three models, each with its own validation cache: the deployed model (step 3 above), the segmenter
before identity training, and the identity training in which every positive is the raw embedding of its reference
frame (p = 0, Section IV-B1).

```bash
# segmenter before identity training: its weights with an unused embedding head
python scripts/experiments/make_segmenter_cache_checkpoint.py --segmenter checkpoints/segmenter.pth \
    --output checkpoints/segmenter_for_cache.pth
python scripts/pipeline/build_cache_gt.py --checkpoint checkpoints/segmenter_for_cache.pth --split validation \
    --cache-dir outputs/segmenter_val_cache
# identity training with p = 0
python scripts/pipeline/train_tracker.py --config configs/training/tracker_ctvis_p0.yaml
python scripts/pipeline/build_cache_gt.py --checkpoint checkpoints/tracker_ctvis_p0.pth --split validation \
    --cache-dir outputs/tracker_ctvis_p0_val_cache
# overlap tables, one folder per cache (likewise segmenter_val and tracker_ctvis_p0_val)
python scripts/experiments/build_overlap_tables.py --cache-dir outputs/tracker_ctvis_val_cache \
    --out-dir outputs/overlap_tables/tracker_ctvis_val
```

The trackers and the linker change only identities, never masks or classes, so every tracker and linker run below
is scored exactly from the overlap tables of its cache, with the same scores as `scripts/pipeline/eval_tracker_cached.py`
and `scripts/pipeline/eval_linker_wstq.py`.

**Rule of the evaluator.** When a camera frame has no ground-truth pixel of a thing class, `STQuality.update`
returns before it adds the predicted instances of that frame (`panoctvps/evaluation/stq.py`), so predicted thing
masks in such frames are skipped and do not lower AQ. Every validation score of the paper is computed this way, and
the overlap tables follow the same rule. Without it, 12 of the 20 validation scenes would differ by up to 0.0005 in
wAQ. The test score is computed by the benchmark server and is not affected.

### Tracker cells

A cell is one tracker at one setting on one cache (`configs/tracker_cells/`). Run them per cache; the gate sweeps
go to a folder of their own, so that the held-out check below can select the grids by file name.

```bash
C=tracker_ctvis_val      # likewise segmenter_val (baselines_grid only) and tracker_ctvis_p0_val
for g in baselines_grid ctvis_grid full_tracker tracker_components; do
  python scripts/experiments/run_tracker_cells.py --cache-dir outputs/${C}_cache --tables-dir outputs/overlap_tables/$C \
      --cells configs/tracker_cells/$g.json --out-dir outputs/tracker_cells/$C
done
for m in 0.60 0.70; do   # deployed model only
  python scripts/experiments/run_tracker_cells.py --cache-dir outputs/${C}_cache --tables-dir outputs/overlap_tables/$C \
      --cells configs/tracker_cells/gate_sweep_m$m.json --out-dir outputs/tracker_cells/${C}_gate
done
```

Long runs can be split by scene (`--scenes`, one folder per scene) and merged with `scripts/experiments/merge_tracker_cells.py`.

### Script and command per table

| Table | Script and command |
|---|---|
| Table 2 (splits) | `python scripts/analysis/label_statistics.py --split training --output-dir outputs/analyses/labels` and the same with `--split validation`, then `python scripts/analysis/dataset_statistics.py labels --masks-dir outputs/analyses/labels --output outputs/analyses/dataset_labels.json` |
| Table 4 (test leaderboard) | step 6 above (scored by the benchmark server) |
| Table 5 (segmenter) | steps 1 and 3 above |
| Table 6 (trackers) | the tracker cells above: rows 1 and 2 from `baselines_grid` on `segmenter_val`; rows 3 to 6 from `baselines_grid` and `ctvis_grid` on `tracker_ctvis_p0_val`; rows 7 to 10 from `baselines_grid` and `ctvis_grid` on `tracker_ctvis_val`; row 11 from `full_tracker` (each cell writes `<cell>.json`) |
| Table 7 (linker) | steps: `python scripts/pipeline/eval_linker_wstq.py --cache-dir outputs/tracker_ctvis_val_cache --cosines-dir outputs/linker/adapted_cosines --configs configs/linker_table7.json --output outputs/linker/table7.json` and, for precision and recall, `python scripts/experiments/eval_linker_links.py --scores-dir outputs/linker/scored_candidates --cosines-dir outputs/linker/adapted_cosines --configs configs/linker_table7.json --output outputs/linker/table7_links.json`; alternative strategies: `python scripts/experiments/eval_linker_alternatives.py prep --tables-dir outputs/overlap_tables/tracker_ctvis_val --out-dir outputs/linker/alternatives`, then the same with `cells` |
| Table 8 (failure modes) | segmenter rows: `python scripts/analysis/analyze_classes.py --cache-dir outputs/tracker_ctvis_val_cache --tid-field tracker_id_deployed --output outputs/analyses/classes.json` and `python scripts/analysis/analyze_unmatched_masks.py --cache-dir outputs/tracker_ctvis_val_cache --output outputs/analyses/unmatched_masks.json`; tracker and linker rows: `python scripts/analysis/analyze_failures.py --tables-dir outputs/overlap_tables/tracker_ctvis_val --output outputs/analyses/failures.json` |
| Runtime (Section VI-C) | `python scripts/experiments/time_pipeline.py pipeline --checkpoint <ckpt> --output outputs/timing/timing_pipeline.json`, `python scripts/experiments/time_pipeline.py linker --output outputs/timing/timing_linker_frames.json` and `python scripts/experiments/time_pipeline.py tracker --output outputs/timing/timing_tracker_frames.json` |
| Table S1 (thing classes) | `dataset_statistics.py labels` (as Table 2) |
| Table S2 (motion between frames) | `python scripts/analysis/dataset_statistics.py motion --output outputs/analyses/dataset_motion.json` |
| Table S3 (instances per camera) | `dataset_statistics.py labels` (as Table 2) |
| Table S4 (span of the ground-truth tracks) | `analyze_age_gap.py --gt-masks` (as Table S15) |
| Table S5 (camera pairs) | instances: `dataset_statistics.py labels` (as Table 2); overlap band: `python scripts/analysis/dataset_statistics.py band --output outputs/analyses/dataset_band.json` |
| Table S6 (scene conditions) | `python scripts/analysis/dataset_statistics.py conditions --output outputs/analyses/dataset_conditions.json` |
| Tables S7 and S8 (outputs per thing class) | `analyze_classes.py` and `analyze_unmatched_masks.py` (as Table 8) |
| Tables S9 and S10 (IoU tracker, MinVIS) | the cells of `baselines_grid` on the three caches |
| Table S11 (our tracker) | the cells of `ctvis_grid` on `tracker_ctvis_p0_val` and `tracker_ctvis_val` |
| Table S12 (motion gate) | the cells of `gate_sweep_m0.70` (per camera in `per_camera` of each cell JSON) |
| Table S13 (held-out check) | `python scripts/experiments/heldout_check.py --counts-from outputs/tracker_cells/tracker_ctvis_val/ours_full_tracker_deployed.json --row "1|outputs/tracker_cells/segmenter_val/iou_t*.json|iou_t0.30_d5" --row "2|outputs/tracker_cells/segmenter_val/minvis_m0.[5-9]0_*.json|minvis_m0.90_i0.01_d5" --row "3|outputs/tracker_cells/tracker_ctvis_p0_val/iou_t*.json|iou_t0.30_d5" --row "4|outputs/tracker_cells/tracker_ctvis_p0_val/minvis_m0.[5-9]0_*.json|minvis_m0.90_i0.01_d5" --row "6|outputs/tracker_cells/tracker_ctvis_p0_val/ours_m*.json|ours_m0.70_i0.01_d5" --row "7|outputs/tracker_cells/tracker_ctvis_val/iou_t*.json|iou_t0.30_d5" --row "8|outputs/tracker_cells/tracker_ctvis_val/minvis_m0.[5-9]0_*.json|minvis_m0.90_i0.01_d5" --row "10|outputs/tracker_cells/tracker_ctvis_val/ours_m*.json|ours_m0.70_i0.01_d5" --full-tracker 10 --tables-dir outputs/overlap_tables/tracker_ctvis_val --sweep "ours_m0.70_i0.01_d5=outputs/tracker_cells/tracker_ctvis_val_gate/ours_m0.70_i0.01_d5_gate" --sweep "ours_m0.60_i0.01_d5=outputs/tracker_cells/tracker_ctvis_val_gate/ours_m0.60_i0.01_d5_gate" --deployed outputs/tracker_cells/tracker_ctvis_val/ours_full_tracker_deployed.json --output outputs/heldout/heldout.json` (row 11 is the full tracker) |
| Table S14 (tracker components) | the cells of `full_tracker` and `tracker_components`; the note on the class gate: `python scripts/analysis/analyze_class_gate.py --tables-dir outputs/overlap_tables/tracker_ctvis_val --cells-dir outputs/tracker_cells/tracker_ctvis_val --output outputs/analyses/class_gate.json` |
| Tables S15 and S16 (decisions by gap and age) | `python scripts/analysis/analyze_age_gap.py --model "p0:outputs/tracker_ctvis_p0_val_cache:outputs/overlap_tables/tracker_ctvis_p0_val" --model "deployed:outputs/tracker_ctvis_val_cache:outputs/overlap_tables/tracker_ctvis_val" --cell "p0:published:outputs/tracker_cells/tracker_ctvis_p0_val/ours_published_m0.30_i0.01_d20" --cell "p0:retuned:outputs/tracker_cells/tracker_ctvis_p0_val/ours_m0.70_i0.01_d5" --cell "deployed:published:outputs/tracker_cells/tracker_ctvis_val/ours_published_m0.30_i0.01_d20" --cell "deployed:retuned:outputs/tracker_cells/tracker_ctvis_val/ours_m0.70_i0.01_d5" --gt-masks outputs/analyses/labels/validation_thing_masks.npz --output outputs/analyses/age_gap.json` (also the paired per-scene differences of Section VI-B2) |
| Tables S17 and S19 (decisions by active tracks; the switch of Fig. 7(c4)) | `python scripts/analysis/analyze_tracker_trace.py --tables-dir outputs/overlap_tables/tracker_ctvis_val --check-ids outputs/tracker_cells/tracker_ctvis_val/ours_full_tracker_deployed_ids.npz --output outputs/analyses/tracker_trace.json` |
| Table S18 (failures of the tracker) | `analyze_failures.py` (as Table 8); AQ per class: `python scripts/experiments/eval_linker_tables.py --tables-dir outputs/overlap_tables/tracker_ctvis_val --output outputs/linker/linker_tables_deployed.json --keep-per-gt`; identities per instance and camera: `analyze_classes.py` (as Table 8) |
| Table S20 (links per camera pair) | `eval_linker_links.py` (as Table 7) |
| Tables S21, S22 and S24 (stages of the linker, geometry against appearance, frames with evidence) | `eval_linker_tables.py` (as Table S18), then `python scripts/analysis/analyze_linker_stages.py --tables-dir outputs/overlap_tables/tracker_ctvis_val --linker-tables outputs/linker/linker_tables_deployed.json --output outputs/analyses/linker_stages.json` |
| Table S23 (alternative linking strategies) | `eval_linker_alternatives.py` (as Table 7) |
| Table S25 (causal linker) | `python scripts/experiments/run_online_linker.py --stage build --workers 8`, then `--stage sweep`; writes `outputs/online_linker/REPORT.md` |
| Table S26 (failures of the linker) | `analyze_failures.py` (as Table 8) |
| Table S27 (per scene: instances, candidates, links) | instances: `dataset_statistics.py labels` (as Table 2); candidates, links and the gain of the linker: `analyze_linker_stages.py` (as Table S21) |
| Table S28 (per scene: conditions and scores) | conditions: `dataset_statistics.py conditions` (as Table S6); scores: `per_scene` of the full system in `outputs/linker/linker_tables_deployed.json` (`eval_linker_tables.py`, as Table S18) |

## Citation

If you use this code, please cite the article and the dataset:

```bibtex
@article{turk2026panoctvps,
  author  = {Türk, Erdinç and Danışman, Taner},
  title   = {{Pano-CTVPS}: A Three-Stage Pipeline for Panoramic Video Panoptic Segmentation},
  journal = {IEEE Access},
  year    = {2026},
  note    = {under review}
}

@misc{waymo_open_dataset,
  title   = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year    = {2019-2025}
}

@inproceedings{sun2020scalability,
  author    = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai
               and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
  title     = {Scalability in Perception for Autonomous Driving: {Waymo} Open Dataset},
  booktitle = {Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}

@inproceedings{mei2022waymo,
  author    = {Mei, Jieru and Zhu, Alex Zihao and Yan, Xinchen and Yan, Hang and Qiao, Siyuan and Chen, Liang-Chieh
               and Kretzschmar, Henrik},
  title     = {{Waymo} Open Dataset: Panoramic Video Panoptic Segmentation},
  booktitle = {Proc. European Conf. Computer Vision (ECCV)},
  year      = {2022}
}
```

## Acknowledgment

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under Grant
225E028.

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). The Waymo Open Dataset is not part of
this repository and is subject to its own [terms of use](https://waymo.com/open/terms).
