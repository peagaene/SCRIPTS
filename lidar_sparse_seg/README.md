# LiDAR Sparse Semantic Segmentation (spconv)

Single-source guide for training, validation, inference, experimentation, and pipeline maintenance.

## 1. Goal

This project performs semantic segmentation on LAS/LAZ point clouds using a sparse 3D CNN (`spconv`) and predicts one class per point/voxel.

Current working setup (internal classes):

- `0 = ground`
- `1 = building`
- `2 = vegetation`

Ignored by default:

- LAS class `7` (noise)
- optional extra ignored classes from `configs/ignore_las_classes.json`

## 2. Repository Structure

- `configs/default.yaml`: main runtime configuration
- `src/train.py`: training loop + periodic validation + checkpoints
- `src/validate.py`: offline validation/test reports
- `src/infer.py`: tile/file inference and optional post-processing
- `src/datasets/lidar_dataset.py`: reading, label mapping, feature building, voxelization
- `src/models/spconv_unet.py`: sparse UNet model
- `src/utils/*.py`: IO, geometry features, metrics, split generation, sparse helpers
- `TEMPLATE_EXPERIMENTOS.csv`: experiment tracking table (semicolon-separated)

## 3. Data and Label Mapping

Label conversion is defined in:

- `configs/las_to_train.json`
- `configs/classes.json`
- `configs/ignore_las_classes.json`

Typical mapping in this project:

- LAS `2 -> train 0` (ground)
- LAS `6 -> train 1` (building)
- LAS `3,4,5,13 -> train 2` (vegetation)
- LAS `7` ignored

## 4. Full Pipeline

1. Read LAS/LAZ points and attributes.
2. Drop ignored LAS classes.
3. Map LAS classes to training labels.
4. Crop in XY (`crop_size`).
5. Normalize features:
   - centered XY
   - relative Z (`z - z_min`)
   - normalized intensity
6. Optional engineered features:
   - HAG
   - return features
   - scan angle
   - local geometry (`normal_z`, roughness, slope, planarity, linearity)
7. Voxelize (`voxel_size`) and aggregate.
8. Train/evaluate sparse UNet logits with weighted CE loss.
9. Convert internal classes back to LAS classes at inference time.

## 5. Configuration (default.yaml)

Core:

- `seed`: reproducibility
- `num_classes`: number of internal classes
- `voxel_size`: voxel resolution in meters
- `crop_size`: XY crop window size in meters
- `max_points_per_crop`: hard crop cap
- `batch_size`, `epochs`, `lr`, `weight_decay`, `scheduler`, `amp`

Feature toggles:

- `use_hag`, `hag_cell_size`
- `use_return_features`
- `use_scan_angle`
- `use_normal_features`
- `use_roughness_feature`
- `use_slope_feature`
- `use_planarity_feature`
- `use_linearity_feature`
- `normal_cell_size`, `normal_min_points`, `roughness_scale`

Performance/runtime:

- `num_workers`
- `tile_cache_size`
- `cache_full_samples_val`

Training controls (new):

- `fixed_val_batches`: validate on first N batches every epoch (0 = full val)
- `early_stopping_patience`: epochs without improvement before stop (0 = disabled)
- `early_stopping_min_delta`: minimum `mIoU` gain to count as improvement
- `checkpoint_every_n_epochs`: save `epoch_XXX.pth` periodically (0 = disabled)
- `validate_max_batches`: cap batches in `src.validate` benchmark mode (0 = full)

Split management:

- `split.auto_generate`
- `split.regenerate`
- `split.train_ratio`, `split.val_ratio`, `split.test_ratio`
- `split.seed`

Paths:

- `paths.data_root`
- `paths.train_split`, `paths.val_split`, `paths.test_split`
- `paths.checkpoints_dir`, `paths.logs_dir`

## 6. Train

```powershell
python -m src.train --config configs/default.yaml
```

Resume from checkpoint:

```powershell
python -m src.train --config configs/default.yaml --resume_checkpoint E:/training/checkpoints/main/best_mIoU.pth
```

Warm-start weights only:

```powershell
python -m src.train --config configs/default.yaml --resume_checkpoint E:/training/checkpoints/main/best_mIoU.pth --resume_weights_only
```

Training outputs:

- `best_mIoU.pth`
- `last.pth`
- optional `epoch_XXX.pth` periodic snapshots
- `logs/history.csv`
- `logs/tb` (TensorBoard)
- `run_artifacts/config_resolved.yaml`
- `run_artifacts/class_stats.csv`

## 7. Validate and Test

Validation:

```powershell
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --split val
```

Test:

```powershell
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --split test
```

Benchmark mode (fixed number of batches):

```powershell
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --split val --max_batches 50
```

Reports are saved under:

- `paths.checkpoints_dir/reports/metrics_*.json`
- `paths.checkpoints_dir/reports/confusion_*.csv`
- `paths.checkpoints_dir/reports/summary_*.csv`

## 8. Inference

Single file:

```powershell
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --input E:/training/inf/input/197039.laz --output E:/training/inf/output/197039_pred.laz
```

Directory:

```powershell
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --input E:/training/inf/input --output E:/training/inf/output
```

Force post-processing at runtime:

```powershell
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints/main/best_mIoU.pth --input E:/training/inf/input/197039.laz --output E:/training/inf/output/197039_pp.laz --postprocess
```

## 9. How to Interpret Results

Primary model selection criteria:

1. `test mIoU`
2. `IoU_building`
3. building/vegetation confusion rates
4. visual quality on representative tiles

Key warning signs:

- train loss down, val/test IoU flat: overfit or weak split
- high accuracy but low mIoU: dominant classes masking errors
- building IoU unstable: class confusion and/or weak sampling diversity

## 10. Experiment Protocol

Use `TEMPLATE_EXPERIMENTOS.csv` and keep this discipline:

1. Freeze split (`split.regenerate: false`) before comparisons.
2. Change only one variable per experiment.
3. Run val + test every time.
4. Record quantitative metrics and visual observations.
5. Promote/reject based on operational quality, not mIoU alone.

Recommended operational checks per tile:

- building-to-vegetation confusion frequency
- isolated false building islands
- boundary quality around roofs and terrain
- estimated manual correction effort

## 11. Recommended Optimization Roadmap

1. Keep one stable baseline config.
2. Expand training data gradually with controlled resumes.
3. Use early stopping and periodic checkpoints for long runs.
4. Apply benchmark mode (`fixed_val_batches`, `--max_batches`) for fast A/B cycles.
5. Only promote settings that improve both metrics and visual consistency.

## 12. Troubleshooting

`spconv requires CUDA`:

- install CUDA-enabled PyTorch and matching `spconv` wheel.

No `best_mIoU.pth` update after resume:

- expected when resumed `best_mIoU` is already higher than new run values.
- use `--resume_weights_only` for new-domain fine-tuning comparisons.

Inference appears stuck at 0%:

- first stage may spend time reading/voxelizing large files.
- try smaller `--block_size` and/or lower `--max_points_per_block`.

Prediction noise remains:

- improve training data diversity
- tune post-processing thresholds
- add hard examples for problematic patterns

## 13. Notes on Compatibility

- Checkpoints are only compatible with the same input feature set.
- If you toggle feature flags, retrain or use a matching checkpoint.
- Keep train/validate/infer feature logic aligned.
