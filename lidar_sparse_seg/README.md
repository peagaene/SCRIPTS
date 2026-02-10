# LiDAR Aerial Semantic Segmentation (3D Sparse CNN + spconv)

Projeto para treino, validacao e inferencia de segmentacao semantica 3D em LAS/LAZ usando PyTorch + spconv.

## Classes usadas no treino (config atual)

Mapeamento LAS para labels internas:

- `2 -> 0` (`terreno`)
- `6 -> 1` (`edificacao`)
- `3, 4, 5, 13 -> 2` (`vegetacao`)
- `7` (`noise`) removida de treino/val/test

Arquivos:

- `configs/classes.json`
- `configs/las_to_train.json`
- `configs/ignore_las_classes.json`

## Estrutura

- `configs/default.yaml`
- `src/datasets/lidar_dataset.py`
- `src/models/minkunet.py` (backend spconv)
- `src/utils/io_las.py`
- `src/utils/voxelize.py`
- `src/utils/spconv_utils.py`
- `src/utils/splits.py`
- `src/utils/metrics.py`
- `src/utils/repro.py`
- `src/train.py`
- `src/validate.py`
- `src/infer.py`

## Instalacao

Python 3.10+ recomendado.

```bash
pip install -r requirements.txt
```

Observacoes:

- Para `*.laz`, use backend `lazrs`.
- `requirements.txt` usa `spconv-cu121`. Se seu CUDA for diferente, ajuste para a variante correta.

## Split automatico (sem editar .txt manualmente)

No `configs/default.yaml`, bloco `split`:

- `auto_generate: true`
- `train_ratio`, `val_ratio`, `test_ratio`
- `seed`
- `regenerate`

Comportamento:

- Se os arquivos de split ja existirem, eles sao reaproveitados.
- Se `regenerate: true`, os splits sao recriados aleatoriamente a partir de `paths.data_root`.

Dica quando adicionar novos `.laz`:

1. adicione os arquivos em `paths.data_root`
2. mude temporariamente `split.regenerate: true`
3. rode `train`
4. volte `split.regenerate: false` para fixar a divisao

## Pre-processamento implementado

- Remove classe `7` antes de treino/val/test.
- Normalizacao por tile/crop:
  - XY centralizado por media.
  - Z relativo (`z - z_min`).
- Intensidade normalizada para `[0,1]`.
- Voxelizacao com `voxel_size` (padrao `0.5 m`).
- Features base: `[z_rel, intensity_norm]`.
- Opcional: HAG (`use_hag`).

## Treino

```bash
python -m src.train --config configs/default.yaml
```

Saidas em `paths.checkpoints_dir` (config atual: `E:/training/checkpoints_3classes`):

- `best_mIoU.pth`
- `last.pth`
- `logs/tb` (TensorBoard)
- `logs/history.csv`
- `run_artifacts/config_resolved.yaml`
- `run_artifacts/class_stats.csv`

## Validacao/Teste

```bash
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes/best_mIoU.pth --split val
```

Saidas padrao em `E:/training/checkpoints_3classes/reports`.

## Inferencia

Arquivo unico:

```bash
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes/best_mIoU.pth --input E:/training/data_root/222027.laz --output E:/training/checkpoints_3classes/preds/222027_pred.laz
```

Comportamento no output:

- classe `7` original e preservada
- labels internas convertidas para LAS via `las_to_train.json`
