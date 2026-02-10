# Guia Completo de Interpretacao e Evolucao do Projeto LiDAR (spconv)

Este documento e um guia de estudo do projeto `lidar_sparse_seg` para voce entender o que cada parte faz, como interpretar resultados, e como evoluir o pipeline com seguranca.

## 1. Objetivo do projeto

Segmentacao semantica 3D em nuvem de pontos LAS/LAZ com rede sparse (spconv), prevendo classe por ponto/voxel.

Configuracao atual de referencia:

- Classes de treino (internas):
  - `0 = terreno`
  - `1 = edificacao`
  - `2 = vegetacao`
- Classe LAS `7` (noise): ignorada.
- Features ativas no melhor setup:
  - `z_rel`
  - `intensity_norm`
  - `HAG`
  - `return_number/number_of_returns`
  - `scan_angle`
  - `normal_z`
  - `roughness`

## 2. Estrutura de codigo (quem faz o que)

## `src/train.py`

Responsavel por:

- ler config (`yaml`)
- gerar/usar split
- montar dataset e dataloader
- montar modelo (`SparseUNet`)
- treinar por epoca
- validar no split val a cada epoca
- salvar `last.pth` e `best_mIoU.pth`
- salvar logs (`history.csv`, TensorBoard)

Argumentos importantes:

- `--config`: caminho do YAML
- `--resume_checkpoint`: checkpoint para retomar
- `--resume_weights_only`: carrega somente pesos (recomendado quando muda dataset)

## `src/validate.py`

Responsavel por:

- carregar checkpoint
- rodar no split (`val` ou `test`)
- gerar metricas:
  - `overall_accuracy`
  - `IoU por classe`
  - `mIoU`
  - `confusion_matrix`

Saidas:

- `metrics_*.json`
- `summary_*.csv`
- `confusion_*.csv`

## `src/infer.py`

Responsavel por:

- carregar checkpoint
- ler LAS/LAZ
- inferir por blocos (`block_size`)
- devolver LAS/LAZ classificado

Pontos importantes:

- classe `7` preservada
- mapeamento treino -> LAS feito por `las_to_train.json`

## `src/datasets/lidar_dataset.py`

Core do pre-processamento de treino/val:

- leitura de pontos
- filtro de classes ignoradas
- mapeamento LAS -> classes internas
- crop (`crop_size`)
- normalizacao (`z_rel`, intensidade)
- calculo de HAG
- calculo de features extras (returns, scan angle, normal/roughness)
- voxelizacao

## `src/utils/io_las.py`

Leitura/escrita LAS/LAZ:

- atributos base: `x, y, z, intensity, classification`
- atributos extras: `return_number`, `number_of_returns`, `scan_angle` (ou fallback)

## `src/utils/geom_features.py`

Features geometricas locais:

- `normal_z` (magnitude do componente Z da normal local)
- `roughness` (desvio local ao plano)

## `src/utils/voxelize.py`

- quantizacao de coordenadas
- agregacao de features por voxel (media)
- agregacao de label por voxel (maioria)
- collate para batch sparse

## `src/models/minkunet.py`

Apesar do nome, backend atual e `spconv`.
Define a arquitetura sparse que consome features e devolve logits por voxel.

## 3. Fluxo de dados completo

1. Arquivo LAS/LAZ entra.
2. Remove classe 7.
3. Mapeia classes LAS para label interna.
4. Faz crop no XY (treino/val).
5. Normaliza:
   - XY centralizado
   - `z_rel = z - z_min`
   - intensidade para `[0,1]`
6. Calcula features extras ativas.
7. Voxeliza e agrega.
8. Modelo prediz logits.
9. Loss CE + pesos por classe.
10. Validacao calcula `mIoU/IoU/acc`.

## 4. Entendendo o YAML (campo por campo)

## Bloco geral

- `seed`: reproducibilidade.
- `num_classes`: quantidade de classes alvo da rede.
- `voxel_size`: resolucao espacial do voxel (m).
  - menor = mais detalhe, mais custo.
- `max_intensity`: escala para normalizacao da intensidade.

## Features

- `use_hag`: usa altura acima do terreno.
- `hag_cell_size`: resolucao do terreno para HAG.
- `use_return_features`: inclui returns.
- `use_scan_angle`: inclui angulo de varredura.
- `use_normal_features`: inclui `normal_z`.
- `use_roughness_feature`: inclui rugosidade.
- `normal_cell_size`: celula XY para normal/roughness.
- `normal_min_points`: minimo de pontos por celula.
- `roughness_scale`: normalizacao da rugosidade.

## Amostragem

- `crop_size`: tamanho da janela XY (m).
- `max_points_per_crop`: limite de pontos por crop.

## Treino

- `batch_size`: lotes (1 e comum em LiDAR grande).
- `epochs`: numero de epocas.
- `lr`: learning rate.
- `weight_decay`: regularizacao.
- `scheduler`: `cosine` ou `onecycle`.
- `amp`: mixed precision.

## Split

- `auto_generate`: gera split automatico.
- `recursive`: busca recursiva em `data_root`.
- `train_ratio/val_ratio/test_ratio`: proporcoes.
- `seed`: seed do sorteio de split.
- `regenerate`:
  - `true`: recria split
  - `false`: reaproveita split existente

## Model

- `base_channels`: largura da rede.
- `depth`: profundidade dos blocos.

## Paths

- `data_root`: pasta com dados.
- `train_split/val_split/test_split`: txt de split.
- `las_to_train`: mapeamento LAS->train.
- `classes`: nomes de classes.
- `ignore_las_classes`: classes ignoradas (ex.: 7).
- `checkpoints_dir/logs_dir`: saidas de treino.

## 5. Como interpretar resultados

Prioridade:

1. `test mIoU`
2. `IoU_edificacao`
3. `IoU` das demais classes
4. estabilidade val vs test

Sinais:

- `val ~ test` => boa generalizacao
- `test << val` => overfit/split ruim
- `IoU_edificacao` baixo com outros altos => confusao edif x vegetacao

## 6. Quando ajustar cada parametro

- Erro de fronteira/fragmentacao: aumentar `crop_size`.
- Perda de detalhe: reduzir `voxel_size` (com cuidado).
- Instabilidade: reduzir `lr`.
- Pouca capacidade: subir `base_channels`.
- Edificacao confunde com vegetacao:
  - manter HAG
  - manter normal/roughness
  - adicionar mais dados com telhados variados

## 7. Como adicionar nova feature corretamente

Regra de ouro: alterar treino/val/infer juntos.

Checklist:

1. Ler atributo (ou calcular) em `dataset`.
2. Normalizar feature para escala razoavel.
3. Adicionar ao `feat_list` no dataset.
4. Adicionar o mesmo calculo no `infer.py`.
5. Atualizar `get_input_channels(...)`.
6. Expor flags no YAML.
7. Rodar experimento controlado (ablation).

## 8. Como remover feature sem quebrar

1. Desativar flag no YAML.
2. Garantir que `get_input_channels` reflete isso.
3. Nao misturar checkpoint antigo com in_channels diferente.

Regra:

- Checkpoint so e compativel com o mesmo conjunto de features (mesmo numero de canais).

## 9. Fine-tune vs treino do zero

## Treino do zero

- mais limpo para comparar configuracoes.
- melhor quando mudou muito o dataset/objetivo.

## Fine-tune

- mais rapido quando base ja e boa.
- para dataset novo, preferir:
  - `--resume_checkpoint ... --resume_weights_only`

Por que:

- carrega conhecimento dos pesos
- evita herdar `best_mIoU` antigo e bloquear novo best

## 10. Protocolo recomendado para experimentos

1. Fixar split (`regenerate: false` apos gerar).
2. Mudar 1 variavel por vez.
3. Rodar 15-20 epocas para triagem.
4. Promover top-1 para 40+ epocas.
5. Comparar sempre:
   - `test mIoU`
   - `IoU_edificacao`
   - inspeção visual

## 11. Avaliacao visual (indispensavel)

Conferir:

- telhados parcialmente vegetacao
- solo com pontos flutuantes
- bordas de edificacao
- areas densas de vegetacao baixa

Se metricas sobem mas visual piora, revisar objetivo/prioridade da classe critica.

## 12. Comandos uteis

Treino:

```powershell
python -m src.train --config configs/default.yaml
```

Fine-tune (pesos apenas):

```powershell
python -m src.train --config configs/finetune_lote09_warmstart.yaml --resume_checkpoint E:/training/checkpoints_3classes_v05/best_mIoU.pth --resume_weights_only
```

Validacao:

```powershell
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes_v05/best_mIoU.pth --split val
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes_v05/best_mIoU.pth --split test
```

Inferencia 1 tile:

```powershell
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes_v05/best_mIoU.pth --input E:/training/inf/input/197039.laz --output E:/training/inf/output/197039_pred.laz
```

## 13. O que e essencial x opcional

Essencial:

- split confiavel
- labels consistentes
- mapeamento correto
- validacao em test
- inspeção visual

Opcional (mas poderoso):

- fine-tune com warmstart
- features geometricas extras
- pos-processamento na inferencia
- ensemble

## 14. Erros comuns

- comparar modelos com split diferente
- usar `last.pth` em vez de `best_mIoU.pth`
- mudar features e reutilizar checkpoint antigo
- regenerar split sem perceber (`regenerate: true`)

## 15. Direcao de evolucao do seu projeto

1. consolidar baseline atual (ja forte)
2. fine-tune em dados novos com `resume_weights_only`
3. testar opcao 2-classes mantendo ground fixo (sem substituir pipeline principal)
4. comparar tecnicamente e decidir por evidencia

---

Se quiser, o proximo passo e eu gerar um `TEMPLATE_EXPERIMENTOS.csv` para voce registrar cada rodada (config, data, mIoU, IoU por classe, observacao visual, decisao).
