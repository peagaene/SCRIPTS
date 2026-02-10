# Guia Pratico de Interpretacao de Treino LiDAR (Projeto Atual)

Este guia explica como interpretar os valores do seu treinamento e decidir o que ajustar em seguida.

## 1) Ordem de prioridade das metricas

Sempre olhe nesta ordem:

1. `val mIoU` (principal)
2. `IoU por classe` (principal para diagnostico)
3. `val loss`
4. `train loss`
5. `overall accuracy` (apoio)

Regra:

- Melhor modelo = maior `val mIoU`.
- Se `accuracy` sobe mas `mIoU` nao sobe, geralmente o modelo esta favorecendo classes mais frequentes.

## 2) Como ler os logs da sua execucao

Voce ja tem:

- Console por epoca (`train_loss`, `val_loss`, `acc`, `mIoU`, `IoU por classe`)
- `E:/training/checkpoints_3classes/logs/history.csv`
- TensorBoard em `E:/training/checkpoints_3classes/logs/tb`
- `class_stats.csv` com contagem e peso por classe

No seu caso (3 classes), padrao saudavel:

- `train_loss` cai ou oscila levemente para baixo
- `val_loss` cai no geral
- `mIoU` sobe nas primeiras epocas e depois entra em plato

## 3) Sinais de problema e o que fazer

### Caso A: `train_loss` cai, mas `val mIoU` nao sobe

Possivel causa: overfitting ou split ruim.

Acoes:

1. reduzir complexidade (`model.base_channels` menor)
2. aumentar dados de treino (mais `.laz`)
3. revisar split (regenerar com outro `split.seed`)

### Caso B: classe `edificacao` com IoU baixo persistente

Possivel causa: poucos exemplos/contexto insuficiente.

Acoes:

1. testar `voxel_size: 0.4`
2. testar `use_hag: true`
3. aumentar `crop_size` (ex.: 96)
4. adicionar mais tiles com edificacao no treino

### Caso C: treino muito instavel (sobe e desce forte)

Acoes:

1. diminuir `lr` para `0.0005`
2. manter `batch_size: 1`
3. manter `scheduler: cosine`

### Caso D: mIoU trava cedo

Acoes:

1. rodar mais epocas (80-120)
2. testar `voxel_size` (0.4 / 0.5 / 0.6)
3. mudar apenas 1 variavel por experimento

## 4) Como escolher hiperparametros no seu projeto

## `voxel_size`

- `0.4`: mais detalhe, mais VRAM/tempo
- `0.5`: equilibrio (padrao recomendado)
- `0.6`: mais rapido, pode perder detalhe

Regra pratica:

- Se edificacao ruim, tente `0.4`.
- Se VRAM apertada, tente `0.6`.

## `batch_size`

- Comece em `1` (estavel para nuvem 3D grande).
- So aumente se houver VRAM sobrando e tempo de treino estiver alto.

## `epochs`

- Teste rapido: 2-5
- Treino real: 60-100
- Pare pelo `best_mIoU.pth` (nao pelo `last.pth` necessariamente)

## `lr`

- Base: `0.001`
- Se instavel: `0.0005`
- Se muito lento: testar `0.0015` com cuidado

## 5) Como comparar experimentos corretamente

Sempre manter fixo:

- mesmo `seed`
- mesmo split
- mesma versao de dados

Mudar 1 item por vez:

1. `voxel_size`
2. depois `use_hag`
3. depois `lr`

Nomeie experimentos por pasta de checkpoint, por exemplo:

- `E:/training/checkpoints_3classes_v05`
- `E:/training/checkpoints_3classes_v04_hag`

## 6) Checklist de decisao por rodada

Depois de cada treino, responda:

1. `best mIoU` melhorou em relacao ao experimento anterior?
2. IoU de `edificacao` melhorou?
3. O ganho foi consistente (nao apenas 1 epoca isolada)?
4. O tempo/custo de treino ainda compensa?

Se 1-3 = sim, mantenha mudanca.
Se 1 = nao e 2 = nao, reverta e teste outro parametro.

## 7) Pipeline recomendado para voce (agora)

1. finalizar treino atual (80 epocas)
2. validar com `best_mIoU.pth`
3. rodar inferencia em 2-3 tiles de controle
4. analisar erros visuais (principalmente edificacao vs vegetacao)
5. proximo experimento: `voxel_size=0.4` mantendo todo o resto igual

## 8) Comandos uteis

Treino:

```powershell
python -m src.train --config configs/default.yaml
```

Validacao:

```powershell
python -m src.validate --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes/best_mIoU.pth --split val
```

Inferencia:

```powershell
python -m src.infer --config configs/default.yaml --checkpoint E:/training/checkpoints_3classes/best_mIoU.pth --input E:/training/data_root --output E:/training/checkpoints_3classes/preds
```

---

Se quiser, o proximo passo e eu gerar tambem um template de planilha CSV para registrar experimentos (`experimento, voxel_size, lr, best_mIoU, IoU_terreno, IoU_edificacao, IoU_vegetacao, observacoes`).
