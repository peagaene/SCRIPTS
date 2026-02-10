# Guia Rapido de Experimentos Operacionais

Este guia complementa `GUIA_COMPLETO_PROJETO.md` com foco em reduzir revisao manual.

## 1) Arquivo de controle

Use `TEMPLATE_EXPERIMENTOS.csv` para registrar cada rodada.

Campos mais importantes para decisao:

- `test_mIoU`
- `test_IoU_edificacao`
- `test_F1_edificacao`
- `conf_edif_para_veg_test`
- `fragmentacao_edificacao_score`
- `tempo_revisao_estimado_min_por_tile`

## 2) Metricas operacionais incorporadas

O `validate.py` agora salva e imprime:

- `f1_per_class`
- `confusion_edificacao_para_vegetacao_rate`
- `confusion_vegetacao_para_edificacao_rate`

Esses campos vao para:

- `reports/metrics_val.json`
- `reports/metrics_test.json`
- `reports/summary_val.csv`
- `reports/summary_test.csv`

## 3) Como usar na pratica

1. Treine um experimento.
2. Rode validacao em `val` e `test`.
3. Copie os numeros principais para o CSV.
4. Adicione observacao visual objetiva (2-3 linhas).
5. Marque a coluna `decisao`:
   - `promover`
   - `retestar`
   - `descartar`

## 4) Regra de aprovacao sugerida

Promover experimento quando:

- `test_mIoU` nao cai
- `test_IoU_edificacao` sobe
- `conf_edif_para_veg_test` cai
- visual mostra menos fragmentacao

Se `mIoU` sobe mas confusao de edificacao piora, nao promover automaticamente.

## 5) Proximas incorporacoes possiveis

- proxy de fragmentacao por componentes conectados
- proxy de borda (Boundary F1 simplificado)
- script automatico para consolidar varios `metrics_*.json` em uma tabela unica
