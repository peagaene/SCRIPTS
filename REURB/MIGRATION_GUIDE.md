# Migration Guide

## Objetivo

Esta refatoração mantém 100% de compatibilidade funcional, com estrutura modular.

## Código legado

Imports antigos continuam funcionando:

```python
from reurb import LAYER_LOTES, Params, ler_txt
```

## Código novo recomendado

```python
from reurb.processors.txt_blocks import TxtBlockProcessor
from reurb.ui import abrir_ui
```

## Entry point

Use:

```
python -m reurb.main
```

Ou o `run_reurb.bat`.

## Notas

- `reurb_auto_all.py` agora é um shim que chama `reurb.main`.
- As mudanças são internas e não alteram formatos de entrada/saída.
- A UI permanece em Tkinter.
