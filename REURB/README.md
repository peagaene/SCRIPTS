# REURB (Modular)

Sistema de processamento geoespacial para projetos REURB (TXT/DXF/MDT) com UI Tkinter.

## Estrutura

```
reurb/
  config/        Constantes, parâmetros e mapeamentos
  geometry/      Cálculos geométricos
  io/            Leitura de TXT/DXF/MDT/SHP
  processors/    Processadores de regras de negócio
  renderers/     Renderização de textos/tabelas/blocos
  symbology/     Perfis de layers
  ui/            UI Tkinter
  utils/         Logging e resource managers
```

## Como rodar

### Via Anaconda env `geo_env`

```
cd /d D:\SCRIPTS\REURB
C:\Users\compartilhar\anaconda3\Scripts\activate.bat geo_env
python -m reurb.main
```

### Via launcher

Duplo clique em `run_reurb.bat`.

## Dependencias principais

- Tkinter (UI, ja incluso no Python)
- ezdxf
- shapely
- rasterio
- gdal (osgeo)

## Compatibilidade

O pacote `reurb` re-exporta símbolos públicos para compatibilidade com imports legados.

## Testes

```
pytest -q
```
