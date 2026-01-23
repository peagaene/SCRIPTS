# -*- coding: utf-8 -*-
import arcpy, os

def _find_images(folder, exts, recursive):
    exts = [e.lower() for e in exts]
    imgs = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    imgs.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                imgs.append(p)
    imgs.sort(key=lambda s: s.lower())
    return imgs

def _chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

class Toolbox(object):
    def __init__(self):
        self.label = "Importar Ortofotos (Raster Dataset)"
        self.alias = "importar_ortos"
        self.tools = [ImportarOrtos]

class ImportarOrtos(object):
    def __init__(self):
        self.label = "Importar Ortofotos (Mosaic para Raster Dataset)"
        self.description = (
            "Mosaica ortofotos para um Raster Dataset existente dentro de uma File Geodatabase. "
            "Parâmetros: Mosaic Operator=LAST, Colormap Mode=FIRST, Color Matching Method=NONE. "
            "Ao final, BuildPyramids (NEAREST/DEFAULT)."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        # GDB e seleção do alvo
        p_gdb = arcpy.Parameter(
            displayName="Geodatabase (.gdb)",
            name="gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )

        p_mode = arcpy.Parameter(
            displayName="Modo do alvo",
            name="mode",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_mode.filter.type = "ValueList"
        p_mode.filter.list = ["Selecionar dataset", "Compor BLOCO_TIPO"]
        p_mode.value = "Selecionar dataset"

        p_dataset = arcpy.Parameter(
            displayName="Raster Dataset (quando selecionar dataset)",
            name="dataset",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_dataset.filter.type = "ValueList"  # será populado dinamicamente
        p_dataset.enabled = True

        p_block = arcpy.Parameter(
            displayName="BLOCO (quando compor BLOCO_TIPO)",
            name="block",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_block.enabled = False

        p_tipo = arcpy.Parameter(
            displayName="TIPO (RGB/IR)",
            name="tipo",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_tipo.filter.type = "ValueList"
        p_tipo.filter.list = ["RGB", "IR"]
        p_tipo.value = "RGB"
        p_tipo.enabled = False

        # Origem das imagens: ARQUIVOS (novo) ou PASTA (legado)
        p_inmode = arcpy.Parameter(
            displayName="Origem das ortofotos",
            name="inmode",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p_inmode.filter.type = "ValueList"
        p_inmode.filter.list = ["Selecionar ARQUIVOS", "Pasta das fotos (legado)"]
        p_inmode.value = "Selecionar ARQUIVOS"

        # Seleção de ARQUIVOS (multi)
        p_files = arcpy.Parameter(
            displayName="Arquivos de imagem (TIF/TIFF/JP2)",
            name="files",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        # Filtro de extensão na janela de seleção
        p_files.filter.list = ["tif", "tiff", "jp2"]
        p_files.enabled = True

        # PASTA (legado)
        p_photos = arcpy.Parameter(
            displayName="Pasta das fotos (se usar modo legado)",
            name="photos",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )
        p_photos.enabled = False

        p_exts = arcpy.Parameter(
            displayName="Extensões (pasta) ex.: .tif;.tiff;.jp2",
            name="exts",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_exts.value = ".tif;.tiff;.jp2"
        p_exts.enabled = False

        p_recursive = arcpy.Parameter(
            displayName="Recursivo (pasta)",
            name="recursive",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        p_recursive.value = True
        p_recursive.enabled = False

        # Execução
        p_batch = arcpy.Parameter(
            displayName="Tamanho do lote (batch size)",
            name="batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p_batch.value = 100

        p_parallel = arcpy.Parameter(
            displayName="Parallel Processing Factor (ex.: 100%, 4, vazio=desliga)",
            name="parallel",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_parallel.value = "100%"

        p_bkg = arcpy.Parameter(
            displayName="Background value a ignorar (opcional)",
            name="bkg",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_bkg.value = ""

        p_nodata = arcpy.Parameter(
            displayName="Valor NoData a aplicar (opcional)",
            name="nodata",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        p_nodata.value = ""

        p_buildpyr = arcpy.Parameter(
            displayName="Construir Pyramids ao final",
            name="build_pyramids",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        p_buildpyr.value = True

        return [p_gdb, p_mode, p_dataset, p_block, p_tipo,
                p_inmode, p_files, p_photos, p_exts, p_recursive,
                p_batch, p_parallel, p_bkg, p_nodata, p_buildpyr]

    def updateParameters(self, params):
        (p_gdb, p_mode, p_dataset, p_block, p_tipo,
         p_inmode, p_files, p_photos, p_exts, p_recursive, *_) = params

        gdb = p_gdb.valueAsText

        # Habilita campos conforme modo do ALVO
        if p_mode.value == "Selecionar dataset":
            p_dataset.enabled = True
            p_block.enabled = False
            p_tipo.enabled = False
        else:
            p_dataset.enabled = False
            p_block.enabled = True
            p_tipo.enabled = True

        # Popular a lista de datasets quando a GDB mudar
        try:
            if gdb and os.path.isdir(gdb) and gdb.lower().endswith(".gdb"):
                arcpy.env.workspace = gdb
                rds = arcpy.ListRasters() or []
                p_dataset.filter.list = sorted(rds, key=lambda s: s.lower())
        except Exception:
            pass

        # Habilita campos conforme origem das imagens
        if p_inmode.value == "Selecionar ARQUIVOS":
            p_files.enabled = True
            p_photos.enabled = False
            p_exts.enabled = False
            p_recursive.enabled = False
        else:
            p_files.enabled = False
            p_photos.enabled = True
            p_exts.enabled = True
            p_recursive.enabled = True

        return

    def updateMessages(self, params):
        (p_gdb, p_mode, p_dataset, p_block, p_tipo,
         p_inmode, p_files, p_photos, p_exts, p_recursive,
         p_batch, p_parallel, p_bkg, p_nodata, p_buildpyr) = params

        # GDB
        if p_gdb.valueAsText and (not os.path.isdir(p_gdb.valueAsText) or not p_gdb.valueAsText.lower().endswith(".gdb")):
            p_gdb.setErrorMessage("Selecione uma pasta .gdb válida.")

        # Alvo
        if p_mode.value == "Selecionar dataset":
            if p_dataset.enabled and not p_dataset.valueAsText:
                p_dataset.setErrorMessage("Escolha um Raster Dataset da lista.")
        else:
            if p_block.enabled and not p_block.valueAsText:
                p_block.setErrorMessage("Informe o BLOCO.")
            if p_tipo.enabled and not p_tipo.valueAsText:
                p_tipo.setErrorMessage("Informe o TIPO (RGB/IR).")

        # Imagens
        if p_inmode.value == "Selecionar ARQUIVOS":
            if p_files.enabled and not p_files.valueAsText:
                p_files.setErrorMessage("Selecione ao menos um arquivo de imagem (.tif/.tiff/.jp2).")
        else:
            if p_exts.value:
                bad = [e for e in str(p_exts.value).split(";") if e and not e.startswith(".")]
                if bad:
                    p_exts.setErrorMessage("Cada extensão deve começar com ponto. Ex.: .tif;.jp2")
            if p_photos.enabled and p_photos.valueAsText and not os.path.isdir(p_photos.valueAsText):
                p_photos.setErrorMessage("Pasta de fotos inválida.")

        return

    def execute(self, parameters, messages):
        (p_gdb, p_mode, p_dataset, p_block, p_tipo,
         p_inmode, p_files, p_photos, p_exts, p_recursive,
         p_batch, p_parallel, p_bkg, p_nodata, p_buildpyr) = parameters

        # Coleta de parâmetros
        gdb = p_gdb.valueAsText
        mode = p_mode.valueAsText
        dataset_sel = p_dataset.valueAsText
        block = (p_block.valueAsText or "").strip() if p_block.valueAsText else ""
        tipo = (p_tipo.valueAsText or "").strip() if p_tipo.valueAsText else ""

        inmode = p_inmode.valueAsText
        files_text = p_files.valueAsText  # multiValue → string separada por ';'
        photos = p_photos.valueAsText
        exts = [e.strip() for e in (p_exts.valueAsText or ".tif;.tiff;.jp2").split(";") if e.strip()]
        recursive = bool(p_recursive.value)

        batch_size = int(p_batch.value) if p_batch.value else 100
        parallel = (p_parallel.valueAsText or "").strip()
        bkg = (p_bkg.valueAsText or "").strip()
        nodata = (p_nodata.valueAsText or "").strip()
        build_pyr = bool(p_buildpyr.value)

        arcpy.AddMessage(f"GDB: {gdb}")

        # Determinar o alvo (Raster Dataset)
        if mode == "Selecionar dataset":
            target = os.path.join(gdb, dataset_sel)
        else:
            cand1 = os.path.join(gdb, f"{block}_{tipo}")
            cand2 = os.path.join(gdb, f"{block}")
            target = cand1 if arcpy.Exists(cand1) else cand2

        if not arcpy.Exists(target):
            raise arcpy.ExecuteError(f"Raster Dataset alvo não encontrado: {target}")

        # Montar lista de imagens conforme a origem
        if inmode == "Selecionar ARQUIVOS":
            imgs = [s for s in (files_text.split(";") if files_text else []) if s]
            if not imgs:
                raise arcpy.ExecuteError("Nenhum arquivo selecionado.")
        else:
            if not photos or not os.path.isdir(photos):
                raise arcpy.ExecuteError(f"Pasta de fotos inválida: {photos}")
            imgs = _find_images(photos, exts, recursive)
            if not imgs:
                raise arcpy.ExecuteError(f"Nenhuma imagem encontrada em {photos} com extensões {exts}")

        if parallel:
            arcpy.env.parallelProcessingFactor = parallel
            arcpy.AddMessage(f"parallelProcessingFactor = {arcpy.env.parallelProcessingFactor}")

        arcpy.AddMessage(f"Target: {target}")
        arcpy.AddMessage(f"Total de imagens: {len(imgs)} | Lote: {batch_size}")
        arcpy.SetProgressor("step", "Mosaicking...", 0, len(imgs), batch_size)

        total = 0
        for batch in _chunker(imgs, batch_size):
            in_list = ";".join(batch)
            arcpy.management.Mosaic(
                in_list,
                target,
                "LAST",     # Mosaic Operator
                "FIRST",    # Colormap Mode
                bkg or "",
                nodata or "",
                "",         # onebit_to_eightbit
                "",         # mosaicking_tolerance
                "NONE"      # Color Matching Method
            )
            total += len(batch)
            arcpy.AddMessage(f"  ✓ Lote concluído (acumulado: {total})")
            arcpy.SetProgressorPosition(total)

        if build_pyr:
            arcpy.AddMessage("Construindo Pyramids (NEAREST / DEFAULT, SKIP_EXISTING)...")
            arcpy.management.BuildPyramids(
                target,
                pyramid_level="-1",
                SKIP_FIRST="NONE",
                resample_technique="NEAREST",
                compression_type="DEFAULT",
                compression_quality="",
                skip_existing="SKIP_EXISTING",
            )

        arcpy.ResetProgressor()
        arcpy.AddMessage(f"Finalizado. Imagens mosaificadas: {total}")
        return
