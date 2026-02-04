import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import laspy
import numpy as np

FLIGHTLINES_PATCH = (111, 211, 311, 411, 611, 511, 711, 811, 911, 1011, 1111, 1211, 1311)
CLASSES_REFERENCIA: Optional[Tuple[int, ...]] = None  # None = todas as classes
GROUP_BY_ATTRIBUTES: Sequence[str] = ("classification", "return_number", "number_of_returns")
MIN_POINTS_PER_BIN = 500
PERCENTIS_LOG = (5, 95)
PERCENTIS_CURVA = (5, 25, 50, 75, 95)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normaliza intensidades dos flightlines especificados.")
    parser.add_argument(
        "--patch",
        nargs="+",
        help="Arquivo(s) .laz que terao intensidades corrigidas. Se omitido, usa a forma antiga ou o padrao.",
    )
    parser.add_argument(
        "--refs",
        nargs="*",
        default=[],
        help="Arquivos adicionais usados apenas como referencia estatistica (nao sao alterados).",
    )
    parser.add_argument(
        "arquivos",
        nargs="*",
        help="Modo abreviado: lista de arquivos a corrigir (equivalente a --patch).",
    )
    return parser.parse_args()


def _pick_reference_files_gui() -> List[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - tkinter indisponivel
        print(f"Interface grafica indisponivel ({exc}); prosseguindo sem selecionar referencias.")
        return []

    root = tk.Tk()
    root.withdraw()
    root.update()
    print("Selecione os arquivos LAZ de referencia (Ctrl+clique para varios). Feche a janela para concluir.")
    filenames = filedialog.askopenfilenames(
        title="Selecione arquivos LAZ de referencia", filetypes=[("Arquivos LAZ", "*.laz"), ("Todos", "*.*")]
    )
    root.destroy()
    return [Path(name) for name in filenames]


def _pick_patch_files_gui() -> List[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - tkinter indisponivel
        print(f"Interface grafica indisponivel ({exc}); prosseguindo com caminho padrao.")
        return []

    root = tk.Tk()
    root.withdraw()
    root.update()
    print("Selecione os arquivos LAZ que terao as intensidades corrigidas. Feche a janela para concluir.")
    filenames = filedialog.askopenfilenames(
        title="Selecione arquivos LAZ para corrigir", filetypes=[("Arquivos LAZ", "*.laz"), ("Todos", "*.*")]
    )
    root.destroy()
    return [Path(name) for name in filenames]


def _run_gui_queue() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Interface grafica indisponivel: {exc}")

    root = tk.Tk()
    root.title("Normalizacao de Intensidade (LAZ)")
    root.geometry("620x480")

    refs: List[Path] = []
    queue: List[Path] = []
    processing = False

    # Placeholders for widgets to acalmar type-checker
    btn_refs: "tk.Button"
    btn_add: "tk.Button"
    btn_clear: "tk.Button"
    btn_process: "tk.Button"
    lbl_refs: "tk.Label"
    lbl_queue_count: "tk.Label"
    lst_queue: "tk.Listbox"

    def pick_refs() -> None:
        nonlocal refs
        filenames = filedialog.askopenfilenames(
            title="Selecione arquivos LAZ de referencia",
            filetypes=[("Arquivos LAZ", "*.laz"), ("Todos", "*.*")],
        )
        if filenames:
            refs = [Path(f) for f in filenames]
            lbl_refs.config(text=f"{len(refs)} referencia(s) selecionada(s)")

    def add_patches() -> None:
        nonlocal queue
        filenames = filedialog.askopenfilenames(
            title="Selecione arquivos LAZ para corrigir",
            filetypes=[("Arquivos LAZ", "*.laz"), ("Todos", "*.*")],
        )
        new_paths = [Path(f) for f in filenames if f]
        if new_paths:
            queue.extend(new_paths)
            refresh_queue()

    def clear_queue() -> None:
        nonlocal queue
        queue = []
        refresh_queue()

    def refresh_queue() -> None:
        lst_queue.delete(0, tk.END)
        for p in queue:
            lst_queue.insert(tk.END, str(p))
        lbl_queue_count.config(text=f"{len(queue)} arquivo(s) na fila")

    def set_buttons_state(disabled: bool) -> None:
        state = tk.DISABLED if disabled else tk.NORMAL
        btn_refs.config(state=state)
        btn_add.config(state=state)
        btn_clear.config(state=state)
        btn_process.config(state=state)

    def process_queue() -> None:
        nonlocal queue, processing
        if processing:
            return
        if not refs:
            messagebox.showerror("Erro", "Selecione ao menos um arquivo de referencia.")
            return
        if not queue:
            messagebox.showinfo("Aviso", "Fila vazia, nada a processar.")
            return

        processing = True
        set_buttons_state(True)

        # Copia fila e limpa lista antes de iniciar
        work_queue = list(queue)
        queue = []
        refresh_queue()

        def worker():
            nonlocal processing
            try:
                for caminho in [*refs, *work_queue]:
                    if not caminho.exists():
                        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

                print(f"{len(refs)} arquivo(s) de referencia carregados (serao reutilizados).")
                ref_stats, ref_curve_global, _ = _collect_reference_samples(refs)

                for caminho in work_queue:
                    _process_file(caminho, ref_stats, ref_curve_global)

                root.after(0, lambda: messagebox.showinfo("Concluido", "Processamento finalizado."))
            except Exception as exc:
                root.after(0, lambda exc=exc: messagebox.showerror("Erro", str(exc)))
            finally:
                processing = False
                root.after(0, lambda: set_buttons_state(False))

        threading.Thread(target=worker, daemon=True).start()

    frame_refs = tk.Frame(root, pady=10, padx=10)
    frame_refs.pack(fill=tk.X)
    tk.Button(frame_refs, text="Selecionar referencias", command=pick_refs).pack(side=tk.LEFT)
    lbl_refs = tk.Label(frame_refs, text="0 referencia(s) selecionada(s)")
    lbl_refs.pack(side=tk.LEFT, padx=10)

    frame_queue = tk.Frame(root, pady=10, padx=10)
    frame_queue.pack(fill=tk.BOTH, expand=True)
    btn_add = tk.Button(frame_queue, text="Adicionar patches", command=add_patches)
    btn_add.pack(side=tk.TOP, anchor="w")
    lst_queue = tk.Listbox(frame_queue, height=15)
    lst_queue.pack(fill=tk.BOTH, expand=True, pady=5)
    lbl_queue_count = tk.Label(frame_queue, text="0 arquivo(s) na fila")
    lbl_queue_count.pack(anchor="w")

    frame_actions = tk.Frame(root, pady=10, padx=10)
    frame_actions.pack(fill=tk.X)
    tk.Button(frame_actions, text="Limpar fila", command=clear_queue).pack(side=tk.LEFT)
    tk.Button(frame_actions, text="Processar fila", command=process_queue).pack(side=tk.RIGHT)

    refresh_queue()
    root.mainloop()


def _get_attribute_arrays(las_file: laspy.LasData, names: Sequence[str], context: str) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    missing: List[str] = []
    for name in names:
        if hasattr(las_file, name):
            arrays.append(np.asarray(getattr(las_file, name)))
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Atributos ausentes no arquivo {context}: {', '.join(missing)}")
    return arrays


def _compute_bin_percentiles(
    intensity_arr: np.ndarray,
    attribute_arrays: Sequence[np.ndarray],
    mask: np.ndarray,
    percentis: Sequence[int],
    min_points: int,
) -> Dict[Tuple[int, ...], Tuple[np.ndarray, int]]:
    if not np.any(mask) or not attribute_arrays:
        return {}

    stacked = np.column_stack([attr[mask] for attr in attribute_arrays])
    samples = intensity_arr[mask]
    keys, inverse = np.unique(stacked, axis=0, return_inverse=True)
    stats: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}

    for idx, key_vals in enumerate(keys):
        subset = samples[inverse == idx]
        if subset.size < min_points:
            continue
        curve = np.percentile(subset, percentis)
        stats[tuple(int(v) for v in key_vals)] = (curve.astype(np.float64), int(subset.size))

    return stats


def _build_class_mask(class_array: np.ndarray, allowed: Optional[Tuple[int, ...]]) -> np.ndarray:
    if allowed is None:
        return np.ones(class_array.shape, dtype=bool)
    return np.isin(class_array, allowed)


def _build_transforms(
    ref_stats: Dict[Tuple[int, ...], Tuple[np.ndarray, int]],
    patch_stats: Dict[Tuple[int, ...], Tuple[np.ndarray, int]],
) -> Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]]:
    transforms: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}
    shared_keys = ref_stats.keys() & patch_stats.keys()

    for key in shared_keys:
        ref_curve, _ = ref_stats[key]
        patch_curve, _ = patch_stats[key]

        if patch_curve[-1] <= patch_curve[0]:
            continue

        transforms[key] = (patch_curve, ref_curve)

    return transforms


def _map_curve(values: np.ndarray, src_curve: np.ndarray, dst_curve: np.ndarray) -> np.ndarray:
    src = np.asarray(src_curve, dtype=np.float64)
    dst = np.asarray(dst_curve, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)

    if src.size != dst.size or src.size < 2:
        return vals

    src_span = src[-1] - src[0]
    if src_span == 0:
        return vals - src[0] + dst[0]

    clipped = np.clip(vals, src[0], src[-1])
    mapped = np.interp(clipped, src, dst)

    below = vals < src[0]
    above = vals > src[-1]

    if below.any():
        low_idx = 1 if src.size > 2 else 1
        delta_low = src[low_idx] - src[0]
        if delta_low == 0:
            slope = (dst[-1] - dst[0]) / max(src[-1] - src[0], 1e-6)
        else:
            slope = (dst[low_idx] - dst[0]) / delta_low
        mapped[below] = dst[0] + (vals[below] - src[0]) * slope

    if above.any():
        high_idx = -2 if src.size > 2 else 0
        delta_high = src[-1] - src[high_idx]
        if delta_high == 0:
            slope = (dst[-1] - dst[0]) / max(src[-1] - src[0], 1e-6)
        else:
            slope = (dst[-1] - dst[high_idx]) / delta_high
        mapped[above] = dst[-1] + (vals[above] - src[-1]) * slope

    return mapped


def _apply_patch_correction(
    intensity_arr: np.ndarray,
    mask_patch: np.ndarray,
    attribute_arrays: Sequence[np.ndarray],
    transforms: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]],
    default_curve: Tuple[np.ndarray, np.ndarray],
) -> None:
    if not np.any(mask_patch):
        return

    default_src, default_dst = default_curve

    if not attribute_arrays:
        intensity_arr[mask_patch] = _map_curve(intensity_arr[mask_patch], default_src, default_dst)
        return

    stacked = np.column_stack([attr[mask_patch] for attr in attribute_arrays])
    idx_patch = np.where(mask_patch)[0]
    unique_keys, inverse = np.unique(stacked, axis=0, return_inverse=True)

    for key_idx, key_vals in enumerate(unique_keys):
        indices = idx_patch[inverse == key_idx]
        curve = transforms.get(tuple(int(v) for v in key_vals), default_curve)
        src_curve, dst_curve = curve
        intensity_arr[indices] = _map_curve(intensity_arr[indices], src_curve, dst_curve)


def _collect_reference_samples(paths: Sequence[Path]) -> Tuple[
    Dict[Tuple[int, ...], Tuple[np.ndarray, int]], np.ndarray, Tuple[float, float]
]:
    attr_count = len(GROUP_BY_ATTRIBUTES)
    ref_intensity_chunks: List[np.ndarray] = []
    ref_attr_chunks: List[List[np.ndarray]] = [list() for _ in range(attr_count)]
    total_ref = 0

    for path in paths:
        print(f"Lendo {path} para referencia...")
        las = laspy.read(path)
        intensity = las.intensity.astype(np.float64)
        flightline = las.point_source_id
        classes = las.classification
        class_mask = _build_class_mask(classes, CLASSES_REFERENCIA)
        mask_ref = class_mask & ~np.isin(flightline, FLIGHTLINES_PATCH)
        count = np.count_nonzero(mask_ref)
        print(f"  Pontos referencia neste arquivo: {count}")
        if count:
            ref_intensity_chunks.append(intensity[mask_ref])
            if attr_count:
                attribute_arrays = _get_attribute_arrays(las, GROUP_BY_ATTRIBUTES, context=str(path))
                for idx, arr in enumerate(attribute_arrays):
                    ref_attr_chunks[idx].append(arr[mask_ref])
        total_ref += count
        del las

    if total_ref == 0:
        raise RuntimeError("Sem pontos de referencia nos arquivos fornecidos.")

    ref_intensity = np.concatenate(ref_intensity_chunks)
    mask_all = np.ones(ref_intensity.shape, dtype=bool)

    if attr_count:
        ref_attributes = [np.concatenate(chunks) for chunks in ref_attr_chunks]
    else:
        ref_attributes = []

    ref_stats = _compute_bin_percentiles(ref_intensity, ref_attributes, mask_all, PERCENTIS_CURVA, MIN_POINTS_PER_BIN)
    ref_curve_global = np.percentile(ref_intensity, PERCENTIS_CURVA)
    p_ref_low, p_ref_high = np.percentile(ref_intensity, PERCENTIS_LOG)
    print(f"Total referencia acumulado: {ref_intensity.size}")
    print(f"Ref   {PERCENTIS_LOG[0]}-{PERCENTIS_LOG[1]}%: {p_ref_low:.1f} {p_ref_high:.1f}")
    return ref_stats, ref_curve_global, (p_ref_low, p_ref_high)


def _process_file(
    path: Path,
    ref_stats: Dict[Tuple[int, ...], Tuple[np.ndarray, int]],
    ref_curve_global: np.ndarray,
) -> None:
    print(f"\nProcessando {path}...")
    las = laspy.read(path)
    intensity = las.intensity.astype(np.float64)
    flightline = las.point_source_id
    classes = las.classification
    attribute_arrays = (
        _get_attribute_arrays(las, GROUP_BY_ATTRIBUTES, context=str(path)) if GROUP_BY_ATTRIBUTES else []
    )

    class_mask = _build_class_mask(classes, CLASSES_REFERENCIA)
    intensity_corr = intensity.copy()

    for patch_id in FLIGHTLINES_PATCH:
        mask_patch_base = class_mask & (flightline == patch_id)
        int_patch = intensity[mask_patch_base]

        print(f"Pontos flightline {patch_id}: {int_patch.size}")

        if int_patch.size == 0:
            print(f"Sem pontos elegiveis no flightline {patch_id}; pulando.")
            continue

        p_patch_low, p_patch_high = np.percentile(int_patch, PERCENTIS_LOG)
        print(f"Patch {PERCENTIS_LOG[0]}-{PERCENTIS_LOG[1]}%: {p_patch_low:.1f} {p_patch_high:.1f}")

        patch_curve_global = np.percentile(int_patch, PERCENTIS_CURVA)

        if patch_curve_global[-1] == patch_curve_global[0]:
            print(f"Intensidade quase constante no flightline {patch_id}; sem normalizacao.")
            continue

        patch_stats = _compute_bin_percentiles(
            intensity, attribute_arrays, mask_patch_base, PERCENTIS_CURVA, MIN_POINTS_PER_BIN
        )
        bin_transforms = _build_transforms(ref_stats, patch_stats)

        if bin_transforms:
            pontos_custom = sum(patch_stats[key][1] for key in bin_transforms if key in patch_stats)
            print(
                f"{len(bin_transforms)} bins custom para flightline {patch_id} (>= {MIN_POINTS_PER_BIN} pontos). "
                f"Pontos cobertos: {pontos_custom}"
            )
        else:
            print(f"Nenhum bin suficiente no flightline {patch_id}; usando apenas transformacao global.")

        mask_all_patch = flightline == patch_id
        _apply_patch_correction(
            intensity_corr,
            mask_all_patch,
            attribute_arrays,
            bin_transforms,
            (patch_curve_global, ref_curve_global),
        )

    intensity_corr = np.clip(intensity_corr, 0, 65535).astype(np.uint16)
    las.intensity = intensity_corr

    arquivo_saida = path.with_name(path.stem + "_intnorm.laz")
    las.write(arquivo_saida)
    print("Arquivo salvo:", arquivo_saida)


def main() -> None:
    args = _parse_args()
    cli_patch = [Path(p) for p in args.patch] if args.patch else [Path(p) for p in args.arquivos] if args.arquivos else []
    ref_only_paths = [Path(p) for p in args.refs] if args.refs else []

    # Caminho 0: se nenhumas flags de CLI forem usadas, facilitar com GUI completa (refs + fila de patches)
    if not cli_patch and not ref_only_paths:
        _run_gui_queue()
        return

    # Caminho 1: patches vindos da linha de comando (mantem comportamento anterior)
    if cli_patch:
        patch_paths = cli_patch
        if not ref_only_paths:
            ref_only_paths = _pick_reference_files_gui()
        reference_pool = list(dict.fromkeys([*patch_paths, *ref_only_paths]))

        if not reference_pool:
            raise RuntimeError("Nenhum arquivo informado.")

        for caminho in reference_pool:
            if not caminho.exists():
                raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

        print(f"{len(patch_paths)} arquivo(s) serao normalizados.")
        if ref_only_paths:
            print(f"{len(ref_only_paths)} arquivo(s) adicionais usados apenas como referencia.")

        ref_stats, ref_curve_global, _ = _collect_reference_samples(reference_pool)

        for caminho in patch_paths:
            _process_file(caminho, ref_stats, ref_curve_global)
        return

    # Caminho 2: fluxo interativo completo - referencias fixas, varias selecoes de patches
    ref_only_paths = ref_only_paths or _pick_reference_files_gui()
    if not ref_only_paths:
        raise RuntimeError("Nenhum arquivo de referencia selecionado.")

    for caminho in ref_only_paths:
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    print(f"{len(ref_only_paths)} arquivo(s) de referencia carregados (serao reutilizados).")
    ref_stats, ref_curve_global, _ = _collect_reference_samples(ref_only_paths)

    while True:
        patch_paths = _pick_patch_files_gui()
        if not patch_paths:
            print("Nenhum patch selecionado; encerrando.")
            break

        for caminho in patch_paths:
            if not caminho.exists():
                raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

        print(f"{len(patch_paths)} arquivo(s) serao normalizados com as mesmas referencias.")
        for caminho in patch_paths:
            _process_file(caminho, ref_stats, ref_curve_global)


if __name__ == "__main__":
    main()
