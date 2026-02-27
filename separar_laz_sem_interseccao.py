from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import laspy
import numpy as np


Point = Tuple[float, float]
EPS = 1e-9


@dataclass
class CloudFootprint:
    path: Path
    hull: List[Point] | None
    bbox: Tuple[float, float, float, float]


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points: Sequence[Point]) -> List[Point]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def bbox_from_points(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_intersects(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], pad: float = 0.0
) -> bool:
    return not (
        a[2] < b[0] - pad
        or b[2] < a[0] - pad
        or a[3] < b[1] - pad
        or b[3] < a[1] - pad
    )


def on_segment(a: Point, b: Point, p: Point) -> bool:
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
        and abs(orientation(a, b, p)) <= EPS
    )


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    if abs(o1) <= EPS and on_segment(a1, a2, b1):
        return True
    if abs(o2) <= EPS and on_segment(a1, a2, b2):
        return True
    if abs(o3) <= EPS and on_segment(b1, b2, a1):
        return True
    if abs(o4) <= EPS and on_segment(b1, b2, a2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def point_in_convex_polygon(p: Point, poly: Sequence[Point]) -> bool:
    if len(poly) == 0:
        return False
    if len(poly) == 1:
        return math.dist(poly[0], p) <= EPS
    if len(poly) == 2:
        return on_segment(poly[0], poly[1], p)

    sign = 0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = orientation(a, b, p)
        if abs(cross) <= EPS:
            continue
        if sign == 0:
            sign = 1 if cross > 0 else -1
        elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
            return False
    return True


def hulls_intersect(h1: Sequence[Point], h2: Sequence[Point], pad: float = 0.0) -> bool:
    b1 = bbox_from_points(h1)
    b2 = bbox_from_points(h2)
    if not bbox_intersects(b1, b2, pad=pad):
        return False

    if len(h1) == 1 and len(h2) == 1:
        return math.dist(h1[0], h2[0]) <= pad + EPS
    if len(h1) == 2 and len(h2) == 2:
        return segments_intersect(h1[0], h1[1], h2[0], h2[1])
    if len(h1) == 1 and len(h2) == 2:
        return on_segment(h2[0], h2[1], h1[0])
    if len(h1) == 2 and len(h2) == 1:
        return on_segment(h1[0], h1[1], h2[0])

    n1 = len(h1)
    n2 = len(h2)
    if n1 >= 2 and n2 >= 2:
        for i in range(n1):
            a1 = h1[i]
            a2 = h1[(i + 1) % n1]
            for j in range(n2):
                b1 = h2[j]
                b2 = h2[(j + 1) % n2]
                if segments_intersect(a1, a2, b1, b2):
                    return True

    if point_in_convex_polygon(h1[0], h2):
        return True
    if point_in_convex_polygon(h2[0], h1):
        return True
    return False


def read_laz_hull(path: Path) -> CloudFootprint:
    las = laspy.read(path)
    if len(las.x) == 0:
        raise ValueError(f"Arquivo vazio: {path}")

    pts = list(zip(np.asarray(las.x), np.asarray(las.y)))
    hull = convex_hull(pts)
    if not hull:
        hull = [pts[0]]
    bbox = bbox_from_points(hull)
    return CloudFootprint(path=path, hull=hull, bbox=bbox)


def read_laz_bbox_from_header(path: Path) -> CloudFootprint:
    with laspy.open(path) as reader:
        mins = reader.header.mins
        maxs = reader.header.maxs
    minx, miny = float(mins[0]), float(mins[1])
    maxx, maxy = float(maxs[0]), float(maxs[1])
    bbox = (minx, miny, maxx, maxy)
    return CloudFootprint(path=path, hull=None, bbox=bbox)


def build_intersection_graph(clouds: Sequence[CloudFootprint], pad: float, bbox_only: bool) -> List[int]:
    n = len(clouds)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not bbox_intersects(clouds[i].bbox, clouds[j].bbox, pad=pad):
                continue
            if bbox_only:
                intersects = True
            else:
                intersects = hulls_intersect(clouds[i].hull or [], clouds[j].hull or [], pad=pad)
            if intersects:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def greedy_dsatur_coloring(adj: Sequence[int]) -> List[int]:
    n = len(adj)
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    while uncolored:
        v = max(uncolored, key=lambda x: (len(neighbor_colors[x]), (adj[x]).bit_count()))
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)
        bits = adj[v]
        u = 0
        while bits:
            if bits & 1 and u in uncolored:
                neighbor_colors[u].add(c)
            bits >>= 1
            u += 1
    return colors


def exact_coloring(adj: Sequence[int]) -> List[int]:
    n = len(adj)
    best = greedy_dsatur_coloring(adj)
    best_k = max(best) + 1 if best else 0

    colors = [-1] * n
    sat = [set() for _ in range(n)]
    degree = [a.bit_count() for a in adj]
    uncolored = set(range(n))

    def choose_vertex() -> int:
        return max(uncolored, key=lambda v: (len(sat[v]), degree[v]))

    def paint(v: int, c: int) -> List[int]:
        changed = []
        bits = adj[v]
        u = 0
        while bits:
            if bits & 1 and u in uncolored and c not in sat[u]:
                sat[u].add(c)
                changed.append(u)
            bits >>= 1
            u += 1
        return changed

    def unpaint(c: int, changed: Sequence[int]) -> None:
        for u in changed:
            sat[u].remove(c)

    def dfs(used_colors: int) -> None:
        nonlocal best, best_k
        if not uncolored:
            if used_colors < best_k:
                best = colors.copy()
                best_k = used_colors
            return

        if used_colors >= best_k:
            return

        v = choose_vertex()
        forbidden = set()
        bits = adj[v]
        u = 0
        while bits:
            if bits & 1 and colors[u] != -1:
                forbidden.add(colors[u])
            bits >>= 1
            u += 1

        uncolored.remove(v)
        for c in range(used_colors):
            if c in forbidden:
                continue
            colors[v] = c
            changed = paint(v, c)
            dfs(used_colors)
            unpaint(c, changed)
            colors[v] = -1

        if used_colors + 1 < best_k:
            c = used_colors
            colors[v] = c
            changed = paint(v, c)
            dfs(used_colors + 1)
            unpaint(c, changed)
            colors[v] = -1

        uncolored.add(v)

    dfs(0)
    return best


def solve_coloring(adj: Sequence[int], exact_max_n: int) -> List[int]:
    n = len(adj)
    if n <= exact_max_n:
        return exact_coloring(adj)
    return greedy_dsatur_coloring(adj)


def group_by_colors(clouds: Sequence[CloudFootprint], colors: Sequence[int]) -> List[List[CloudFootprint]]:
    k = max(colors) + 1 if colors else 0
    groups: List[List[CloudFootprint]] = [[] for _ in range(k)]
    for cloud, c in sorted(zip(clouds, colors), key=lambda t: (t[1], t[0].path.name.lower())):
        groups[c].append(cloud)
    return groups


def copy_or_move_groups(groups: Sequence[Sequence[CloudFootprint]], output_dir: Path, move: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, group in enumerate(groups, start=1):
        folder = output_dir / f"grupo_{idx:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        for cloud in group:
            dst = folder / cloud.path.name
            if move:
                shutil.move(str(cloud.path), str(dst))
            else:
                shutil.copy2(str(cloud.path), str(dst))


def find_laz_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.laz" if recursive else "*.laz"
    return sorted(input_dir.glob(pattern))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Separa arquivos .laz no menor numero de pastas possivel, "
            "garantindo que dentro de cada pasta nao haja interseccao "
            "entre os contornos convexos (XY) das nuvens."
        )
    )
    p.add_argument("input_dir", type=Path, help="Pasta com arquivos .laz")
    p.add_argument("output_dir", type=Path, help="Pasta de saida para grupos")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Busca .laz recursivamente dentro de subpastas de input_dir",
    )
    p.add_argument(
        "--move",
        dest="move",
        action="store_true",
        help="Move os arquivos para as pastas de saida (padrao)",
    )
    p.add_argument(
        "--copy",
        dest="move",
        action="store_false",
        help="Copia os arquivos para as pastas de saida",
    )
    p.set_defaults(move=True)
    p.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Margem em unidades XY para considerar interseccao (padrao: 0.0)",
    )
    p.add_argument(
        "--footprint-mode",
        choices=["convex-hull", "bbox-header"],
        default="convex-hull",
        help=(
            "Modo de contorno: 'convex-hull' (mais fiel, mais lento) ou "
            "'bbox-header' (muito mais rapido, conservador: area igual/maior)"
        ),
    )
    p.add_argument(
        "--exact-max-n",
        type=int,
        default=60,
        help=(
            "Numero maximo de arquivos para resolver minimo exato de pastas "
            "(acima disso usa heuristica DSATUR)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Pasta de entrada nao existe: {args.input_dir}")

    files = find_laz_files(args.input_dir, args.recursive)
    if not files:
        raise FileNotFoundError("Nenhum arquivo .laz encontrado.")

    bbox_only = args.footprint_mode == "bbox-header"
    if bbox_only:
        clouds = [read_laz_bbox_from_header(path) for path in files]
    else:
        clouds = [read_laz_hull(path) for path in files]

    adj = build_intersection_graph(clouds, pad=args.padding, bbox_only=bbox_only)
    colors = solve_coloring(adj, exact_max_n=args.exact_max_n)
    groups = group_by_colors(clouds, colors)

    copy_or_move_groups(groups, args.output_dir, move=args.move)

    k = len(groups)
    print(f"Arquivos lidos: {len(files)}")
    print(f"Pastas geradas: {k}")
    print(f"Saida: {args.output_dir}")
    print(f"Operacao nos arquivos: {'mover' if args.move else 'copiar'}")
    print(f"Modo de contorno: {args.footprint_mode}")
    if len(files) > args.exact_max_n:
        print("Metodo de agrupamento: heuristico (DSATUR)")
    else:
        print("Metodo de agrupamento: minimo exato")


if __name__ == "__main__":
    main()
