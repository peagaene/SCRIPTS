#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calcula convergencia meridiana e declinacao magnetica a partir de UTM.

Uso:
  python convergencia_decl_magnetica.py

Edite os DEFAULTS abaixo para ajustar variaveis padrao.
"""

from __future__ import annotations

from datetime import date
from math import atan, degrees, radians, sin, tan
from typing import Optional, Tuple
import json
import urllib.parse
import urllib.request

try:
    from pyproj import Transformer  # type: ignore
except Exception:
    Transformer = None

# ---------------------------------------------------------------------------
# CONFIGURACOES PADRAO (EDITE SE QUISER)
# ---------------------------------------------------------------------------

DEFAULT_ZONE = 23
DEFAULT_HEMISPHERE = "S"
NOAA_MODEL = "WMM"
NOAA_API_KEY = "zNEw7"


# ---------------------------------------------------------------------------
# UTILITARIOS
# ---------------------------------------------------------------------------


def _utm_central_meridian(zone: int) -> float:
    if zone < 1 or zone > 60:
        raise ValueError("Zona UTM deve estar entre 1 e 60.")
    return -183.0 + 6.0 * zone


def _sirgas_epsg_from_zone(zone: int, hemisphere: str) -> int:
    # SIRGAS 2000 / UTM (Sul) cobre zonas 18S-25S (Brasil).
    hemi = hemisphere.upper()
    if hemi != "S":
        raise ValueError("SIRGAS 2000 UTM disponivel aqui apenas para hemisferio Sul.")
    if zone < 18 or zone > 25:
        raise ValueError("Para SIRGAS 2000 UTM (Brasil), use zona 18 a 25.")
    return 31960 + zone


def _format_deg(value: Optional[float]) -> str:
    if value is None:
        return "N/D"
    return f"{value:.3f}°"


def _format_latlon(value: float) -> str:
    return f"{value:.6f}°".replace(".", ",")


def _format_dms(value: Optional[float]) -> str:
    if value is None:
        return "N/D"
    sign = "-" if value < 0 else ""
    v = abs(value)
    deg = int(v)
    minutes_full = (v - deg) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    return f"{sign}{deg}°{minutes:02d}'{seconds:05.2f}\""


def _format_declination(value: Optional[float], uncertainty: Optional[float]) -> str:
    if value is None:
        return "N/D"
    if uncertainty is not None:
        return f"{value:.2f}° \u00b1 {uncertainty:.2f}°"
    return f"{value:.2f}°"


# ---------------------------------------------------------------------------
# CALCULOS
# ---------------------------------------------------------------------------


def _compute_lat_lon_from_utm(easting: float, northing: float, zone: int, hemisphere: str) -> Tuple[float, float]:
    if Transformer is None:
        raise RuntimeError("pyproj nao esta instalado.")
    epsg = _sirgas_epsg_from_zone(zone, hemisphere)
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4674", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def _compute_convergence(lat_deg: float, lon_deg: float, lon0_deg: float) -> float:
    lat_rad = radians(lat_deg)
    dlon_rad = radians(lon_deg - lon0_deg)
    conv_rad = atan(tan(dlon_rad) * sin(lat_rad))
    return degrees(conv_rad)


def _fetch_noaa_declination(
    lat_deg: float, lon_deg: float, ref_date: date
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    base_url = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination"
    params = {
        "lat1": f"{lat_deg:.6f}",
        "lon1": f"{lon_deg:.6f}",
        "model": NOAA_MODEL,
        "startYear": ref_date.year,
        "startMonth": ref_date.month,
        "startDay": ref_date.day,
        "resultFormat": "json",
        "key": NOAA_API_KEY,
    }
    url = base_url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        result = (data.get("result") or [None])[0]
        if not result:
            return None, None, "Resposta vazia do NOAA."
        decl = result.get("declination")
        uncertainty = result.get("declination_uncertainty")
        sv = result.get("declination_sv")
        return (
            float(decl) if decl is not None else None,
            float(uncertainty) if uncertainty is not None else None,
            float(sv) if sv is not None else None,
            None,
        )
    except Exception as exc:
        return None, None, None, f"Falha ao consultar NOAA: {exc}"


# ---------------------------------------------------------------------------
# UI (Tkinter)
# ---------------------------------------------------------------------------


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", "."))
    except Exception:
        return None


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except Exception:
        return None


def _build_ui() -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Convergencia e Declinacao")
    root.geometry("520x360")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill=tk.BOTH, expand=True)

    title = ttk.Label(frm, text="Convergencia Meridiana e Declinacao Magnetica")
    title.pack(anchor="w")

    fields = ttk.Frame(frm, padding=(0, 10, 0, 0))
    fields.pack(fill=tk.X)

    def add_field(label: str, default: str = "") -> tk.StringVar:
        row = ttk.Frame(fields)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    e_var = add_field("E (UTM)", "")
    n_var = add_field("N (UTM)", "")
    zone_var = add_field("Zona UTM", str(DEFAULT_ZONE))
    hemi_var = add_field("Hemisferio (S/N)", DEFAULT_HEMISPHERE)
    ttk.Label(fields, text="Datum fixo: SIRGAS 2000").pack(anchor="w", pady=(6, 0))

    results = tk.Text(frm, height=8, wrap="word")
    results.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def set_result(text: str) -> None:
        results.delete("1.0", tk.END)
        results.insert("1.0", text)

    def compute() -> None:
        easting = _safe_float(e_var.get() or "")
        northing = _safe_float(n_var.get() or "")
        if easting is None or northing is None:
            set_result("E e N sao obrigatorios (numericos).")
            return

        zone = _safe_int(zone_var.get() or "")
        hemisphere = (hemi_var.get() or "").strip().upper()
        if zone is None:
            set_result("Zona UTM obrigatoria.")
            return
        if hemisphere not in {"S", "N"}:
            set_result("Hemisferio deve ser S ou N.")
            return
        ref_date = date.today()

        try:
            lat, lon = _compute_lat_lon_from_utm(easting, northing, zone, hemisphere)
        except Exception as exc:
            set_result(f"Falha ao converter UTM para lat/lon: {exc}")
            return

        lon0 = _utm_central_meridian(zone)
        conv = _compute_convergence(lat, lon, lon0)
        decl, decl_unc, decl_sv, decl_err = _fetch_noaa_declination(lat, lon, ref_date)

        out = []
        out.append(f"Latitude:  {_format_latlon(lat)}")
        out.append(f"Longitude: {_format_latlon(lon)}")
        out.append(f"Convergencia meridiana: {_format_dms(conv)}")
        out.append(f"Declinacao magnetica:   {_format_declination(decl, decl_unc)}")
        if decl_sv is not None:
            out.append(f"Mudanca por ano:        {decl_sv:.2f}°/ano")
        out.append(f"Modelo NOAA: {NOAA_MODEL}-2025")
        if decl_err:
            out.append(decl_err)
        set_result("\n".join(out))

    actions = ttk.Frame(frm, padding=(0, 8, 0, 0))
    actions.pack(fill=tk.X)
    ttk.Button(actions, text="Calcular", command=compute).pack(side=tk.LEFT)

    root.mainloop()


def main() -> None:
    _build_ui()


if __name__ == "__main__":
    main()
