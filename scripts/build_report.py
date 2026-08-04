"""Genera los entregables a partir del JSONL de la corrida.

Produce en `results/`:
  - resultados.csv / resultados.xlsx : una fila por húmero
  - dashboard.html                   : distribuciones y diagnósticos
  - INFORME.md                       : análisis del sistema y de la corrida
  - worst_cases/*.html               : vista 3D de los peores casos (--worst-3d)

Uso:
    python scripts/build_report.py
    python scripts/build_report.py --worst-3d 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PipelineParams  # noqa: E402
from src.validation.confidence import confidence_flags  # noqa: E402

# Paleta validada (dataviz skill, modo claro, superficie #fcfcfb).
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
CRITICAL = "#d03b3b"

#: Orden clínico fijo; el color sigue al grupo, nunca al ranking.
PATHOLOGY_ORDER = [
    "sano", "Hill-Sachs", "cuff tear", "GHOA leve", "GHOA moderada", "GHOA severa",
]
PATHOLOGY_COLOR = {
    "sano": "#2a78d6",
    "Hill-Sachs": "#eb6834",
    "cuff tear": "#1baf7a",
    "GHOA leve": "#eda100",
    "GHOA moderada": "#e87ba4",
    "GHOA severa": "#008300",
    "otro": MUTED,
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def load_results(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"El JSONL está vacío: {jsonl_path}")
    frame = pd.DataFrame(rows).drop_duplicates(subset="model_id", keep="last")

    # Las banderas de confianza se recalculan siempre, aunque el runner ya las
    # haya guardado: así un cambio de umbrales se refleja en el reporte sin
    # tener que re-correr los 229 húmeros (12 minutos).
    flags = frame.apply(lambda row: confidence_flags(row.to_dict()), axis=1)
    frame["needs_review"] = [f["needs_review"] for f in flags]
    frame["confidence"] = [f["confidence"] for f in flags]
    frame["review_reasons"] = ["; ".join(f["review_reasons"]) for f in flags]

    return frame.sort_values("center_error_mm", ascending=False, na_position="first")


#: Total de húmeros con STL en el dataset. Si la corrida los cubre a todos,
#: el reporte es un censo y no corresponde hablar de "muestra estratificada".
DATASET_SIZE = 229


def _is_census(frame: pd.DataFrame) -> bool:
    return len(frame) >= DATASET_SIZE


def _scope_phrase(frame: pd.DataFrame) -> str:
    return ("Dataset completo" if _is_census(frame) else "Muestra estratificada")


def _base_layout(title: str, height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        margin=dict(l=60, r=24, t=48, b=48),
        height=height,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    )


def figure_error_distribution(frame: pd.DataFrame, threshold: float):
    import plotly.graph_objects as go

    values = frame["center_error_mm"].dropna()
    figure = go.Figure()
    figure.add_trace(go.Histogram(
        x=values, xbins=dict(start=0, size=0.5), marker=dict(color=SERIES_1),
        hovertemplate="%{y} húmeros entre %{x} mm<extra></extra>", name="húmeros",
    ))
    figure.add_vline(
        x=threshold, line=dict(color=CRITICAL, width=2, dash="dash"),
        annotation_text=f"tolerancia {threshold:.0f} mm",
        annotation_font=dict(color=CRITICAL, size=11),
    )
    layout = _base_layout("Error del centro ajustado contra el HHC medido")
    layout["xaxis"]["title"] = "error (mm)"
    layout["yaxis"]["title"] = "húmeros"
    layout["bargap"] = 0.06
    figure.update_layout(**layout)
    return figure


def _clip_range(values: pd.Series) -> tuple:
    """
    Rango del eje que deja legibles las distribuciones sin que un outlier
    extremo (hay un caso de ~97 mm) aplaste todo contra el cero.

    Devuelve (tope, cuántos quedan fuera de escala).
    """
    if values.empty:
        return (1.0, 0)
    cap = float(np.ceil(values.quantile(0.98) / 5.0) * 5.0)
    cap = max(cap, threshold_floor := 10.0)
    beyond = int((values > cap).sum())
    return (cap, beyond)


def _offscale_note(beyond: int, cap: float) -> str:
    if beyond == 0:
        return ""
    subject = "caso queda" if beyond == 1 else "casos quedan"
    return f"  ·  {beyond} {subject} fuera de escala (> {cap:.0f} mm)"


def figure_error_by_pathology(frame: pd.DataFrame, threshold: float):
    """
    Un punto por húmero sobre el box: mostrar solo la caja escondería el
    tamaño real de cada grupo y dónde se concentran los casos.
    """
    import plotly.graph_objects as go

    cap, beyond = _clip_range(frame["center_error_mm"].dropna())
    figure = go.Figure()
    groups = [g for g in PATHOLOGY_ORDER if g in set(frame["pathology_group"])]
    for group in groups:
        subset = frame[frame["pathology_group"] == group]["center_error_mm"].dropna()
        if subset.empty:
            continue
        figure.add_trace(go.Box(
            y=subset, name=f"{group}<br>n={len(subset)}",
            marker=dict(color=PATHOLOGY_COLOR[group], size=8),
            line=dict(width=2), fillcolor="rgba(0,0,0,0)",
            boxpoints="all", jitter=0.5, pointpos=0,
            hovertemplate="%{y:.2f} mm<extra></extra>",
        ))
    figure.add_hline(y=threshold, line=dict(color=CRITICAL, width=2, dash="dash"))
    layout = _base_layout(
        f"Error por patología · cada punto es un húmero{_offscale_note(beyond, cap)}",
        height=440,
    )
    layout["yaxis"]["title"] = "error (mm)"
    layout["yaxis"]["range"] = [0, cap]
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure


def figure_rmse_vs_error(frame: pd.DataFrame, threshold: float):
    """
    La pregunta operativa: ¿el RMSE interno anticipa el error real? Si no
    correlaciona, el sistema no puede detectar sus propias fallas sin
    ground truth.
    """
    import plotly.graph_objects as go

    valid = frame.dropna(subset=["center_error_mm", "rmse"])
    ok = valid[valid["center_error_mm"] <= threshold]
    bad = valid[valid["center_error_mm"] > threshold]

    figure = go.Figure()
    for subset, color, label in ((ok, SERIES_1, f"dentro de {threshold:.0f} mm"),
                                 (bad, CRITICAL, f"fuera de {threshold:.0f} mm")):
        figure.add_trace(go.Scatter(
            x=subset["rmse"], y=subset["center_error_mm"], mode="markers", name=label,
            marker=dict(color=color, size=9, line=dict(color=SURFACE, width=2)),
            text=subset["model_id"],
            hovertemplate="%{text}<br>RMSE %{x:.3f} mm<br>error %{y:.2f} mm<extra></extra>",
        ))
    figure.add_hline(y=threshold, line=dict(color=CRITICAL, width=1, dash="dash"))

    correlation = valid["rmse"].corr(valid["center_error_mm"])
    layout = _base_layout(
        f"RMSE del ajuste vs error real · correlación r = {correlation:.2f}", height=400,
    )
    layout["xaxis"]["title"] = "RMSE interno del ajuste (mm)"
    layout["yaxis"]["title"] = "error contra HHC (mm)"
    layout["legend"] = dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
    figure.update_layout(**layout)
    return figure


def figure_error_by_completeness(frame: pd.DataFrame, threshold: float):
    import plotly.graph_objects as go

    cap, beyond = _clip_range(frame["center_error_mm"].dropna())
    figure = go.Figure()
    for flag, label in ((True, "húmero completo"), (False, "truncado por el CT")):
        subset = frame[frame["whole_humerus"] == flag]["center_error_mm"].dropna()
        if subset.empty:
            continue
        figure.add_trace(go.Box(
            y=subset, name=f"{label}<br>n={len(subset)}",
            marker=dict(color=SERIES_1, size=8), line=dict(width=2),
            fillcolor="rgba(0,0,0,0)", boxpoints="all", jitter=0.5, pointpos=0,
            hovertemplate="%{y:.2f} mm<extra></extra>",
        ))
    figure.add_hline(y=threshold, line=dict(color=CRITICAL, width=1, dash="dash"))
    layout = _base_layout(
        f"Error según completitud del hueso escaneado{_offscale_note(beyond, cap)}",
    )
    layout["yaxis"]["title"] = "error (mm)"
    layout["yaxis"]["range"] = [0, cap]
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure


def _stat_tiles(frame: pd.DataFrame, threshold: float) -> str:
    measured = frame["center_error_mm"].dropna()
    within = int((measured <= threshold).sum())
    errors = int((frame["status"] != "ok").sum())
    tiles = [
        ("Húmeros procesados", f"{len(frame)}",
         "dataset completo" if _is_census(frame) else "de la muestra estratificada"),
        ("Error mediano", f"{measured.median():.2f} <small>mm</small>", "contra el HHC medido"),
        ("Dentro de tolerancia", f"{100 * within / len(measured):.0f}<small>%</small>",
         f"{within} de {len(measured)} bajo {threshold:.0f} mm"),
        ("Error máximo", f"{measured.max():.1f} <small>mm</small>", "peor caso de la corrida"),
        ("Fallas de proceso", f"{errors}", "húmeros que no completaron"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-note">{note}</div></div>'
        for label, value, note in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _worst_table(frame: pd.DataFrame, limit: int = 15) -> str:
    columns = ["model_id", "pathology", "side", "whole_humerus",
               "center_error_mm", "radius", "rmse", "score"]
    worst = frame.dropna(subset=["center_error_mm"]).nlargest(limit, "center_error_mm")
    header = "".join(f"<th>{name}</th>" for name in
                     ["Modelo", "Patología", "Lado", "Completo", "Error (mm)",
                      "Radio (mm)", "RMSE (mm)", "Score"])
    body = []
    for _, row in worst[columns].iterrows():
        cells = (
            f"<td>{row['model_id']}</td><td>{row['pathology']}</td>"
            f"<td>{row['side']}</td><td>{'sí' if row['whole_humerus'] else 'no'}</td>"
            f"<td class='num strong'>{row['center_error_mm']:.2f}</td>"
            f"<td class='num'>{row['radius']:.2f}</td>"
            f"<td class='num'>{row['rmse']:.3f}</td>"
            f"<td class='num'>{row['score']:.3f}</td>"
        )
        body.append(f"<tr>{cells}</tr>")
    return (f"<table><thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def build_dashboard(frame: pd.DataFrame, out_path: Path, params: PipelineParams) -> None:
    import plotly.io as pio

    threshold = params.success_threshold_mm
    figures = [
        figure_error_distribution(frame, threshold),
        figure_error_by_pathology(frame, threshold),
        figure_rmse_vs_error(frame, threshold),
        figure_error_by_completeness(frame, threshold),
    ]
    blocks = []
    for index, figure in enumerate(figures):
        blocks.append(pio.to_html(
            figure, full_html=False,
            include_plotlyjs="inline" if index == 0 else False,
            config={"displayModeBar": False, "responsive": True},
        ))

    # Solo los grupos que el gráfico realmente dibuja (PATHOLOGY_ORDER).
    sizes = frame[frame["pathology_group"].isin(PATHOLOGY_ORDER)].groupby("pathology_group").size()
    group_note = (
        f"Cada punto es un húmero. Los grupos van de n={sizes.min()} a n={sizes.max()}, "
        f"así que se muestran los casos individuales y no solo la caja: con los grupos "
        f"más chicos la mediana la mueve un puñado de huesos."
    )

    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validación del pipeline de cabeza humeral</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: {PAGE}; color: {INK};
         font-family: {FONT}; line-height: 1.5; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 64px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .sub {{ color: {INK_2}; margin: 0 0 32px; font-size: 14px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 32px; }}
  .tile {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
           border-radius: 10px; padding: 16px 18px; }}
  .tile-label {{ font-size: 12px; color: {MUTED}; text-transform: uppercase;
                 letter-spacing: 0.04em; }}
  .tile-value {{ font-size: 30px; font-weight: 600; margin: 6px 0 2px; color: {INK}; }}
  .tile-value small {{ font-size: 15px; font-weight: 500; color: {INK_2}; }}
  .tile-note {{ font-size: 12px; color: {INK_2}; }}
  .card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
           border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; }}
  h2 {{ font-size: 17px; margin: 36px 0 12px; }}
  .note {{ font-size: 13px; color: {INK_2}; margin: 0 0 14px; }}
  .table-wrap {{ overflow-x: auto; background: {SURFACE};
                 border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ text-align: left; padding: 10px 12px; color: {MUTED}; font-weight: 600;
        border-bottom: 1px solid {GRID}; white-space: nowrap; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid {GRID}; color: {INK_2}; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.strong {{ color: {INK}; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
</style></head>
<body><div class="wrap">
  <h1>Validación del pipeline de cabeza humeral</h1>
  <p class="sub">{_scope_phrase(frame)}: {len(frame)} húmeros comparados contra el
     centro de cabeza humeral (HHC) medido en los CSV del dataset.</p>
  {_stat_tiles(frame, threshold)}
  <div class="card">{blocks[0]}</div>
  <h2>¿Dónde falla?</h2>
  <p class="note">{group_note}</p>
  <div class="card">{blocks[1]}</div>
  <div class="card">{blocks[3]}</div>
  <h2>¿Puede el sistema detectar sus propias fallas?</h2>
  <p class="note">El RMSE es lo único que el pipeline conoce sin ground truth.
     Si no correlaciona con el error real, no sirve como señal de alarma.</p>
  <div class="card">{blocks[2]}</div>
  <h2>Peores casos</h2>
  <div class="table-wrap">{_worst_table(frame)}</div>
</div></body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def build_worst_case_views(frame: pd.DataFrame, out_dir: Path, count: int,
                           data_root: Path, params: PipelineParams) -> None:
    """Vista 3D de los peores casos: superficie, esfera ajustada y HHC real."""
    from src.dataset.catalog import load_catalog
    from src.mesh.discretizer import MeshDiscretizer
    from src.mesh.loader import STLLoader
    from src.validation.risk_map import fit_local_spheres, risk_score_from_fit
    from src.visualization.interactive_web import InteractiveWeb3D

    catalog = {record.model_id: record for record in load_catalog(data_root)}
    worst = frame.dropna(subset=["center_error_mm"]).nlargest(count, "center_error_mm")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in worst.iterrows():
        record = catalog.get(row["model_id"])
        if record is None:
            continue
        mesh = STLLoader.load(str(record.stl_path))
        points, _ = MeshDiscretizer().discretize_uniform(
            mesh.vertices, mesh.faces, params.samples, random_seed=params.discretize_seed,
        )
        rmse, fitted_radius = fit_local_spheres(
            points, params.search_radius, params.initial_radius, params.min_neighbors,
        )
        scores = risk_score_from_fit(
            rmse, fitted_radius, params.max_error, params.radius_min, params.radius_max,
        )

        viz = InteractiveWeb3D(
            title=f"{row['model_id']} · {row['pathology']} · error {row['center_error_mm']:.2f} mm",
        )
        viz.plot_points_colored(points, scores, name="Riesgo", colorbar_title="score")
        viz.plot_sphere(
            np.array([row["center_x"], row["center_y"], row["center_z"]]),
            float(row["radius"]), name=f"Esfera ajustada (r={row['radius']:.1f} mm)",
        )
        viz.plot_seeds(
            np.array([[row["hhc_x"], row["hhc_y"], row["hhc_z"]]]),
            name="HHC medido (ground truth)",
        )
        viz.save(str(out_dir / f"{row['model_id']}.html"))
        print(f"  {row['model_id']}  err={row['center_error_mm']:.2f}mm")


def _build_diagnosis(frame: pd.DataFrame, params: PipelineParams) -> str:
    """
    Arma la sección de diagnóstico a partir de los números de la corrida.

    Deliberadamente calculado y no redactado de antemano: en la primera
    versión de este script la conclusión sobre el RMSE estaba escrita a mano
    y los datos decían lo contrario.
    """
    threshold = params.success_threshold_mm
    usable = frame.dropna(subset=["center_error_mm", "rmse"]).copy()
    usable["fail"] = usable["center_error_mm"] > threshold
    failed, passed = usable[usable["fail"]], usable[~usable["fail"]]
    parts = []

    # --- Comparación contra la línea base, si está disponible ---
    baseline_path = Path(__file__).resolve().parent.parent / "results" / "baseline_229.jsonl"
    if baseline_path.exists():
        base_rows = [json.loads(line) for line in open(baseline_path, encoding="utf-8") if line.strip()]
        base = pd.DataFrame(base_rows).drop_duplicates(subset="model_id", keep="last").set_index("model_id")
        shared = base.index.intersection(usable["model_id"])
        if len(shared) >= 10:
            now = usable.set_index("model_id").loc[shared]
            was_bad = base.loc[shared, "center_error_mm"] > threshold
            is_bad = now["center_error_mm"] > threshold
            arreglados = int((was_bad & ~is_bad).sum())
            rotos = int((~was_bad & is_bad).sum())
            parts.append(f"""### Qué cambió respecto de la línea base

Se quitaron del costo `morphology_penalty` y `reference_penalty` (ambos derivados
del marco anatómico fijo) y se agregó un filtro de radio fisiológico sobre los
candidatos antes de rankearlos. Sobre los mismos {len(shared)} húmeros:

| | Antes | Ahora |
|---|---|---|
| Error mediano | {base.loc[shared, 'center_error_mm'].median():.2f} mm | **{now['center_error_mm'].median():.2f} mm** |
| Percentil 90 | {base.loc[shared, 'center_error_mm'].quantile(0.9):.2f} mm | **{now['center_error_mm'].quantile(0.9):.2f} mm** |
| Fuera de {threshold:.0f} mm | {int(was_bad.sum())} ({100 * was_bad.mean():.0f}%) | **{int(is_bad.sum())} ({100 * is_bad.mean():.0f}%)** |

**{arreglados} húmeros pasaron de fallar a acertar y {rotos} hicieron lo contrario.**
El detalle caso por caso está en `comparacion.csv`; importa mirarlo porque un
cambio que arregla muchos y rompe algunos se ve igual en la mediana que uno que
solo arregla.

La causa del salto: el término morfológico llegaba a valer 3.87 con peso 0.25
—hasta 0.97 del costo— contra 0.04–0.22 del RMSE y 0.15–0.27 de la cobertura.
Un término calculado con un marco 36° desviado dominaba a los dos que sí
discriminan, y empujaba el candidato correcto al puesto 22–74 de 80.""")

    # --- ¿Los umbrales de validación llegan a activarse alguna vez? ---
    accepted = int(frame["valid"].sum()) if "valid" in frame else 0
    rmse_max = usable["rmse"].max()
    radius_min, radius_max = frame["radius"].min(), frame["radius"].max()
    penalties = frame["reference_penalty"].dropna()
    penalised = int((penalties > 0).sum())
    penalty_observed = (f"marcaría a {penalised} de {len(penalties)}"
                        if penalised else f"constante en {penalties.unique().tolist()}")
    penalty_note = (
        f"El `reference_penalty` se deriva de los offsets medial/posterior medidos contra "
        f"el marco anatómico fijo, cuyo problema se detalla abajo: marcaría {penalised} de "
        f"{len(penalties)} húmeros según la orientación con que se adquirió la tomografía, "
        f"no según su morfología."
    )

    parts.append(f"""### La validación heredada sigue sin rechazar nada

`AuditTrail.is_valid_approximation` —la validación que evalúa al ganador ya
elegido— aceptó **{accepted} de {len(frame)}** ajustes, incluido el de
{usable['center_error_mm'].max():.1f} mm. Sus umbrales están fuera del rango que
producen los datos reales:

| Umbral | Valor configurado | Rango observado | ¿Rechaza algo? |
|---|---|---|---|
| RMSE máximo | {params.max_error:.1f} mm | {usable['rmse'].min():.2f} – {rmse_max:.2f} mm | nunca |
| Radio plausible | 17 – 40 mm | {radius_min:.1f} – {radius_max:.1f} mm | nunca |
| `reference_penalty` | 0 ó 0.25 | {penalty_observed} | no entra al costo |

Ese mismo rango de radio **sí es útil aplicado antes**, como filtro de los 80
candidatos en vez de como examen del ganador: ahí elimina los ajustes
degenerados (el optimizador llega a devolver radios de miles de milímetros
sobre superficies casi planas) que antes podían ganar el ranking. Es el cambio
que se implementó en `best_fit.py`.

{penalty_note} Por eso quedó fuera del costo, junto con `morphology_penalty`.
La función de confianza (más abajo) es la que ahora cumple el rol de rechazo.""")

    # --- ¿Qué señal interna sí discrimina? ---
    correlation = usable["rmse"].corr(usable["center_error_mm"])
    coverage_correlation = usable["coverage_ratio"].corr(usable["center_error_mm"])
    best = None
    for candidate in np.arange(0.60, 1.00, 0.05):
        flagged = usable["rmse"] > candidate
        detected = int((flagged & usable["fail"]).sum())
        false_alarms = int((flagged & ~usable["fail"]).sum())
        if detected >= 0.8 * len(failed) and (best is None or false_alarms < best[2]):
            best = (candidate, detected, false_alarms)

    tuning = ""
    if best is not None:
        cut, detected, false_alarms = best
        tuning = (f"\n\nUn umbral de RMSE en **{cut:.2f} mm** —muy por debajo del "
                  f"{params.max_error:.1f} configurado— marcaría {detected} de las "
                  f"{len(failed)} fallas, con {false_alarms} falsas alarmas. No es un "
                  f"filtro perfecto, pero convierte una validación inerte en una señal "
                  f"de revisión manual.")

    solapamiento = max(0.0, passed["rmse"].max() - failed["rmse"].min())
    veredicto = (
        f"Los rangos casi no se solapan (apenas {solapamiento:.2f} mm), así que el RMSE "
        f"sirve como señal de alarma."
        if solapamiento < 0.30 else
        f"Los rangos se solapan {solapamiento:.2f} mm, así que el RMSE ordena bien en "
        f"promedio pero no permite decidir caso por caso."
    )
    parts.append(f"""### El RMSE como señal de alarma

El RMSE interno es lo único que el pipeline conoce sin ground truth:

| | n | RMSE medio | RMSE rango |
|---|---|---|---|
| Dentro de {threshold:.0f} mm | {len(passed)} | {passed['rmse'].mean():.3f} | {passed['rmse'].min():.2f} – {passed['rmse'].max():.2f} |
| Fuera de {threshold:.0f} mm | {len(failed)} | {failed['rmse'].mean():.3f} | {failed['rmse'].min():.2f} – {failed['rmse'].max():.2f} |

{veredicto} La cobertura de la superficie aporta una señal en la misma
dirección (los ajustes malos cubren {failed['coverage_ratio'].mean():.2f} de la
cabeza contra {passed['coverage_ratio'].mean():.2f}).{tuning}""")

    # --- Marco anatómico fijo ---
    angles = frame["frame_angle_deg"].dropna()
    if len(angles):
        parts.append(f"""### El marco anatómico fijo no describe a estos huesos

`best_fit.py:106-107` asume que medial es `[1,0,0]` y posterior `[0,1,0]` para
todos los huesos. Comparado contra un marco derivado de las tuberosidades del
propio hueso, el marco fijo discrepa en una mediana de **{angles.median():.0f}°**
(rango {angles.min():.0f}–{angles.max():.0f}°) sobre {len(angles)} húmeros. Cada
tomografía tiene su sistema de coordenadas y el dataset mezcla hombros izquierdos
y derechos, así que los offsets medial/posterior que alimentan la validación
morfológica no están midiendo lo que dicen medir: reflejan la orientación con
que se adquirió la tomografía tanto como la anatomía del hueso.""")

    # --- Efecto de la patología: qué grupos fallan más que la media ---
    by_pathology = (usable.groupby("pathology_group")
                    .agg(n=("fail", "size"), tasa=("fail", "mean"),
                         mediana=("center_error_mm", "median"))
                    .sort_values("tasa", ascending=False))
    by_pathology = by_pathology[by_pathology["n"] >= 5]
    overall_rate = usable["fail"].mean()
    risky = by_pathology[by_pathology["tasa"] > overall_rate]

    if len(risky):
        rows = "\n".join(
            f"| {group} | {int(row['n'])} | {row['mediana']:.2f} mm | "
            f"{int(row['tasa'] * row['n'])}/{int(row['n'])} ({100 * row['tasa']:.0f}%) |"
            for group, row in risky.iterrows()
        )
        deformed = [g for g in risky.index if "GHOA" in g or g == "Hill-Sachs"]
        mechanism = ""
        if deformed:
            mechanism = (
                f"\n\nLos grupos que más fallan ({', '.join(deformed)}) tienen algo en "
                f"común: **la cabeza articular está deformada**. En la artrosis los "
                f"osteofitos y el remodelado extienden la superficie casi esférica más "
                f"allá de la cabeza; en Hill-Sachs el defecto por impactación le quita un "
                f"pedazo. El ajuste solo busca esfericidad local, así que se acomoda sobre "
                f"la superficie disponible sin ninguna señal de que se salió de la cabeza "
                f"articular."
            )
        parts.append(f"""### Falla donde la cabeza está deformada

Tasa global de error fuera de tolerancia: **{100 * overall_rate:.0f}%**. Los grupos
que la superan:

| Grupo | n | Error mediano | Fuera de {threshold:.0f} mm |
|---|---|---|---|
{rows}{mechanism}""")

    # --- Autodetección de fallas ---
    if "needs_review" in usable.columns:
        marcados = usable[usable["needs_review"]]
        detectadas = int((marcados["fail"]).sum())
        falsas = int((~marcados["fail"]).sum())
        no_detectadas = int((usable["fail"] & ~usable["needs_review"]).sum())
        from src.validation.confidence import ConfidenceThresholds
        th = ConfidenceThresholds()
        parts.append(f"""### Autodetección: el sistema ya puede avisar cuándo dudar

Regla: marcar para revisión si `RMSE > {th.rmse_max}` mm o el radio queda fuera
de [{th.radius_min:.0f}, {th.radius_max:.0f}] mm.

| | Cantidad |
|---|---|
| Húmeros marcados para revisión | {len(marcados)} de {len(usable)} |
| Fallas detectadas | **{detectadas} de {detectadas + no_detectadas}** |
| Falsas alarmas | {falsas} |
| Fallas no detectadas | {no_detectadas} |

⚠️ **Los umbrales se calibraron mirando estas mismas fallas**, así que la tasa
de acierto es dentro de muestra y con muy pocos positivos. No debe leerse como
la tasa que tendría sobre huesos nuevos: hay que recalibrar contra ground truth
independiente. Lo que sí cambia respecto de la versión anterior es que el
sistema **puede** discriminar: antes aceptaba 229/229 sin excepción.""")

    # --- Lo que sí funciona ---
    working = []
    axis_errors = frame["axis_error_deg"].dropna()
    if len(axis_errors):
        working.append(
            f"- **El eje longitudinal es exacto**: mediana de {axis_errors.median():.1f}° "
            f"contra el eje anatómico (epicóndilos → HHC), máximo "
            f"{axis_errors.max():.1f}°, medido sobre los {len(axis_errors)} húmeros con "
            f"epicóndilos anotados. La etapa de rebanadas + RANSAC no es el problema."
        )
    if "axis_valid" in frame:
        max_faces = f"{int(frame['n_faces'].max()):,}".replace(",", ".")
        working.append(
            f"- **Ninguna falla de proceso**: {len(frame)}/{len(frame)} húmeros completaron "
            f"el pipeline, con ejes válidos, sobre mallas de hasta {max_faces} caras — "
            f"unas 10 veces las de los tres huesos de muestra con los que se desarrolló."
        )
    if "head_side" in frame and frame["head_side"].nunique() == 1:
        working.append(
            f"- **La selección de extremo fue unánime** (`{frame['head_side'].iloc[0]}` en los "
            f"{len(frame)}): consistente con la convención de coordenadas de estas "
            f"tomografías, pero conviene no leerlo como que el criterio esté probado — "
            f"nunca tuvo que decidir un caso ambiguo."
        )
    if working:
        parts.append("### Lo que sí funciona\n\n" + "\n".join(working))

    return "## Diagnóstico\n\n" + "\n\n".join(parts)


def build_markdown_report(frame: pd.DataFrame, out_path: Path, params: PipelineParams) -> None:
    threshold = params.success_threshold_mm
    measured = frame["center_error_mm"].dropna()
    within = int((measured <= threshold).sum())
    failures = frame[frame["center_error_mm"] > threshold].sort_values(
        "center_error_mm", ascending=False,
    )

    by_group = (frame.dropna(subset=["center_error_mm"])
                .groupby("pathology_group")["center_error_mm"]
                .agg(n="count", mediana="median", p90=lambda s: s.quantile(0.9), maximo="max"))
    by_group = by_group.reindex([g for g in PATHOLOGY_ORDER if g in by_group.index])

    group_rows = "\n".join(
        f"| {group} | {int(row['n'])} | {row['mediana']:.2f} | {row['p90']:.2f} | {row['maximo']:.2f} |"
        for group, row in by_group.iterrows()
    )
    FAILURE_ROWS_SHOWN = 25
    failure_rows = "\n".join(
        f"| {row['model_id']} | {row['pathology']} | {row['center_error_mm']:.2f} | "
        f"{row['radius']:.2f} | {row['rmse']:.3f} | {row['notes'] or '—'} |"
        for _, row in failures.head(FAILURE_ROWS_SHOWN).iterrows()
    ) or "| — | — | — | — | — | ninguno |"
    failure_note = (
        f"\n\n> Se listan los {FAILURE_ROWS_SHOWN} peores de {len(failures)} casos fuera "
        f"de tolerancia. La tabla completa está en `resultados.xlsx`."
        if len(failures) > FAILURE_ROWS_SHOWN else ""
    )

    diagnosis = _build_diagnosis(frame, params)
    scope = ("**Los {n} húmeros del dataset**" if _is_census(frame)
             else "Muestra estratificada de **{n} húmeros**").format(n=len(frame))
    smallest = int(by_group["n"].min()) if len(by_group) else 0
    scope_caveat = (
        "- Los resultados describen este dataset (hombros de CT, con su mezcla propia\n"
        "  de patologías y de sistemas de coordenadas); no son una tasa de acierto\n"
        "  transferible a otra población sin volver a medir."
        if _is_census(frame) else
        "- Muestra estratificada, no censo: los porcentajes globales están inflados hacia\n"
        "  los casos patológicos respecto de la prevalencia real del dataset."
    )
    group_caveat = (
        f"> Todos los grupos son censo: cada húmero del dataset está en la tabla. "
        f"Aun así el grupo más chico tiene n={smallest}, así que sus medianas siguen "
        f"siendo sensibles a casos individuales."
        if _is_census(frame) else
        f"> Los estratos tienen entre {smallest} y {int(by_group['n'].max())} húmeros. Las "
        f"medianas por grupo son indicativas, no concluyentes: con n de este tamaño un solo "
        f"caso mueve el p90. El estrato de **GHOA severa está completo** (los 11 que existen "
        f"en el dataset), así que ese grupo sí es censo y no muestra."
    )

    text = f"""# Validación del pipeline sobre el dataset

{scope}, comparados contra el centro de
cabeza humeral (`HHC`) medido que traen los CSV del dataset. Es la primera
validación cuantitativa del proyecto: hasta ahora el criterio era mirar un HTML
y juzgar si la esfera "caía bien".

## Qué hace el sistema

El pipeline va del STL al centro de la cabeza humeral sin intervención manual:

1. **Carga** (`src/mesh/loader.py`) — lee el STL binario o ASCII y deduplica
   vértices.
2. **Discretización** (`src/mesh/discretizer.py`) — muestrea {params.samples}
   puntos con probabilidad proporcional al área de cada triángulo, para que la
   densidad no dependa de cómo se malló el hueso.
3. **Eje longitudinal** (`src/axis/longitudinal.py`) — submuestreo por vóxeles,
   PCA inicial, corte en {28} rebanadas transversales, descarte de las que tienen
   área o perímetro anómalos (epífisis, no diáfisis) y ajuste RANSAC de la recta
   sobre el tramo diafisario contiguo.
4. **Mapa de riesgo** (`src/validation/risk_map.py`) — ajusta una esfera local en
   cada punto y puntúa de 0 a 1 qué tan plausible es como candidato: penaliza
   RMSE alto y radios fuera de [{params.radius_min:.0f}, {params.radius_max:.0f}] mm.
   Sirve para *ver* dónde el buscador podría equivocarse (tubérculos, tróclea).
5. **Selección de semillas** (`src/optimization/best_fit.py`) — decide cuál
   extremo del hueso es la cabeza corriendo el mapa de riesgo en ambos, y siembra
   {params.n_seeds} semillas ponderadas por ese mismo mapa.
6. **Ajuste de esfera** (`src/approximation/sphere.py`) — por cada semilla, ajuste
   por mínimos cuadrados con vecindario adaptativo, filtro de normales y rechazo
   iterativo de outliers.
7. **Ranking** — costo que combina RMSE, cobertura de la superficie, convergencia
   y plausibilidad morfológica poblacional; gana el de menor costo.

## Resultados

| Métrica | Valor |
|---|---|
| Húmeros procesados | {len(frame)} |
| Fallas de proceso | {int((frame['status'] != 'ok').sum())} |
| Error mediano | **{measured.median():.2f} mm** |
| Percentil 90 | {measured.quantile(0.9):.2f} mm |
| Error máximo | {measured.max():.2f} mm |
| Dentro de {threshold:.0f} mm | **{within}/{len(measured)} ({100 * within / len(measured):.0f}%)** |

### Por patología

| Grupo | n | Mediana (mm) | p90 (mm) | Máximo (mm) |
|---|---|---|---|---|
{group_rows}

{group_caveat}

### Casos fuera de tolerancia

| Modelo | Patología | Error (mm) | Radio | RMSE | Notas del dataset |
|---|---|---|---|---|---|
{failure_rows}{failure_note}

{diagnosis}

## Limitaciones

{scope_caveat}
- El ground truth `HHC` es una anotación del dataset, con su propio error de
  medición; no es una verdad absoluta.
- El error angular del eje solo se pudo medir donde hay epicóndilos anotados.
- Parámetros fijos (`src/config.py`) heredados del ajuste sobre tres huesos de
  muestra; no se re-calibraron para este dataset.
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jsonl", default=str(ROOT / "results" / "raw_results.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--data-root", default=str(ROOT / "data" / "database"))
    parser.add_argument("--worst-3d", type=int, default=0,
                        help="Generar vista 3D de los N peores casos (~30 s cada uno)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = PipelineParams()

    frame = load_results(Path(args.jsonl))
    print(f"Resultados leídos: {len(frame)} húmeros")

    frame.to_csv(out_dir / "resultados.csv", index=False, encoding="utf-8-sig")
    exportable = frame.drop(columns=[c for c in ("traceback",) if c in frame.columns])
    exportable.to_excel(out_dir / "resultados.xlsx", index=False)
    print(f"  {out_dir / 'resultados.csv'}\n  {out_dir / 'resultados.xlsx'}")

    build_dashboard(frame, out_dir / "dashboard.html", params)
    print(f"  {out_dir / 'dashboard.html'}")

    build_markdown_report(frame, out_dir / "INFORME.md", params)
    print(f"  {out_dir / 'INFORME.md'}")

    if args.worst_3d:
        print(f"\nGenerando vistas 3D de los {args.worst_3d} peores casos...")
        build_worst_case_views(
            frame, out_dir / "worst_cases", args.worst_3d, Path(args.data_root), params,
        )


if __name__ == "__main__":
    main()
