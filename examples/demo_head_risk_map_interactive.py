"""Mapa de riesgo interactivo: sliders en el navegador para ajustar hiperparámetros.

Extiende `demo_head_risk_map.py` con una interfaz web con dos niveles de
control:

- Instantáneo (sin volver a Python): --max-error, --radius-min, --radius-max.
  Solo cambian el criterio de aceptación sobre el rmse/radio ya calculados;
  se recalculan en JavaScript al mover el slider.
- Con recálculo (~unos segundos, botón "Recalcular geometría"):
  --radius-estimate, --search-radius, --min-neighbors. Estos cambian el
  ajuste de esfera local en cada punto, que es el paso caro (~10s para 4000
  puntos), así que se disparan con un botón, no con cada pixel de arrastre.

Uso:

    python3 examples/demo_head_risk_map_interactive.py --stl data/sample_humeri/HumeroFinal1.stl
    python3 examples/demo_head_risk_map_interactive.py --synthetic-demo
"""

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from plotly.utils import PlotlyJSONEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_head_risk_map import synthetic_humerus_points
from src.mesh.discretizer import MeshDiscretizer
from src.mesh.loader import STLLoader
from src.validation.risk_map import fit_local_spheres
from src.visualization.interactive_web import InteractiveWeb3D


def load_points(args: argparse.Namespace) -> np.ndarray:
    """Carga superficie desde STL o usa datos sintéticos (solo posiciones)."""
    if args.stl:
        mesh = STLLoader.load(args.stl)
        discretizer = MeshDiscretizer()
        points, _ = discretizer.discretize_uniform(mesh.vertices, mesh.faces, args.samples, random_seed=42)
        return points
    return synthetic_humerus_points()


def to_json_array(values: np.ndarray) -> List[Optional[float]]:
    """Convierte a lista JSON-serializable, con null donde hay NaN."""
    return [None if not np.isfinite(v) else float(v) for v in values]


def build_html(
    points: np.ndarray,
    rmse: np.ndarray,
    fitted_radius: np.ndarray,
    args: argparse.Namespace,
) -> str:
    """Construye el HTML con sliders y el handler de recalculo."""
    scores = np.ones(len(points))
    viz = InteractiveWeb3D(title="Mapa de riesgo: cabeza humeral")
    viz.plot_points_colored(points, scores, name="Riesgo", colorbar_title="score")
    fig_json = json.dumps(viz.fig, cls=PlotlyJSONEncoder)

    initial = {
        "rmse": to_json_array(rmse),
        "fitted_radius": to_json_array(fitted_radius),
        "radius_estimate": args.radius_estimate,
        "search_radius": args.search_radius if args.search_radius is not None else round(args.radius_estimate * 1.15, 4),
        "min_neighbors": args.min_neighbors,
        "max_error": args.max_error,
        "radius_min": args.radius_min,
        "radius_max": args.radius_max,
    }
    initial_json = json.dumps(initial)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mapa de riesgo interactivo - cabeza humeral</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f8; color: #1f2933; display: flex; height: 100vh; }}
    #sidebar {{ width: 300px; flex-shrink: 0; background: white; border-right: 1px solid #d9dee3; padding: 16px; overflow-y: auto; box-sizing: border-box; }}
    #plot {{ flex: 1; height: 100vh; }}
    fieldset {{ border: 1px solid #d9dee3; border-radius: 6px; margin-bottom: 16px; }}
    legend {{ font-weight: 600; font-size: 13px; padding: 0 4px; }}
    label {{ display: block; font-size: 12px; margin-top: 10px; }}
    input[type=number], input[type=range] {{ width: 100%; box-sizing: border-box; }}
    .row {{ display: flex; align-items: center; gap: 6px; }}
    .row span {{ font-size: 11px; color: #52606d; min-width: 46px; text-align: right; }}
    button {{ width: 100%; margin-top: 12px; border: 1px solid #b9c2cc; background: #f8fafc; padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
    button:disabled {{ opacity: 0.55; cursor: default; }}
    #status {{ font-size: 12px; margin-top: 10px; color: #52606d; min-height: 32px; }}
    #stats {{ font-size: 12px; margin-top: 8px; padding: 8px; background: #eef1f4; border-radius: 6px; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <fieldset>
      <legend>Ajuste de esfera local (recalcula)</legend>
      <label>Radio estimado (mm)
        <input id="radiusEstimate" type="number" step="0.5" min="1">
      </label>
      <label class="row" style="margin-top:10px;">
        <input id="autoSearchRadius" type="checkbox" checked style="width:auto;">
        auto = radio_estimado &times; 1.15
      </label>
      <label>Radio de busqueda (mm)
        <input id="searchRadius" type="number" step="0.5" min="1">
      </label>
      <label>Vecinos minimos
        <input id="minNeighbors" type="number" step="1" min="4">
      </label>
      <button id="recomputeBtn">Recalcular geometria</button>
      <div id="status"></div>
    </fieldset>

    <fieldset>
      <legend>Criterio de aceptacion (instantaneo)</legend>
      <label>Error maximo (RMSE, mm)
        <div class="row"><input id="maxError" type="range" min="0.2" max="10" step="0.1"><span id="maxErrorValue"></span></div>
      </label>
      <label>Radio minimo (mm)
        <div class="row"><input id="radiusMin" type="range" min="5" max="60" step="0.5"><span id="radiusMinValue"></span></div>
      </label>
      <label>Radio maximo (mm)
        <div class="row"><input id="radiusMax" type="range" min="5" max="60" step="0.5"><span id="radiusMaxValue"></span></div>
      </label>
      <div id="stats"></div>
    </fieldset>
  </div>
  <div id="plot"></div>

  <script>
    const fig = {fig_json};
    const initial = {initial_json};
    let rmseArr = initial.rmse;
    let radiusArr = initial.fitted_radius;

    const plot = document.getElementById('plot');
    Plotly.newPlot(plot, fig.data, fig.layout, {{responsive: true}});

    const radiusEstimateInput = document.getElementById('radiusEstimate');
    const autoSearchRadius = document.getElementById('autoSearchRadius');
    const searchRadiusInput = document.getElementById('searchRadius');
    const minNeighborsInput = document.getElementById('minNeighbors');
    const recomputeBtn = document.getElementById('recomputeBtn');
    const statusEl = document.getElementById('status');
    const maxErrorInput = document.getElementById('maxError');
    const radiusMinInput = document.getElementById('radiusMin');
    const radiusMaxInput = document.getElementById('radiusMax');
    const maxErrorValue = document.getElementById('maxErrorValue');
    const radiusMinValue = document.getElementById('radiusMinValue');
    const radiusMaxValue = document.getElementById('radiusMaxValue');
    const statsEl = document.getElementById('stats');

    radiusEstimateInput.value = initial.radius_estimate;
    searchRadiusInput.value = initial.search_radius;
    minNeighborsInput.value = initial.min_neighbors;
    maxErrorInput.value = initial.max_error;
    radiusMinInput.value = initial.radius_min;
    radiusMaxInput.value = initial.radius_max;

    function syncSearchRadius() {{
      if (autoSearchRadius.checked) {{
        searchRadiusInput.value = (parseFloat(radiusEstimateInput.value) * 1.15).toFixed(2);
        searchRadiusInput.disabled = true;
      }} else {{
        searchRadiusInput.disabled = false;
      }}
    }}
    autoSearchRadius.addEventListener('change', syncSearchRadius);
    radiusEstimateInput.addEventListener('input', syncSearchRadius);
    syncSearchRadius();

    function computeScores(maxError, radiusMin, radiusMax) {{
      const radiusHalf = 0.5 * (radiusMax - radiusMin);
      const scores = new Array(rmseArr.length);
      for (let i = 0; i < rmseArr.length; i++) {{
        const r = rmseArr[i];
        const fr = radiusArr[i];
        if (r === null || fr === null) {{ scores[i] = 1.0; continue; }}
        const rmseRatio = r / maxError;
        const outsideBy = Math.max(radiusMin - fr, fr - radiusMax, 0);
        const radiusRatio = radiusHalf > 0 ? outsideBy / radiusHalf : (outsideBy > 0 ? 1.0 : 0.0);
        scores[i] = Math.min(Math.max(Math.max(rmseRatio, radiusRatio), 0), 1);
      }}
      return scores;
    }}

    function hoverText(scores) {{
      return scores.map((s, i) => {{
        const r = rmseArr[i];
        const fr = radiusArr[i];
        if (r === null) {{ return `score=${{s.toFixed(3)}}<br>sin vecinos suficientes`; }}
        return `score=${{s.toFixed(3)}}<br>rmse=${{r.toFixed(3)}}mm<br>radio_ajustado=${{fr.toFixed(2)}}mm`;
      }});
    }}

    function updatePlot() {{
      const maxError = parseFloat(maxErrorInput.value);
      const radiusMin = parseFloat(radiusMinInput.value);
      const radiusMax = parseFloat(radiusMaxInput.value);
      maxErrorValue.textContent = maxError.toFixed(1) + 'mm';
      radiusMinValue.textContent = radiusMin.toFixed(1) + 'mm';
      radiusMaxValue.textContent = radiusMax.toFixed(1) + 'mm';

      const scores = computeScores(maxError, radiusMin, radiusMax);
      Plotly.restyle(plot, {{'marker.color': [scores], text: [hoverText(scores)]}}, [0]);

      const total = scores.length;
      const candidates = scores.filter(s => s < 1.0).length;
      const lowRisk = scores.filter(s => s < 0.3).length;
      statsEl.innerHTML =
        `Candidatos (score&lt;1.0): <code>${{candidates}}/${{total}}</code><br>` +
        `Bajo riesgo (score&lt;0.3): <code>${{lowRisk}}/${{total}}</code>`;
    }}

    [maxErrorInput, radiusMinInput, radiusMaxInput].forEach(el => el.addEventListener('input', updatePlot));

    recomputeBtn.addEventListener('click', async function() {{
      const radiusEstimate = parseFloat(radiusEstimateInput.value);
      const searchRadius = autoSearchRadius.checked
        ? radiusEstimate * 1.15
        : parseFloat(searchRadiusInput.value);
      const minNeighbors = parseInt(minNeighborsInput.value, 10);

      recomputeBtn.disabled = true;
      statusEl.textContent = 'Recalculando ajuste de esfera local (puede tardar unos segundos)...';
      try {{
        const response = await fetch('/recompute', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            radius_estimate: radiusEstimate,
            search_radius: searchRadius,
            min_neighbors: minNeighbors,
          }}),
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || 'Error desconocido');
        }}
        rmseArr = data.rmse;
        radiusArr = data.fitted_radius;
        updatePlot();
        statusEl.innerHTML =
          `Geometria recalculada: radio_estimado=<code>${{radiusEstimate}}mm</code>, ` +
          `search_radius=<code>${{searchRadius.toFixed(2)}}mm</code>, ` +
          `min_neighbors=<code>${{minNeighbors}}</code>.`;
      }} catch (error) {{
        statusEl.textContent = `Error: ${{error.message}}`;
      }} finally {{
        recomputeBtn.disabled = false;
      }}
    }});

    updatePlot();
  </script>
</body>
</html>"""


def run_server(args: argparse.Namespace) -> None:
    print("Cargando y muestreando superficie...")
    points = load_points(args)
    print(f"  {len(points)} puntos muestreados")

    search_radius = args.search_radius if args.search_radius is not None else args.radius_estimate * 1.15
    print(f"Ajuste inicial (radio_estimado={args.radius_estimate}mm, search_radius={search_radius:.2f}mm)...")
    rmse, fitted_radius = fit_local_spheres(points, search_radius, args.radius_estimate, args.min_neighbors)

    state: Dict[str, Any] = {"points": points, "rmse": rmse, "fitted_radius": fitted_radius}
    html = build_html(points, rmse, fitted_radius, args).encode("utf-8")

    class RiskMapHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *handler_args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            if self.path == "/recompute":
                self._handle_recompute()
                return
            self.send_error(404)

        def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
            response = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def _handle_recompute(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                radius_estimate = float(payload["radius_estimate"])
                search_radius = float(payload["search_radius"])
                min_neighbors = int(payload["min_neighbors"])
                if radius_estimate <= 0 or search_radius <= 0 or min_neighbors < 4:
                    raise ValueError("Parametros fuera de rango (radios > 0, min_neighbors >= 4)")

                new_rmse, new_fitted_radius = fit_local_spheres(
                    state["points"], search_radius, radius_estimate, min_neighbors
                )
                state["rmse"] = new_rmse
                state["fitted_radius"] = new_fitted_radius
                self._send_json({
                    "rmse": to_json_array(new_rmse),
                    "fitted_radius": to_json_array(new_fitted_radius),
                })
            except Exception as exc:
                self._send_json({"error": str(exc)}, status_code=400)

    server = ThreadingHTTPServer((args.host, args.port), RiskMapHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Mapa de riesgo interactivo: {url}")
    print("Ajusta los sliders (instantaneo) o cambia radio/vecinos y presiona 'Recalcular geometria'.")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stl", type=str, default=None, help="Ruta a archivo STL")
    parser.add_argument("--synthetic-demo", action="store_true", help="Usar superficie sintetica")
    parser.add_argument("--samples", type=int, default=4000, help="Puntos a muestrear del STL")
    parser.add_argument("--radius-estimate", type=float, default=22.5, help="Radio esperado de la cabeza, mm")
    parser.add_argument("--search-radius", type=float, default=None, help="Radio de vecindario local, mm (default: radius-estimate * 1.15)")
    parser.add_argument("--max-error", type=float, default=2.0, help="RMSE maximo aceptado, mm")
    parser.add_argument("--radius-min", type=float, default=20.0, help="Radio fisiologico minimo, mm")
    parser.add_argument("--radius-max", type=float, default=40.0, help="Radio fisiologico maximo, mm")
    parser.add_argument("--min-neighbors", type=int, default=15, help="Vecinos minimos para ajustar la esfera local")
    parser.add_argument("--host", default="127.0.0.1", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8766, help="Puerto del servidor")
    parser.add_argument("--no-browser", action="store_true", help="No abre navegador; util para tests")
    args = parser.parse_args()

    if not args.stl and not args.synthetic_demo:
        parser.error("Usa --stl <archivo> o --synthetic-demo")
    return args


if __name__ == "__main__":
    run_server(parse_args())
