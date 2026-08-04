# Mapa de riesgo interactivo para la búsqueda de la cabeza humeral

## Resumen

Esta rama agrega una herramienta de diagnóstico para responder una pregunta concreta:
**¿hay alguna zona del hueso, aparte de la cabeza articular, que pueda engañar a
`HumeralHeadBestFitSearch`** haciéndolo pensar que encontró una esfera válida donde no
la hay? La herramienta final es un visualizador 3D interactivo en el navegador con
sliders para ajustar los hiperparámetros de búsqueda en tiempo real.

## Motivación

`HumeralHeadBestFitSearch` lanza semillas aleatorias sobre la superficie y ajusta
esferas locales para encontrar la cabeza humeral. Antes de confiar en sus resultados en
mallas nuevas, conviene poder visualizar **dónde más** el hueso podría parecer
suficientemente esférico como para producir un falso positivo (por ejemplo, en un
tubérculo o un cóndilo). Este PR agrega esa capacidad de inspección.

## Qué cambia

| Archivo | Cambio |
|---|---|
| `src/visualization/interactive_web.py` | Nuevo método `InteractiveWeb3D.plot_points_colored(points, values, ...)`: nube de puntos Plotly coloreada con una escala continua verde→rojo, con colorbar y hover por punto. |
| `examples/demo_head_risk_map.py` | Script CLI: ajusta una esfera local en cada punto muestreado y colorea según si `HumeralHeadBestFitSearch` lo aceptaría como candidato válido. Genera un HTML estático (`--output`) o lo abre directo en el navegador. |
| `examples/demo_head_risk_map_interactive.py` | Misma idea, pero con un servidor local (`http.server`) y sliders en el navegador para ajustar los hiperparámetros sin volver a ejecutar Python para cada cambio. |

No se tocó ningún módulo de `src/` fuera de `interactive_web.py` (ni `SphericalApproximator`,
ni `HumeralHeadBestFitSearch`, ni el pipeline de auditoría). Es una herramienta de
inspección que **reutiliza** `DifferentialAnalyzer.fit_sphere_to_neighbors` (ya existente)
sin modificar su comportamiento.

## Cómo funciona

Para cada punto muestreado de la superficie:

1. Se buscan sus vecinos dentro de un radio de búsqueda (mismo mecanismo que
   `SphericalApproximator._get_local_points`).
2. Se ajusta una esfera a esos vecinos con `DifferentialAnalyzer.fit_sphere_to_neighbors`.
3. Se calcula un score en `[0, 1]`: `0` si el ajuste tiene RMSE bajo **y** el radio cae en
   el rango fisiológico aceptado (candidato válido para el buscador); `1` si el RMSE está
   en o sobre el máximo tolerado, o el radio queda fuera de rango (el buscador lo
   descartaría).

Los hiperparámetros por defecto son exactamente los que usa `HumeralHeadBestFitSearch`
en producción, para que el mapa refleje el comportamiento real del buscador y no un
criterio inventado:

- `radius_estimate = 22.5 mm` (= `HumeralHeadBestFitSearch.initial_radius`)
- `search_radius = radius_estimate * 1.15` (misma lógica que `SphericalApproximator`)
- `max_error = 2.0 mm` (= `HumeralHeadBestFitSearch.max_error`)
- `radius_min / radius_max = 20 / 40 mm` (rango fisiológico documentado en el README)

Todos son ajustables por CLI o, en la versión interactiva, desde el navegador.

### Por qué esfera local y no curvatura diferencial

El primer enfoque probado fue medir curvatura diferencial (κ₁, κ₂) punto por punto. Se
descartó tras confirmar con datos que no era viable en mallas STL decimadas: ni más
vecinos (k=16→40), ni normales suavizadas por PCA, ni más puntos muestreados (4k→50k)
cambiaron una anisotropía que se quedaba pegada en ~0.97–0.99 en toda la superficie,
incluida la cabeza real. La causa: un STL decimado (en la muestra usada, ~2.570 caras
para todo el hueso) guarda la curvatura real en el ángulo entre caras vecinas, no dentro
de cada triángulo; medir segundas derivadas ahí captura las aristas de la decimación,
no la anatomía. El ajuste de esfera local, al depender solo de posiciones (primer orden),
es mucho más robusto a este problema.

### Arquitectura de dos niveles (versión interactiva)

No todos los hiperparámetros cuestan igual de recalcular:

- **Instantáneo, sin servidor:** `max_error`, `radius_min`, `radius_max`. Solo cambian el
  criterio de aceptación sobre un ajuste ya calculado; se recalculan en JavaScript al
  mover el slider.
- **Con recálculo (~10s para 4.000 puntos):** `radius_estimate`, `search_radius`,
  `min_neighbors`. Cambian el ajuste de esfera en sí; se disparan con un botón
  "Recalcular geometría" en vez de en cada pixel de arrastre del slider.

## Validación

Contra `data/sample_humeri/HumeroFinal1.stl` (1.287 vértices / 2.570 caras), con los
hiperparámetros por defecto:

- 4.000 puntos muestreados.
- 85 puntos con score < 0.3 (candidatos de bajo riesgo).
- **100% de esos 85 puntos cae en Z = 141–155 mm**, el 9% superior del hueso — la cabeza
  anatómica real. Sin zonas impostoras en esta muestra específica (el método sí las
  detectaría si existieran en otro hueso o con otro radio esperado).
- Los 49 tests existentes del repo siguen pasando sin cambios.

## Cómo probarlo

```bash
# Mapa estático, un solo HTML
python examples/demo_head_risk_map.py --stl data/sample_humeri/HumeroFinal1.stl

# Mapa interactivo con sliders en el navegador
python examples/demo_head_risk_map_interactive.py --stl data/sample_humeri/HumeroFinal1.stl
```

## Fuera de alcance

- No se agregó extracción semántica ni integración con Graphify (evaluado por separado,
  fuera de este PR).
- La versión con curvatura diferencial punto a punto se descartó y no forma parte de
  este PR (queda documentada aquí solo como contexto de por qué se eligió el enfoque
  final).
