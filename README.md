# Aproximación de Esfera en Cabeza Humeral

Proyecto académico para cargar modelos STL de húmero, estimar el eje diafisario y aproximar la esfera de la cabeza humeral usando validación geométrica, auditoría y visualización 3D interactiva.

El enfoque actual ya no depende de una sola semilla manual ni de comparar el diámetro de la esfera contra la longitud total del húmero. La validación principal usa métricas morfológicas de artroplastia de hombro y ajuste geométrico a la superficie articular.

## Estado Actual

- Carga de STL ASCII/binario.
- Limpieza de malla triangular con conectividad por aristas.
- Muestreo uniforme de superficie con normales.
- Aproximación iterativa de esfera desde una semilla.
- Búsqueda automática principal con RANSAC esférico de 4 puntos y refit robusto.
- Segmentación articular conectada sobre triángulos compatibles.
- Primitivas geométricas de esfera centralizadas en `src/geometry/sphere.py`.
- Reglas de validación de esfera centralizadas en `src/validation/sphere.py`.
- Estimación robusta del eje longitudinal desde la diáfisis, sin usar la esfera.
- Auditoría JSON-friendly de semillas, iteraciones, convergencia y métricas.
- Validación morfológica con ROC, medial offset y posterior offset.
- Demo web local para cargar STL, ver la mejor esfera automática y comparar con semillas manuales.
- Suite de tests: `56 passed`.

## Instalación

```bash
cd /home/jeruah/Documentos/University/geometria/humero
pip install -r requirements.txt
```

Dependencias principales:

```text
numpy
scipy
scikit-learn
matplotlib
plotly
pytest
pytest-cov
```

## Uso

Ejecutar tests:

```bash
pytest -q
```

Abrir la demo web sin STL precargado:

```bash
python examples/demo_interactive_web.py
```

Precargar un STL y ejecutar el best-fit automático al iniciar:

```bash
python examples/demo_interactive_web.py \
  --stl data/sample_humeri/HumeroFinal1.stl \
  --samples 8000 \
  --best-fit-seeds 1000 \
  --best-fit-top 5
```

Usar datos sintéticos para verificación rápida:

```bash
python examples/demo_interactive_web.py --synthetic-demo
```

La interfaz permite:

- Cargar un STL desde el navegador.
- Limpiar la malla y discretizar la superficie.
- Calcular automáticamente la esfera articular con RANSAC.
- Dibujar la esfera completa, el eje longitudinal y la región articular detectada.
- Hacer clic en un punto real de la cabeza para comparar con una semilla manual.
- Ver score, ROC, RMSE, MAD, P95 radial, área de inliers, cobertura angular, compacidad, lado articular, conectividad, offsets y estado de referencia morfológica.

## Flujo Geométrico Actual

1. `STLLoader` carga la malla desde `src/mesh/loader.py`.
2. `MeshCleaner` elimina degenerados, compacta vértices, calcula áreas/normales/centroides y construye adyacencia triangular.
3. `MeshDiscretizer` genera puntos y normales de superficie para visualización y comparación manual.
4. `AxisApproximator` estima el eje diafisario:
   - PCA global solo para orientación inicial.
   - Clasificación completo/incompleto por longitud proyectada.
   - Crop proximal/distal si el modelo está completo.
   - Crop proximal si el modelo está incompleto.
   - Slices transversales sobre la región media.
   - Filtro de cortes con spikes de área o perímetro.
   - RANSAC sobre centros de cortes retenidos.
5. `SphereGeometry` aporta cálculos puros de esfera:
   - esfera desde cuatro puntos,
   - residuo radial,
   - alineación normal-radial,
   - cobertura angular,
   - refit geométrico robusto.
6. `SphereRansacFitter` detecta la esfera articular:
   - Evalúa ambos extremos del húmero para no confundir cabeza humeral y codo.
   - Selecciona 4 caras distribuidas dentro de cada extremo candidato.
   - Calcula la esfera inicial si los puntos no son coplanares.
   - Evalúa residuo radial, concordancia normal y hemisferio articular externo.
   - Conserva el componente conectado más grande.
   - Expande la región por vecindad, reajusta con pérdida Huber y recorta un núcleo conectado de bajo residuo.
7. `SphereValidator` decide validez por RMSE/ROC, soporte superficial, conectividad y morfología.
8. `AuditTrail` registra qué regla se aplicó y qué resultado produjo.

## Best-Fit Automático por RANSAC

La clase principal nueva está en `src/optimization/sphere_ransac.py`:

```python
from src.mesh.cleaner import MeshCleaner
from src.mesh.loader import STLLoader
from src.optimization.sphere_ransac import SphereRansacFitter

mesh = STLLoader.load("humerus.stl")
cleaned = MeshCleaner().clean(mesh.vertices, mesh.faces)
result = SphereRansacFitter().fit(cleaned)
```

La puntuación combina:

- MAD del residuo radial.
- Área total de triángulos compatibles.
- Cobertura angular del casquete detectado.
- Dominancia de un componente conectado.
- Concordancia de normales con la dirección radial.
- Penalización morfológica por z-scores de ROC, medial offset y posterior offset.
- Penalización explícita por cada métrica fuera de referencia.

La morfología participa desde el ranking inicial de RANSAC y también en la comparación final entre extremos, no solo en el reporte final, para evitar que parches distales localmente esféricos superen a la cabeza humeral.

Tras el refit se recorta un núcleo conectado con residuo radial bajo. Por eso la cobertura puede bajar: el objetivo es no inflar la región articular con tuberosidades o cuello que pasan cerca de la esfera pero no forman un casquete limpio.

Además, el soporte debe caer en el hemisferio articular: el lado de la esfera que mira lejos del eje diafisario. Esto evita aceptar parches redondeados del cuello aunque tengan bajo error local.

`src/optimization/best_fit.py` se mantiene como fallback poblacional para datos sin malla limpia, por ejemplo el demo sintético basado solo en puntos.

## Métricas Morfológicas

`AuditTrail.compute_morphological_metrics` calcula:

- `roc`: radio de curvatura de la esfera.
- `medial_offset`: componente medial del vector entre eje y centro de esfera.
- `posterior_offset`: componente posterior del mismo vector.
- `total_offset`: magnitud total del desplazamiento transversal.

Rangos de referencia usados como indicadores estadísticos:

| Métrica | Rango de referencia | Media | SD |
|---|---:|---:|---:|
| ROC | 17-30 mm | 22.5 mm | 2.8 mm |
| Medial offset | 1-14 mm | 6.8 mm | 2.5 mm |
| Posterior offset | 0-10 mm | 2.0 mm | 2.0 mm |

Notas importantes:

- La validación dura mantiene ROC plausible en `17-40 mm` y RMSE menor al umbral configurado.
- Los rangos morfológicos no invalidan por defecto; quedan como referencia y z-score.
- Para invalidar por referencia morfológica se debe pasar `enforce_morphology_reference=True`.
- Las direcciones medial/posterior actuales son aproximaciones de marco global; una fase futura debería derivar un marco anatómico específico del lado y orientación del STL.

## Estructura

```text
humero/
├── data/sample_humeri/              # STL de ejemplo
├── examples/
│   ├── demo_interactive_web.py      # Demo principal con carga STL y best-fit
│   ├── demo_visualization.py
│   └── demo_wayland.py
├── src/
│   ├── approximation/sphere.py      # Ajuste iterativo de esfera
│   ├── audit/trail.py               # Auditoría y métricas morfológicas
│   ├── axis/longitudinal.py         # Eje diafisario robusto
│   ├── geometry/                    # Curvatura, análisis diferencial y primitivas de esfera
│   ├── mesh/                        # Carga, limpieza y discretización STL
│   ├── optimization/
│   │   ├── best_fit.py              # Fallback poblacional de esfera
│   │   ├── sphere_ransac.py         # RANSAC esférico y segmentación articular
│   │   └── refinement.py
│   ├── validation/                  # Validación de semillas y esferas
│   └── visualization/               # Visualización matplotlib/Plotly
└── tests/
    ├── test_audit.py
    ├── test_integration.py
    ├── test_scientific_pipeline.py
    └── test_visualization.py
```

## Resultados de Referencia Rápida

Con RANSAC de 1000 iteraciones sobre los STL incluidos:

```text
Human_humerus_2_reduced.stl:
  ROC 20.020 mm, MO 9.893 mm, PO 2.540 mm, RMSE 0.374 mm, MAD 0.214 mm, P95 0.731 mm, compact 62.2%, lado 56.3%, score 4.949

HumeroFinal1.stl:
  ROC 21.089 mm, MO 3.957 mm, PO 2.806 mm, RMSE 0.303 mm, MAD 0.141 mm, P95 0.695 mm, compact 60.6%, lado 56.6%, score 3.137

Right_humerus_bone_one-piece.stl:
  ROC 21.188 mm, MO 14.402 mm, PO 7.355 mm, RMSE 0.367 mm, MAD 0.199 mm, P95 0.786 mm, compact 56.6%, lado 54.6%, score 22.395
```

Estos valores son controles de funcionamiento, no conclusiones clínicas.

## Limitaciones Actuales

- El RANSAC mejora la separación de superficie articular, pero no sustituye segmentación anatómica validada.
- La detección de cabeza se basa en expansión transversal y puede requerir ajustes en STL muy parciales o mal orientados.
- Las direcciones medial/posterior necesitan un marco anatómico más explícito para estudios bilaterales o poblacionales.
- La calidad de la región articular depende de normales, conectividad y resolución de la malla.

## Próximo Enfoque Recomendado

El cuello de botella ya no es agregar más iteraciones manuales, sino mejorar el modelo geométrico:

1. Validar la región articular RANSAC contra anotaciones manuales.
2. Estimar un marco anatómico local para medial/posterior.
3. Implementar generación del parche faltante usando contorno articular y plano robusto.
4. Guardar reportes comparables por STL para evaluar sensibilidad a iteraciones, tolerancia y muestreo.

## Licencia

Proyecto académico - Universidad.

Última actualización: 2026-07-15
