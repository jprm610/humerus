# Resumen de Implementación

Última actualización: 2026-07-15

Este documento resume el estado real del proyecto después de las iteraciones recientes. La documentación de agentes IA, Copilot y quickstart fue retirada porque ya no representa el flujo de trabajo ni ayuda a mejorar la precisión geométrica.

## Limpieza de Documentación

Se eliminaron:

- `.agent.md`
- `.instructions.md`
- `AGENTS.md`
- `copilot-instructions.md`
- `QUICK_START.md`
- `STATUS.md`

La documentación viva queda concentrada en:

- `README.md`: guía principal de uso, arquitectura, métricas y limitaciones.
- `IMPLEMENTATION_SUMMARY.md`: estado de implementación y decisiones técnicas.

## Capacidades Implementadas

### Carga y Muestreo STL

Archivos:

- `src/mesh/loader.py`
- `src/mesh/discretizer.py`
- `src/mesh/cleaner.py`

Estado:

- Carga STL ASCII y binario.
- Extrae vértices, caras y normales.
- Limpia triángulos degenerados y vértices duplicados.
- Calcula áreas, centroides, normales coherentes y adyacencia por aristas.
- Filtra componentes conectados pequeños cuando se requiere conservar solo el componente principal.
- Discretiza superficie con muestreo uniforme.
- Entrega `surface_points` y `surface_normals` para el pipeline geométrico.

### Ajuste de Esfera Desde Semilla

Archivo:

- `src/approximation/sphere.py`
- `src/geometry/sphere.py`

Estado:

- Implementa `SphericalApproximator`.
- Ajusta esfera usando primitivas comunes de `SphereGeometry`.
- Itera desde una semilla de superficie.
- Registra inicialización, iteraciones y resultado en `AuditTrail`.
- Controla convergencia por cambio de centro/radio.

### Primitivas Geométricas de Esfera

Archivo:

- `src/geometry/sphere.py`

Estado:

- Calcula esfera desde cuatro puntos no coplanares.
- Calcula ajuste algebraico inicial.
- Calcula residuos radiales y alineación normal-radial.
- Calcula cobertura angular.
- Ejecuta refit geométrico robusto con pérdida Huber.
- Es usado por el aproximador clásico, RANSAC, validación de semillas y análisis diferencial.

### Best-Fit Automático por RANSAC

Archivo:

- `src/optimization/sphere_ransac.py`

Clase:

- `SphereRansacFitter`

Estado:

- Evalúa ambos extremos del húmero para no confundir cabeza humeral y codo.
- Selecciona cuatro caras distribuidas por extremo e iteración.
- Calcula esfera inicial desde cuatro puntos no coplanares.
- Filtra por rango plausible de ROC.
- Evalúa residuo radial y concordancia normal.
- Exige que el soporte esté en el hemisferio articular externo respecto al eje diafisario.
- Conserva el componente conectado más grande de triángulos compatibles.
- Incluye ROC, medial offset y posterior offset en el ranking inicial de candidatos.
- Expande la región articular por vecindad con restricciones de residuo, normal y suavidad.
- Reajusta centro/radio con minimización geométrica robusta usando pérdida Huber.
- Recorta un núcleo conectado de bajo residuo radial después del refit.
- Acota el radio durante el refit al rango plausible `17-40 mm`.
- Retorna esfera, región articular, score, área, MAD, P95 radial, cobertura angular, compacidad, lado articular y conectividad.

Componentes del score:

- MAD normalizado.
- Área compatible.
- Cobertura angular.
- Dominancia de componente conectado.
- Concordancia radial de normales.
- Alineación con el lado articular de la cabeza.
- Penalización morfológica por z-scores.
- Penalización explícita por métricas fuera de referencia.

Compatibilidad:

- `src/optimization/best_fit.py` sigue disponible como fallback poblacional para flujos basados solo en puntos.

### Eje Longitudinal Robusto

Archivo:

- `src/axis/longitudinal.py`

Método por defecto:

- `diaphyseal_slice_axis`

Estado:

- No usa la esfera para estimar el eje.
- Usa PCA global solo para orientación inicial.
- Neutraliza densidad mediante voxel downsample.
- Detecta si el húmero parece completo por longitud proyectada.
- En modelo completo descarta cabeza y cola.
- En modelo incompleto descarta solo la porción proximal.
- Divide la región de interés en slices.
- Elimina slices con spikes de área o perímetro.
- Ajusta el eje final con RANSAC sobre centros de slices.
- Devuelve diagnósticos: completitud, crop, slices retenidos, inliers/outliers RANSAC.

### Auditoría y Validación Morfológica

Archivo:

- `src/audit/trail.py`
- `src/validation/sphere.py`

Estado:

- `SphereValidator` concentra las reglas de aceptación/rechazo.
- `AuditTrail` registra pasos en formato JSON-friendly y delega las reglas.
- Valida semillas.
- Valida aproximaciones por RMSE y ROC plausible.
- Valida soporte superficial RANSAC por área, conectividad, conteo de caras y parámetros finitos.
- Calcula métricas morfológicas:
  - ROC.
  - Medial offset.
  - Posterior offset.
  - Total offset transversal.
- Reporta rangos de referencia, medias, desviaciones estándar y z-scores.

Referencias configuradas:

| Métrica | Rango | Media | SD |
|---|---:|---:|---:|
| ROC | 17-30 mm | 22.5 mm | 2.8 mm |
| Medial offset | 1-14 mm | 6.8 mm | 2.5 mm |
| Posterior offset | 0-10 mm | 2.0 mm | 2.0 mm |

Decisión de validación:

- Los rangos de referencia morfológica son indicadores por defecto.
- No invalidan una esfera salvo que se use `enforce_morphology_reference=True`.
- La validación dura mantiene RMSE y ROC plausible como criterios principales.

### Demo Web Interactiva

Archivo:

- `examples/demo_interactive_web.py`

Estado:

- Permite cargar STL desde navegador.
- Limpia la malla STL antes del cálculo automático.
- Discretiza la superficie.
- Ejecuta RANSAC esférico automático al cargar STL real.
- Dibuja superficie, esfera completa, región articular RANSAC, marcador automático y eje.
- Permite hacer clic en una semilla manual para comparar.
- Muestra:
  - Score best-fit.
  - ROC.
  - RMSE.
  - MAD.
  - P95 radial.
  - Área de inliers.
  - Cobertura angular.
  - Compacidad angular.
  - Lado articular.
  - Conectividad.
  - Cobertura.
  - Candidatos válidos.
  - Longitud de eje.
  - Completo/incompleto.
  - Modo de crop.
  - RANSAC inliers.
  - Medial offset.
  - Posterior offset.
  - Estado y z-scores de referencia.

Parámetros CLI relevantes:

```bash
--stl
--samples
--synthetic-demo
--initial-radius
--max-error
--best-fit-seeds
--best-fit-top
--host
--port
--no-browser
```

## Estado de Tests

Comando:

```bash
pytest -q
```

Resultado actual:

```text
56 passed
```

Cobertura funcional en tests:

- Auditoría y serialización.
- Visualización 3D.
- Carga STL ASCII/binaria.
- Discretización uniforme.
- Curvatura en región esférica.
- Estimación de eje robusta en húmero completo e incompleto.
- Validación de offsets morfológicos.
- Delegación de reglas desde auditoría hacia `SphereValidator`.
- Respuesta JSON de la demo web.
- Best-fit automático recuperando una esfera sintética conocida.
- Limpieza de malla y adyacencia triangular.
- Esfera exacta desde cuatro puntos.
- RANSAC esférico tolerando tallo/outliers.
- Serialización de región articular detectada.

## Resultados de Smoke Test en STL de Muestra

Con RANSAC de 1000 iteraciones sobre mallas limpias:

```text
Human_humerus_2_reduced.stl
  ROC: 20.020 mm
  MO: 9.893 mm
  PO: 2.540 mm
  RMSE: 0.374 mm
  MAD: 0.214 mm
  P95 radial: 0.731 mm
  caras compatibles: 1287
  area ratio: 0.0802
  cobertura angular: 0.842
  compacidad angular: 0.622
  lado articular: 0.563
  score: 4.949

HumeroFinal1.stl
  ROC: 21.089 mm
  MO: 3.957 mm
  PO: 2.806 mm
  RMSE: 0.303 mm
  MAD: 0.141 mm
  P95 radial: 0.695 mm
  caras compatibles: 355
  area ratio: 0.1263
  cobertura angular: 0.896
  compacidad angular: 0.606
  lado articular: 0.566
  score: 3.137

Right_humerus_bone_one-piece.stl
  ROC: 21.188 mm
  MO: 14.402 mm
  PO: 7.355 mm
  RMSE: 0.367 mm
  MAD: 0.199 mm
  P95 radial: 0.786 mm
  caras compatibles: 411
  area ratio: 0.0750
  cobertura angular: 0.894
  compacidad angular: 0.566
  lado articular: 0.546
  score: 22.395
```

Estos smoke tests verifican comportamiento del pipeline; no deben interpretarse como validación clínica.

## Decisiones Técnicas Relevantes

### La esfera ya no define el eje

El eje depende solo de la nube de puntos y del análisis diafisario. Esto evita el ciclo lógico donde la validez de la esfera depende del eje y el eje depende de una esfera potencialmente inválida.

### Modelos completos e incompletos

El eje intenta adaptarse a ambos:

- Completo: descarta extremos proximal y distal.
- Incompleto: descarta la porción proximal, conserva más tallo distal disponible.

### Morfología como referencia, no bloqueo automático

Los rangos de ROC/MO/PO ayudan a ordenar y auditar candidatos, pero no invalidan por sí solos salvo modo estricto. Esto evita rechazar geometrías plausibles por variabilidad anatómica o por errores de marco anatómico.

### Best-fit por RANSAC, no por primera convergencia

El resultado automático no es "la primera esfera que funciona"; RANSAC explora hipótesis de cuatro puntos, conserva regiones conectadas y refina el mejor soporte geométrico.

### La conectividad ahora importa

La superficie articular se evalúa como triángulos conectados, no solo como puntos cercanos a una esfera. Esto penaliza parches pequeños y superficies aisladas aunque su RMSE local sea bajo.

## Limitaciones Pendientes

- La segmentación articular RANSAC necesita validación contra anotaciones manuales.
- El marco medial/posterior sigue siendo aproximado.
- La detección de cabeza por expansión transversal puede fallar en STL muy parciales.
- La cobertura depende de conectividad, resolución de malla y calidad de normales.
- La generación del parche faltante de cabeza todavía no está implementada.

## Próximos Pasos Recomendados

1. Validar la región articular RANSAC con casos anotados.
2. Estimar marco anatómico local para medial/posterior en lugar de usar ejes globales.
3. Guardar reportes comparables por STL en JSON/CSV.
4. Evaluar sensibilidad de resultados a iteraciones RANSAC, tolerancia de distancia y ángulo normal.
5. Implementar contorno articular y generación de parche faltante como fase posterior.
