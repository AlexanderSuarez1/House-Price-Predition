# Proyecto Integrador - Fase 4: Monitoreo de la API

Esta fase conserva `fase-3/` sin modificaciones y agrega el registro de actividad
de `POST /predict`. La API reutiliza los modelos de `fase-2/models/`.

## Estructura

- `api.py`: API de Fase 3 con middleware de monitoreo.
- `monitor.py`: agrega un evento JSONL por cada llamada a `/predict`.
- `metricas.py`: calcula solicitudes totales y porcentaje de fallos.
- `activity_log.jsonl`: muestra del registro usado para este reporte.

## Registro de solicitudes

Cada llamada a `/predict`, válida o rechazada, agrega una línea a
`activity_log.jsonl` con estos campos:

```json
{"fecha_hora":"2026-11-22T20:00:00+00:00","estado_http":200,"version":"v3","prediccion":278037.59}
```

El archivo se conserva entre ejecuciones. La ruta puede cambiarse con
`ACTIVITY_LOG_PATH`. En respuestas rechazadas por validación, `version` y
`prediccion` quedan vacíos porque no se utilizó el modelo.

## Métricas

Desde `fase-4/`, ejecutar:

```bash
python metricas.py activity_log.jsonl
```

Con la muestra incluida se obtienen:

```text
Solicitudes totales: 3
Porcentaje de solicitudes fallidas: 33.3 %
```

Se consideran fallidas las respuestas HTTP 4xx y 5xx. Si no existen eventos,
el script muestra total `0` y `sin solicitudes`.

## Versiones y política de reentrenamiento

La API usa `v3` por defecto. Las dos versiones conservadas para comparar son
`v2` y `v3`, ubicadas en `fase-2/models/v2/` y `fase-2/models/v3/`.
Para volver a `v2`, iniciar el servicio con `MODEL_VERSION=v2`; para usar `v3`,
usar `MODEL_VERSION=v3`. El equipo de ML es responsable de ejecutar `POST /train`.

Al reunir 100 casos nuevos con resultado real, el equipo ejecutará `POST /train`
para crear una versión candidata sin sobrescribir las anteriores. Se comparará
la candidata con la versión activa usando MAE sobre los mismos datos reservados
y solo se activará si mejora la métrica.

## Reporte

La muestra contiene dos solicitudes válidas y una rechazada por validación.
El total fue de **3 solicitudes** y el porcentaje de fallos fue **33.3 %**.
Esto indica que una de cada tres solicitudes registradas fue rechazada por el
servicio; la métrica no mide la calidad de las predicciones.

## Ejecución

Instalar las dependencias y arrancar la API:

```bash
pip install -r requirements.txt
MODEL_VERSION=v3 uvicorn api:app --host 0.0.0.0 --port 8000
```

La configuración `MODEL_VERSION` permite cambiar de versión sin modificar el
código. `MODELS_DIR` y `DATA_PATH` conservan la configuración reutilizada de
Fase 3.

## Ensayo reproducible

Sobre un archivo vacío, se deben enviar dos predicciones válidas y una entrada
sin una variable obligatoria. El resultado esperado es una fila por intento:
dos filas `200` con `v3` y la predicción, y una fila `422` con `version` y
`prediccion` vacías. Después, ejecutar `python metricas.py activity_log.jsonl`.

En Docker, montar el registro para conservarlo fuera del contenedor:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v3 \
  -v "$PWD/../fase-2/data:/app/data:ro" \
  -v "$PWD/../fase-2/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  house-price-fase4:1.0
```

La muestra contiene dos solicitudes válidas con `v3` y una solicitud `422`
por ausencia de un campo obligatorio. La tasa de fallos fue `1/3 (33.3 %)`;
el rechazo era el resultado esperado de la prueba de validación.
