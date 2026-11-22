# Proyecto Integrador – Fase 3: API REST

## Modelos y Simulación de Sistemas I – Universidad de Antioquia

### Integrantes
- Hamilton Alexander Suárez Pérez (`AlexanderSuarez1`)
- José Julián Rangel (`JJulianR`)

---

## 1. Descripción y Continuidad con las Fases Anteriores
Esta fase conserva íntegras las carpetas `fase-1/` y `fase-2/` y agrega `fase-3/`, exponiendo el modelo de predicción de precios de viviendas mediante una **API REST construida con FastAPI**.

La API reutiliza directamente la lógica validada en la Fase 2:
- Esquema de las 16 variables predictoras y preprocesamiento (`src/preprocesamiento.py`).
- Carga de versiones guardadas sin reajustar transformaciones (`predict.py`).
- Entrenamiento reproducible y versionado por grupos (`train.py` con `GroupShuffleSplit` por `street-city` y `GroupKFold(n_splits=5)`).
- Datos (`fase-2/data/`) y artefactos versionados (`fase-2/models/v1`, `v2`, `v3`) compartidos mediante rutas relativas locales y montajes en Docker.

## 2. Estructura del Repositorio y de `fase-3/`
```text
House-Price-Predition/
├── fase-1/
├── fase-2/
│   ├── data/
│   │   ├── DataSet_House_Price_Prediction.csv
│   │   └── ejemplo_prediccion.csv
│   └── models/
│       ├── v1/
│       ├── v2/
│       └── v3/
├── fase-3/
│   ├── src/
│   │   ├── __init__.py
│   │   └── preprocesamiento.py    # Preprocesamiento y validación compartidos con Fase 2
│   ├── tests/
│   │   └── test_api.py            # Pruebas automatizadas con pytest y TestClient
│   ├── .dockerignore
│   ├── .gitignore
│   ├── api.py                     # Aplicación FastAPI (/health, /predict, /train)
│   ├── cliente.py                 # Cliente Python de prueba contra /health y /predict
│   ├── Dockerfile                 # Imagen ejecutable en 0.0.0.0:8000
│   ├── predict.py                 # Módulo de resolución de modelos reutilizado de Fase 2
│   ├── README.md
│   ├── requirements.txt           # Dependencias fijadas (FastAPI, Uvicorn, httpx2 y ML)
│   └── train.py                   # Módulo de entrenamiento reutilizado de Fase 2
└── .gitignore
```

## 3. Entorno, Dependencias y Selección de Versión
- **Python utilizado:** Python `3.14.2` (entorno Linux en GitHub Codespaces) y contenedor `python:3.14-slim`.
- **Dependencias fijadas:** `fase-3/requirements.txt` (incluye `numpy`, `pandas`, `scikit-learn`, `joblib`, `pytest`, `fastapi`, `uvicorn` y `httpx2`).

### Variables de entorno para configuración
- `MODEL_VERSION`: Identificador de la versión usada en `POST /predict` (por defecto `v3`). Para cambiar entre versiones disponibles (`v1`, `v2`, `v3` o una nueva creada por `/train`), basta asignar esta variable al iniciar el servidor o el contenedor.
- `MODELS_DIR`: Directorio donde se almacenan las versiones del modelo (por defecto `./models` si existe o `../fase-2/models` en local; en Docker es `/app/models`).
- `DATA_PATH`: Ruta del CSV de entrenamiento para `POST /train` (por defecto `../fase-2/data/DataSet_House_Price_Prediction.csv` en local; en Docker es `/app/data/DataSet_House_Price_Prediction.csv`).
- `GIT_COMMIT`: Hash corto del commit para trazabilidad en `metadata.json` al entrenar dentro de Docker.

### Instalación y ejecución local (Terminal Bash)
```bash
cd fase-3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MODEL_VERSION=v3 uvicorn api:app --host 0.0.0.0 --port 8000
```
*(Para seleccionar otra versión disponible en local, por ejemplo `v2`, ejecute `MODEL_VERSION=v2 uvicorn api:app --host 0.0.0.0 --port 8000`.)*

## 4. Operaciones Implementadas y Validación de Entradas

| Método y ruta | Código éxito | Comportamiento |
| :--- | :---: | :--- |
| `GET /health` | `200 OK` | Devuelve estado operativo, la `version_activa` configurada y si su artefacto está disponible. |
| `POST /predict` | `200 OK` | Valida el JSON de una vivienda (sin requerir `price`), aplica el preprocesamiento guardado y devuelve `prediccion` y `version`. |
| `POST /train` | `201 Created` | Ejecuta el entrenamiento de la Fase 2 (acepta cuerpo vacío), guarda una nueva versión incremental sin sobrescribir las anteriores y devuelve su identificador y métricas. |

### Esquema JSON y manejo de errores
- **Variables obligatorias:** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated` (numéricas), `city` y `statezip` (texto), más la información temporal (`date` en formato fecha válido o bien `year_sale` y `month_sale`). Columnas extra como `street`, `price_per_sqft` o `price` se ignoran automáticamente.
- **Campos ausentes o tipos incompatibles:** Si falta una clave obligatoria o llega un tipo inválido, la API responde **HTTP `422`** indicando el nombre del campo afectado.
- **Valores `null` permitidos:** Si una variable numérica se envía como `null`, se aplica la imputación por mediana aprendida y guardada en el `Pipeline` de la Fase 2.
- **Modelo no disponible:** Si `MODEL_VERSION` apunta a una versión inexistente o no montada, `POST /predict` responde **HTTP `503 Service Unavailable`** con un mensaje comprensible.

## 5. Ejecución con Docker (Terminal Bash)
Construir la imagen desde la carpeta `fase-3/`:
```bash
docker build -t house-price-fase3:1.0 .
```

Iniciar el contenedor publicando el puerto `8000` y montando los datos (`:ro`) y los modelos (lectura/escritura para persistir nuevas versiones generadas por `/train`) de `fase-2/`:
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v3 \
  -e GIT_COMMIT="$(git rev-parse --short HEAD)" \
  -v "$PWD/../fase-2/data:/app/data:ro" \
  -v "$PWD/../fase-2/models:/app/models" \
  house-price-fase3:1.0
```
*(Para cambiar la versión activa en Docker sin reconstruir la imagen, modifique únicamente `-e MODEL_VERSION=v2` o la versión deseada.)*

## 6. Cliente de Prueba y Evidencias Reales

### Ejecutar el cliente Python (`cliente.py`)
Con el servidor activo (local o en Docker), abra otra terminal en `fase-3/` y ejecute:
```bash
python cliente.py --url http://localhost:8000
```

### Ejemplo verificado de `GET /health`
```bash
curl -i http://localhost:8000/health
```
Respuesta (`HTTP 200 OK`):
```json
{
  "status": "ok",
  "mensaje": "API en ejecución",
  "version_activa": "v3",
  "modelo_disponible": true
}
```

### Ejemplo verificado de `POST /predict` (registro real #1 de `ejemplo_prediccion.csv`)
```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2014-05-02",
    "bedrooms": 3.0,
    "bathrooms": 1.5,
    "sqft_living": 1340,
    "sqft_lot": 7912,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1340,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 2005,
    "city": "Shoreline",
    "statezip": "WA 98133"
  }'
```
Respuesta (`HTTP 200 OK`):
```json
{
  "prediccion": 278037.59,
  "version": "v3"
}
```

### Comprobación con valor `null` permitido (`sqft_living: null`)
Respuesta (`HTTP 200 OK`, imputado con la mediana guardada):
```json
{"prediccion":375232.64,"version":"v3"}
```

### Comprobación con campo obligatorio ausente (omitiendo `city`)
Respuesta (`HTTP 422 Unprocessable Content`):
```json
{"detail":"Falta el campo obligatorio 'city'.","errores":["Falta el campo obligatorio 'city'."]}
```

### Comprobación manual de `POST /train` (cuerpo vacío en Docker)
```bash
curl -i -X POST http://localhost:8000/train
```
Respuesta (`HTTP 201 Created`):
```json
{
  "mensaje": "Versión 'v4' entrenada y guardada correctamente.",
  "version": "v4",
  "algoritmo": "RandomForestRegressor",
  "metricas": {
    "MAE": 122850.98,
    "RMSE": 484410.56,
    "R2": -0.9987
  },
  "ruta_modelo": "/app/models/v4/modelo.joblib"
}
```
Los artefactos (`modelo.joblib`, `metrics.json` y `metadata.json`) persisten en `../fase-2/models/v4/` sin alterar `v1`, `v2` ni `v3`.

## 7. Pruebas Automatizadas (`pytest`)
Ejecutar desde la carpeta `fase-3/`:
```bash
python -m pytest -v
```
Las 5 pruebas (`tests/test_api.py`) entrenan un modelo pequeño de prueba (100 filas, 5 árboles) en una carpeta temporal y verifican con `TestClient`:
1. `test_health_responde_200`: confirma `HTTP 200` y estado operativo en `/health`.
2. `test_predict_entrada_valida_devuelve_prediccion_y_version`: valida `HTTP 200`, predicción numérica finita y versión devuelta.
3. `test_predict_campo_obligatorio_ausente_se_rechaza`: omite `city` y verifica rechazo (`422`) mencionando el campo ausente.
4. `test_predict_admite_nulo_en_variable_numerica`: envía `sqft_living: null` y verifica imputación correcta (`200`).
5. `test_predict_modelo_no_disponible_devuelve_503`: configura una versión inexistente y valida `HTTP 503`.