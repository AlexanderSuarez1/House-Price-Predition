# Proyecto Integrador – Fase 2: Scripts y Docker

## Modelos y Simulación de Sistemas I

### Integrantes
- Hamilton Alexander Suárez Pérez
- José Julián Rangel

## Descripción del Problema y Relación con la Fase 1
El proyecto aborda un problema de **regresión supervisada** cuyo objetivo es predecir el precio de venta de una vivienda (`price`) a partir de sus características físicas, de ubicación y fecha de venta, utilizando el conjunto de datos *House Price Prediction* de Kaggle (4600 observaciones originales).

En esta Fase 2 se conserva el enfoque de modelado de la Fase 1 basado en **Gradient Boosting Regressor** comparado contra un modelo base (`DummyRegressor` con estrategia de mediana), transformando el código del notebook original en una aplicación modular de línea de comandos ejecutable tanto en entorno local como dentro de un contenedor Docker.

## Correcciones y Cambios Técnicos respecto a la Fase 1
- **Modularización del preprocesamiento (`src/preprocesamiento.py`):** Se extrajo la limpieza de datos, transformación de la columna `date` en año (`year_sale`) y mes (`month_sale`), y exclusión de variables que producen fuga de información (`price_per_sqft`) o exceso de cardinalidad (`street`).
- **Prevención de fuga de información:** La separación entrenamiento/prueba (80/20) se realiza antes de ajustar cualquier transformación. El `ColumnTransformer` (con `SimpleImputer(strategy="median")` para las 14 variables numéricas y `OneHotEncoder(handle_unknown="ignore")` para `city` y `statezip`) se empaqueta junto al estimador dentro de un único `Pipeline` serializado en `modelo.joblib`.
- **Validación estricta en inferencia:** Se verifica la existencia de archivos, presencia de las columnas requeridas, conversión de tipos numéricos y conservación del orden original de las filas sin requerir la variable objetivo `price`.
- **Control de versiones del modelo:** Cada entrenamiento almacena `modelo.joblib`, `metrics.json` y `metadata.json` en `models/<version>/`, impidiendo sobrescribir versiones previas de forma silenciosa.

## Estructura de la Carpeta `fase-2`
```text
fase-2/
├── src/
│   ├── __init__.py
│   └── preprocesamiento.py
├── data/
│   ├── DataSet_House_Price_Prediction.csv
│   └── ejemplo_prediccion.csv
├── models/
│   ├── v1/
│   │   ├── modelo.joblib
│   │   ├── metrics.json
│   │   └── metadata.json
│   └── v2/
│       ├── modelo.joblib
│       ├── metrics.json
│       └── metadata.json
├── outputs/
├── tests/
│   └── test_aplicacion.py
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Entorno y Dependencias
- **Versión de Python utilizada:** Python 3.14.2 (entorno Linux / GitHub Codespaces y contenedor `python:3.14-slim`).
- **Instalación del entorno local (Terminal Bash):**
```bash
cd fase-2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Acceso a los Datos y Esquema de Entrada
- **Datos de entrenamiento:** Archivo `data/DataSet_House_Price_Prediction.csv` (copiado desde `fase-1/`).
- **Archivo de ejemplo para predicción:** `data/ejemplo_prediccion.csv` (contiene observaciones reales sin las columnas `price` ni `price_per_sqft`).
- **Esquema esperado en predicción:**
  - Variables numéricas: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, y la columna `date` (o directamente `year_sale` y `month_sale`).
  - Variables categóricas: `city`, `statezip`.
  - Comportamiento ante datos especiales: Las columnas adicionales (como `street` o `price_per_sqft`) se descartan automáticamente; las categorías no vistas en entrenamiento son ignoradas por `OneHotEncoder(handle_unknown="ignore")`; las celdas nulas en variables numéricas se imputan con la mediana aprendida en entrenamiento.

## Ejecución de Entrenamiento (`train.py`)
Consultar opciones disponibles:
```bash
python train.py --help
```

> **Aviso Importante sobre la Sobrescritura:** 
> El script cuenta con una validación de seguridad para **prevenir la sobrescritura silenciosa** de versiones existentes. Si las carpetas de las versiones (`models/v1` o `models/v2`) ya se encuentran creadas en el directorio, el proceso se detendrá y generará un error. Por lo tanto, **antes de volver a ejecutar el entrenamiento**, debe asegurarse de eliminar los archivos y carpetas de versiones anteriores utilizando el siguiente comando:
> ```bash
> rm -rf models/v1 models/v2
> ```

Entrenar la **versión `v1`** (100 árboles, semilla 42):
```bash
python train.py --data data/DataSet_House_Price_Prediction.csv --target price --output-dir models --version v1 --seed 42
```

Entrenar la **versión `v2`** (150 árboles, semilla 42):
```bash
python train.py --data data/DataSet_House_Price_Prediction.csv --target price --output-dir models --version v2 --seed 42 --n-estimators 150
```

## Ejecución de Predicción (`predict.py`)
Consultar opciones disponibles:
```bash
python predict.py --help
```

Generar predicciones con la versión `v1` y con la versión `v2`:
```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v1.csv
python predict.py --model models/v2/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```
El archivo CSV generado en `outputs/` conserva todas las filas originales en el mesmo orden e incorpora la columna `prediccion` con el precio estimado en dólares.