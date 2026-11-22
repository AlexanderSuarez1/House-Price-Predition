cat << 'EOF' > README.md
# Proyecto Integrador – Fase 2: Scripts y Docker

## Modelos y Simulación de Sistemas I – Universidad de Antioquia

### Integrantes
- Hamilton Alexander Suárez Pérez (`AlexanderSuarez1`)
- José Julián Rangel (`JJulianR`)

---

## 1. Descripción del Problema y Relación con la Fase 1
El proyecto aborda un problema de **regresión supervisada** cuyo objetivo es predecir el precio de venta de una vivienda (`price`) a partir de sus características físicas, de ubicación y la fecha de transacción, utilizando el conjunto de datos *House Price Prediction* de Kaggle (4600 observaciones originales y 18 columnas).

En esta Fase 2 se conserva íntegra la carpeta `fase-1/` y se transforma su flujo exploratorio en una aplicación modular dentro de `fase-2/`, ejecutable mediante comandos en línea de comandos (`train.py` y `predict.py`) y contenerizada con Docker, sin necesidad de abrir ni ejecutar manualmente el notebook.

## 2. Correcciones y Cambios Técnicos respecto a la Fase 1 y la Retroalimentación
- **Modularización del preprocesamiento (`src/preprocesamiento.py`):** Se extrajo la limpieza de datos, la transformación de `date` en `year_sale` y `month_sale`, el ordenamiento exacto de las 16 variables predictoras y la exclusión de variables que introducen fuga de información (`price_per_sqft`) o alta cardinalidad (`street`).
- **Prevención de fuga de información y selección por grupos (`v3`):** La separación entre desarrollo y prueba final se realiza antes de ajustar cualquier transformación. En la versión `v3`, `GroupShuffleSplit` (80/20) reserva la prueba final por combinación `street-city` (0 grupos compartidos) y `GroupKFold(n_splits=5)` compara `LinearRegression`, `RandomForestRegressor` y `GradientBoostingRegressor` exclusivamente sobre desarrollo. El `ColumnTransformer` (`SimpleImputer(strategy="median")` y `OneHotEncoder(handle_unknown="ignore")`) queda empaquetado junto al estimador dentro de un único `Pipeline` serializado en `modelo.joblib`.
- **Alineación de entorno y trazabilidad:** Se regeneró `models/v3` en el entorno declarado (`scikit-learn==1.9.1`) para eliminar advertencias de versión al cargar el artefacto, se registró `scikit_learn_version` en `metadata.json` y se habilitó la variable de entorno `GIT_COMMIT` en `obtener_commit_git()` para conservar la referencia del código al entrenar dentro de Docker.
- **Validación estricta y protección de versiones:** Se validan archivos, columnas requeridas y tipos numéricos en inferencia; además, cada versión guarda `modelo.joblib`, `metrics.json` y `metadata.json` en `models/<version>/` rechazando identificadores repetidos para evitar sobrescrituras silenciosas.
- **Pruebas ligeras y Docker completo:** Se ampliaron las pruebas a 5 casos sobre una muestra ligera de 100 registros en directorios temporales (`tmp_path`) y se corrigieron los comandos Docker para montar `data/`, `models/` y `outputs/`.

## 3. Estructura de la Carpeta `fase-2/`
```text
fase-2/
├── src/
│   ├── __init__.py
│   └── preprocesamiento.py    # Limpieza, validación de esquema y ColumnTransformer
├── data/
│   ├── DataSet_House_Price_Prediction.csv
│   └── ejemplo_prediccion.csv # 10 registros reales sin price ni price_per_sqft
├── models/
│   ├── v1/                    # Artefacto histórico (train_test_split aleatorio, 100 árboles)
│   │   ├── modelo.joblib
│   │   ├── metrics.json
│   │   └── metadata.json
│   ├── v2/                    # Artefacto histórico (train_test_split aleatorio, 150 árboles)
│   │   ├── modelo.joblib
│   │   ├── metrics.json
│   │   └── metadata.json
│   └── v3/                    # Artefacto actual (GroupShuffleSplit street-city + GroupKFold)
│       ├── modelo.joblib
│       ├── metrics.json
│       └── metadata.json
├── outputs/                   # Predicciones generadas (CSV ignorados en Git)
├── tests/
│   └── test_aplicacion.py     # 5 pruebas automatizadas con pytest
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## 4. Entorno y Dependencias
- **Versión de Python utilizada:** Python `3.14.2` en entorno Linux (GitHub Codespaces) y contenedor `python:3.14-slim`.
- **Dependencias fijadas en `requirements.txt`:**
  - `numpy==2.5.4`
  - `pandas==3.1.1`
  - `scikit-learn==1.9.1`
  - `joblib==1.6.0`
  - `pytest==9.2.0`

### Instalación local (Terminal Bash / GitHub Codespaces)
```bash
cd fase-2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*(En Windows PowerShell, activar el entorno con `.\.venv\Scripts\Activate.ps1` y usar `` ` `` para continuar líneas.)*

## 5. Acceso a los Datos y Esquema del CSV de Predicción
- **Fuente original:** [Kaggle – House Price Prediction (`shree1992/housedata`)](https://www.kaggle.com/datasets/shree1992/housedata).
- **Datos de entrenamiento:** `data/DataSet_House_Price_Prediction.csv` (copiado desde `fase-1/`). Durante `train.py` se descartan viviendas con `price <= 0` (quedando `4551` observaciones válidas de `4600`) y, dado que el dataset original no tiene faltantes, se simula de forma reproducible un 1 % de nulos en `sqft_living` (con la semilla indicada) para ejercitar la imputación igual que en la Fase 1.
- **Datos de ejemplo para predicción:** `data/ejemplo_prediccion.csv` (10 viviendas reales sin `price` ni `price_per_sqft`).

### Esquema completo de las 16 variables predictoras requeridas
El CSV de entrada para `predict.py` **no requiere la variable objetivo `price`** y no depende del orden de las columnas. Para las variables de fecha se acepta enviar la columna `date` (formato fecha válido, por ejemplo `2014-05-02`, que se transforma automáticamente en `year_sale` y `month_sale`) o incluir directamente `year_sale` y `month_sale`:

| # | Variable | Tipo esperado | Regla de validación y preprocesamiento |
| :-: | :--- | :--- | :--- |
| 1 | `bedrooms` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 2 | `bathrooms` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 3 | `sqft_living` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 4 | `sqft_lot` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 5 | `floors` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 6 | `waterfront` | Numérica (`int`/`float`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 7 | `view` | Numérica (`int`/`float`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 8 | `condition` | Numérica (`int`/`float`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 9 | `sqft_above` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 10 | `sqft_basement` | Numérica (`float`/`int`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 11 | `yr_built` | Numérica (`int`/`float`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 12 | `yr_renovated` | Numérica (`int`/`float`) | Celda vacía (`NaN`) admitida e imputada con mediana de entrenamiento. |
| 13 | `city` | Categórica (`str`) | `OneHotEncoder(handle_unknown="ignore")`; categorías nuevas se toleran. |
| 14 | `statezip` | Categórica (`str`) | `OneHotEncoder(handle_unknown="ignore")`; categorías nuevas se toleran. |
| 15 | `year_sale` *(o `date`)* | Numérica (`int`) / fecha | Extraída de `date` si se incluye esa columna; rechaza fechas inválidas. |
| 16 | `month_sale` *(o `date`)* | Numérica (`int`) / fecha | Extraída de `date` si se incluye esa columna; rechaza fechas inválidas. |

### Reglas ante datos faltantes, tipos inválidos y columnas extra
- **Celda nula vs. columna ausente:** Una **celda numérica vacía (`NaN`)** dentro de una columna existente es válida y se imputa con la mediana aprendida durante el entrenamiento. En contraste, **la ausencia de cualquiera de las columnas requeridas** o la presencia de texto no convertible en una columna numérica se rechaza con un mensaje de error claro y código de salida distinto de cero (`1`), sin crear archivo de salida.
- **Columnas adicionales:** Columnas no utilizadas como `street`, `price_per_sqft` o `price` se descartan automáticamente durante la preparación de variables.
- **Categorías no vistas:** Las etiquetas nuevas en `city` o `statezip` se ignoran de forma segura gracias a `OneHotEncoder(handle_unknown="ignore")`.

## 6. Ejecución de Entrenamiento (`train.py`) – Terminal Bash
Consultar todas las opciones y parámetros disponibles:
```bash
python train.py --help
```

> **Nota importante sobre control de versiones y sobrescritura:** El script verifica que la carpeta `models/<version>/` no exista previamente para impedir sobrescrituras silenciosas. Si desea volver a generar una versión ya existente (por ejemplo `v3`), debe **eliminar antes los archivos anteriores** (`rm -rf models/v3`) o indicar un identificador de versión nuevo (por ejemplo `v4`).

Entrenar una nueva versión (`v4`) con el protocolo por grupos (`GroupShuffleSplit` + `GroupKFold(n_splits=5)`):
```bash
python train.py \
  --data data/DataSet_House_Price_Prediction.csv \
  --target price --output-dir models --version v4 --seed 42 --n-estimators 100
```

Regenerar `v3` desde cero (eliminando primero su directorio anterior):
```bash
rm -rf models/v3
python train.py \
  --data data/DataSet_House_Price_Prediction.csv \
  --target price --output-dir models --version v3 --seed 42 --n-estimators 100
```

### Ubicación de artefactos generados
Cada versión queda almacenada en `models/<version>/` con tres archivos:
- `modelo.joblib`: `Pipeline` completo ajustado (preprocesador + modelo ganador).
- `metrics.json`: Métricas de validación cruzada interna, del modelo final y del modelo base (`DummyRegressor`).
- `metadata.json`: Metadatos completos (versión, fecha UTC, algoritmo, hiperparámetros, semilla, esquema de 16 variables, conteo de datos, partición, versión de Python, versión de `scikit-learn` y `git_commit`).

## 7. Ejecución de Predicción (`predict.py`) – Terminal Bash
Consultar opciones disponibles:
```bash
python predict.py --help
```

Generar predicciones con la versión actual (`v3`) o con versiones históricas (`v1`, `v2`):
```bash
python predict.py \
  --model models/v3/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones_v3.csv
```
*(También se puede pasar `--model models/v1/modelo.joblib` o `--model models/v2/modelo.joblib`.)*

El archivo CSV generado en `outputs/` conserva todas las filas originales en el mismo orden e incorpora la columna `prediccion` con el precio estimado en dólares (redondeado a 2 decimales).

## 8. Métricas de Evaluación, Comparación con el Baseline y Limitaciones

> **Aclaración de protocolos:** Las versiones `v1` y `v2` son artefactos históricos entrenados con partición aleatoria `train_test_split` (3640 entrenamiento / 911 prueba). La versión `v3` fue regenerada con el protocolo corregido: reserva de prueba final por grupos `street-city` (`GroupShuffleSplit`, 3649 desarrollo / 902 prueba final, 0 grupos compartidos) y selección interna mediante `GroupKFold(n_splits=5)`. Por esta razón, sus métricas no son directamente comparables entre protocolos distintos.

### Resultados en conjunto de prueba final vs. modelo base
| Versión | Protocolo de evaluación | Modelo seleccionado | MAE modelo | RMSE modelo | R² modelo | MAE baseline (`DummyRegressor`) | Mejora MAE vs. baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `v1` | Aleatorio 80/20 (histórico) | `GradientBoostingRegressor` (100) | `129760.42` | `247900.41` | `0.5869` | `232654.39` | `+102893.97` |
| `v2` | Aleatorio 80/20 (histórico) | `GradientBoostingRegressor` (150) | `123613.03` | `248152.29` | `0.5861` | `232654.39` | `+109041.36` |
| `v3` | Grupos `street-city` + `GroupKFold(5)` | `RandomForestRegressor` (100) | `122850.98` | `484410.56` | `-0.9987` | `220246.43` | `+97395.45` |

### Validación cruzada interna de candidatos en `v3` (`GroupKFold(n_splits=5)` sobre desarrollo)
| Modelo candidato | MAE promedio | Desviación estándar MAE | RMSE promedio | R² promedio |
| :--- | :---: | :---: | :---: | :---: |
| `RandomForestRegressor` *(elegido)* | `131487.31` | `19425.65` | `511022.69` | `-0.4461` |
| `GradientBoostingRegressor` | `136316.75` | `14047.52` | `463528.50` | `0.0803` |
| `LinearRegression` | `150783.23` | `11594.92` | `408599.54` | `0.4810` |

### Limitaciones conocidas
- La partición de prueba por grupos de `v3` contiene valores extremos de precio que penalizan fuertemente el RMSE (`484410.56`) y llevan el R² a negativo (`-0.9987`), aunque el modelo seleccionado supera ampliamente al baseline en la métrica principal MAE (`122850.98` frente a `220246.43`).
- El conjunto de datos abarca una ventana temporal acotada (año 2014), por lo que el desempeño ante cambios de mercado requiere monitoreo y reentrenamiento periódico.

## 9. Pruebas Automatizadas (`pytest`)
Ejecutar desde la carpeta `fase-2/`:
```bash
python -m pytest -v
```

La suite (`tests/test_aplicacion.py`) emplea una muestra ligera de 100 registros válidos y directorios temporales (`tmp_path`), sin modificar los modelos de entrega ni realizar descargas de red:
1. **`test_entrenamiento_y_almacenamiento`:** Verifica que `entrenar_modelo()` genera un `modelo.joblib` cargable, `metrics.json` (modelo final y baseline) y `metadata.json`.
2. **`test_carga_y_prediccion_equivalente`:** Comprueba que `predict.py` genera una predicción por fila y que las predicciones del pipeline en memoria antes de guardar coinciden numéricamente con las del artefacto recargado desde disco.
3. **`test_valores_faltantes_se_imputan`:** Introduce valores `NaN` en columnas numéricas válidas (`sqft_living`, `bedrooms`) y comprueba que se imputan produciendo predicciones finitas sin nulos.
4. **`test_entrada_invalida_columna_ausente`:** Verifica que eliminar una columna requerida (`city`) lanza `ValueError` en la función y finaliza el CLI `predict.py` con código de salida distinto de cero sin crear el archivo de salida.
5. **`test_versionamiento_conserva_versiones_y_no_sobrescribe`:** Crea dos versiones (`ver-1` y `ver-2`) en un directorio temporal, comprueba que `ver-1` permanece intacta byte a byte, carga y predice con ambas y confirma que repetir `ver-1` lanza `FileExistsError`.

## 10. Contenerización y Persistencia con Docker (Terminal Bash)
El `Dockerfile` parte de `python:3.14-slim`, instala `requirements.txt`, copia `src/`, `train.py`, `predict.py` y `data/`, y excluye modelos, salidas, pruebas, cachés y archivos de Git mediante `.dockerignore`.

### Construir la imagen desde `fase-2/`
```bash
docker build -t house-price-fase2:1.0 .
```

### Consultar ayuda dentro del contenedor
```bash
docker run --rm house-price-fase2:1.0 train.py --help
docker run --rm house-price-fase2:1.0 predict.py --help
```

### Entrenar en Docker con persistencia en el equipo anfitrión y trazabilidad `GIT_COMMIT`
```bash
docker run --rm \
  -e GIT_COMMIT="$(git rev-parse --short HEAD)" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  house-price-fase2:1.0 train.py \
  --data data/DataSet_House_Price_Prediction.csv \
  --target price --output-dir models --version v-docker --seed 42 --n-estimators 100
```

### Generar predicciones en Docker montando `data/` (`:ro`), `models/` (`:ro`) y `outputs/`
```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  house-price-fase2:1.0 predict.py \
  --model models/v3/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones_v3_docker.csv
```

### Verificar persistencia fuera del contenedor y trazabilidad del commit
Una vez finalizados y eliminados los contenedores (`--rm`), los artefactos permanecen en el anfitrión:
```bash
ls -la models/v-docker outputs/predicciones_v3_docker.csv
grep "git_commit" models/v-docker/metadata.json
```

## 11. Evidencia de Ejecución
Salida verificada de regeneración de `v3` y predicción local en Codespaces (sin advertencias de `scikit-learn`):
```text
=== Entrenamiento completado exitosamente (Versión: v3) ===
Modelo final (RandomForestRegressor) -> MAE: 122850.98 | RMSE: 484410.56 | R2: -0.9987
Modelo base (Dummy Mediana)      -> MAE: 220246.43 | RMSE: 350888.83 | R2: -0.0487
Archivos generados:
- Modelo:    models/v3/modelo.joblib
- Métricas:  models/v3/metrics.json
- Metadatos: models/v3/metadata.json

=== Predicción completada exitosamente ===
Modelo utilizado:        models/v3/modelo.joblib
Observaciones predichas: 10
Archivo generado:        outputs/predicciones_v3.csv
```

Resumen de comprobaciones adicionales:
- **`python -m pytest -v`:** 5 pruebas aprobadas (`5 passed`).
- **Docker (`house-price-fase2:1.0`):** Entrenamiento de `v-docker` (con `git_commit` registrado en `metadata.json`), predicción con `v-docker` y `v3` (montando `models` en `:ro`) y persistencia comprobada en `models/` y `outputs/` tras finalizar el contenedor `--rm`.

## 12. Control de Versiones y Fuentes
- **Flujo de trabajo Git:** Desarrollo en ramas `feature/*` por cada integrante, integración mediante Pull Requests revisados hacia `develop` y cierre de entrega en la rama principal (`master`), identificada mediante etiqueta/commit de entrega.
- **Fuentes consultadas:** Dataset *House Price Prediction* en Kaggle (`shree1992/housedata`); documentación oficial de `scikit-learn` (`Pipeline`, `ColumnTransformer`, `GroupShuffleSplit`, `GroupKFold`), `pandas`, `joblib`, `pytest` y Docker.
EOF