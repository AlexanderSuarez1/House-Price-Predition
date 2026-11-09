# Proyecto Integrador - Fase 2: Aplicacion reutilizable

## Proposito

Esta fase transforma el notebook de la Fase 1 en una aplicacion modular que permite entrenar modelos y generar predicciones mediante comandos, sin ejecutar manualmente el notebook. Se conservan el problema, los datos y las variables del proyecto original.

La aplicacion selecciona el modelo usando solamente el conjunto de desarrollo: `GroupShuffleSplit` reserva una prueba final por combinacion `street-city` y `GroupKFold(n_splits=5)` compara Regresion Lineal, Random Forest y Gradient Boosting dentro de desarrollo. La prueba final se utiliza solo despues de seleccionar el modelo.

## Estructura

```text
fase-2/
├── src/preprocesamiento.py
├── data/
├── models/v1/ y models/v2/   # artefactos historicos con particion aleatoria
├── models/v3/                # artefacto regenerado con grupos y validacion interna
├── tests/test_aplicacion.py
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Instalacion local

```powershell
cd fase-2
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entrenamiento

```powershell
python train.py --data data/DataSet_House_Price_Prediction.csv `
  --target price --output-dir models --version v4 --seed 42 --n-estimators 100
```

Cada version guarda `modelo.joblib`, `metrics.json` y `metadata.json`. El script rechaza una version existente para evitar sobrescritura silenciosa. `metrics.json` registra la validacion interna de los tres candidatos, el modelo elegido y la evaluacion final.

Las versiones `v1` y `v2` son artefactos historicos de la rama anterior, entrenados con `train_test_split` aleatorio. La version `v3` fue regenerada con el protocolo corregido.

## Prediccion

```powershell
python predict.py --model models/v3/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones_v3.csv
```

La entrada puede contener `date` o `year_sale` y `month_sale`. Las columnas adicionales como `street`, `price_per_sqft` o `price` se descartan. Las categorias nuevas se toleran y las columnas numericas faltantes se imputan con la mediana aprendida durante el entrenamiento. La salida conserva el orden original y agrega `prediccion`.

## Pruebas automatizadas

```powershell
pytest -q tests
```

Las pruebas cubren columnas faltantes, entrenamiento, metadatos, rechazo de sobrescritura, prediccion y conservacion del numero de filas.

## Docker

Construir la imagen desde `fase-2`:

```powershell
docker build -t house-price-fase2 .
```

Consultar la ayuda:

```powershell
docker run --rm house-price-fase2 train.py --help
```

Entrenar conservando los artefactos en el host:

```powershell
docker run --rm -v "${PWD}/models:/app/models" house-price-fase2 train.py `
  --data data/DataSet_House_Price_Prediction.csv `
  --target price --output-dir models --version v4 --seed 42
```

Generar predicciones:

```powershell
docker run --rm -v "${PWD}/outputs:/app/outputs" house-price-fase2 predict.py `
  --model models/v3/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones_v3.csv
```

## Fuente y dependencias

La fuente es [House price prediction - Kaggle](https://www.kaggle.com/datasets/shree1992/housedata). Las versiones de Python y dependencias estan fijadas en `requirements.txt`. El dataset original no contiene faltantes; el entrenamiento simula de forma reproducible un 1 % en `sqft_living` para probar la imputacion, igual que la Fase 1.
