# Proyecto Integrador – Fase 1: Predicción del precio de viviendas

## Modelos y Simulación de Sistemas I

### Integrantes
- Carlos Andrés Teleche
- Hamilton Suárez Pérez
- Julián Rangel

## Descripción del problema
El proyecto aborda un problema de regresión supervisada cuyo propósito es predecir el precio de venta de una vivienda a partir de características físicas, de ubicación y de la fecha de venta.

## Dataset
Se utiliza el dataset House Price Prediction, obtenido de Kaggle. El conjunto de datos original contiene 4600 registros y 18 variables.

Archivo utilizado:
`DataSet_House_Price_Prediction.csv`

## Objetivo del modelo
Construir un modelo predictivo capaz de estimar la variable objetivo `price` a partir de las características disponibles de las viviendas.

## Preparación de los datos
Se realizó análisis exploratorio de datos, identificación de valores atípicos, tratamiento de valores faltantes, eliminación de registros con precio igual a cero y prevención de fuga de información.

La variable `price_per_sqft` fue excluida de los predictores debido a que se deriva directamente del precio. La variable `street` fue retirada por su alta cardinalidad.

Los datos se dividieron en 80 % para entrenamiento y 20 % para prueba utilizando `random_state=42`.

## Modelo base
Se utilizó `DummyRegressor` con estrategia de mediana como modelo de referencia.

MAE del modelo base: **232654.39**

## Modelo seleccionado
El modelo final seleccionado fue **Gradient Boosting Regressor**.

Resultados sobre el conjunto de prueba:

- MAE: **129694.30**
- RMSE: **247623.76**
- R²: **0.5878**

El modelo Gradient Boosting reduce el MAE en aproximadamente 44.25 % respecto al modelo base.

## Archivos de la entrega
- `Entrega_1_Modelos_Simulacion_LIMPIO.ipynb`: notebook ejecutable del proyecto.
- `modelo.joblib`: pipeline y modelo entrenado.
- `DataSet_House_Price_Prediction.csv`: conjunto de datos utilizado.
- `README.md`: documentación de la entrega.

## Ejecución
1. Abrir `Entrega_1_Modelos_Simulacion_LIMPIO.ipynb`.
2. Mantener el archivo `DataSet_House_Price_Prediction.csv` disponible junto al notebook.
3. Ejecutar las celdas en orden desde el inicio.
4. El notebook realiza el análisis, preparación de datos, entrenamiento, evaluación y almacenamiento del modelo.
