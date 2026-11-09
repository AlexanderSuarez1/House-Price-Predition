# Proyecto Integrador - Fase 1: Prediccion del precio de viviendas

## Modelos y Simulacion de Sistemas I

### Integrantes
- Carlos Andres Teleche
- Hamilton Suarez Perez
- Julian Rangel

## Descripcion del problema
El proyecto aborda un problema de regresion supervisada para predecir el precio de venta de viviendas a partir de caracteristicas fisicas, ubicacion y fecha de venta.

## Dataset
Se utiliza el dataset [House price prediction - Kaggle](https://www.kaggle.com/datasets/shree1992/housedata), un conjunto de viviendas vendidas en el area de Seattle/Washington. Contiene 4.600 registros y 18 variables, entre ellas precio, superficies habitables y del lote, habitaciones, banos, pisos, vista, condicion, fecha y ubicacion.

El archivo utilizado es `DataSet_House_Price_Prediction.csv`. La variable `price_per_sqft` se excluye de los predictores porque se deriva directamente de `price` y produciria fuga de informacion.

## EDA y preparacion
El analisis exploratorio combina graficos con interpretaciones cuantitativas. `sqft_living` y `sqft_above` presentan una correlacion aproximada de 0,88, pero se conservan porque representan superficie habitable total y superficie sobre el nivel del suelo; su diferencia aporta informacion sobre sotanos. Los modelos de arboles pueden manejar esta correlacion y la decision se contrasta mediante validacion cruzada.

El dataset original no presenta valores faltantes. Para evaluar la imputacion de manera controlada, el notebook simula aproximadamente un 1 % de faltantes en `sqft_living`; estos valores no son datos faltantes originales. Tambien se eliminan registros con `price = 0`.

## Evaluacion
La prueba final se reserva antes de seleccionar modelos. La seleccion se realiza mediante validacion cruzada por grupos dentro del conjunto de desarrollo, usando como grupo la combinacion de direccion y ciudad. Esto evita que una misma direccion aparezca en desarrollo y prueba. La prueba final se utiliza unicamente para la evaluacion posterior.

Ademas de MAE, RMSE y R2, se analizan los errores absolutos mas grandes, su contribucion al error cuadratico total y el desempeno por cuartiles de precio. Esto permite explicar si el RMSE esta dominado por pocas viviendas con errores extremos.

## Archivos de la entrega
- `Entrega_1_Modelos_Simulacion_LIMPIO.ipynb`: notebook ejecutable del proyecto.
- `modelo.joblib`: pipeline y modelo entrenado.
- `DataSet_House_Price_Prediction.csv`: conjunto de datos utilizado.
- `README.md`: documentacion de la entrega.
- `requirements.txt`: dependencias del entorno vigente.
- `requirements-modelo-legado.txt`: perfil minimo para el modelo serializado historico.

## Versiones, instalacion y compatibilidad

La entrega vigente de esta rama es la ubicada en `fase-1/`. El notebook vigente es `Entrega_1_Modelos_Simulacion_LIMPIO.ipynb` y sus dependencias estan fijadas en `requirements.txt`. El entorno verificado contiene:

- Python 3.14.3
- pandas 3.0.3
- NumPy 2.4.6
- scikit-learn 1.9.0
- joblib 1.5.3

En Windows PowerShell:

```powershell
cd fase-1
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
python -m jupyter notebook
```

### Modelo serializado historico

El archivo `modelo.joblib` entregado originalmente fue serializado con scikit-learn 1.6.1. Los artefactos `joblib` de scikit-learn no garantizan compatibilidad entre versiones; por eso ese archivo fallo al cargarse con scikit-learn 1.8.0. Para intentar cargarlo, use un entorno separado con `requirements-modelo-legado.txt`:

```powershell
cd fase-1
py -3.14 -m venv .venv-legado
.\.venv-legado\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-modelo-legado.txt
python -c "import joblib; joblib.load('modelo.joblib'); print('Modelo legado cargado')"
```

La opcion recomendada es regenerar el modelo desde el notebook vigente en el entorno fijado en `requirements.txt`. Si se cambia la version de scikit-learn, se debe volver a entrenar, guardar y validar el modelo en ese mismo entorno.

### Documentacion historica

Los documentos de la raiz o de ramas anteriores que describen un `train_test_split` aleatorio y una seleccion directa de Gradient Boosting corresponden a una version historica. No deben confundirse con el notebook vigente de `fase-1`, que reserva la prueba final y selecciona el modelo mediante validacion cruzada por grupos. La documentacion historica de `fase-2` se conserva en la rama remota `feature/fase-2-Alexander` y no forma parte del checkout actual de esta rama.

## Ejecucion
1. Abrir `Entrega_1_Modelos_Simulacion_LIMPIO.ipynb`.
2. Mantener el archivo `DataSet_House_Price_Prediction.csv` disponible junto al notebook.
3. Ejecutar las celdas en orden desde el inicio.
