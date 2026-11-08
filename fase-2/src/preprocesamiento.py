import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Listas fijas de columnas definidas en la Fase 1
COLUMNAS_CATEGORICAS = ["city", "statezip"]
COLUMNAS_NUMERICAS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "year_sale", "month_sale"
]
COLUMNAS_ENTRADA = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "city", "statezip", "year_sale", "month_sale"
]


def preparar_variables_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma la fecha, elimina columnas no utilizadas (como street o price_per_sqft)
    y valida que estén presentes todas las columnas predictoras en el orden correcto.
    """
    datos = df.copy()

    # Si viene la columna original 'date', extraemos año y mes de venta
    if "date" in datos.columns:
        fechas = pd.to_datetime(datos["date"], errors="coerce")
        if fechas.isna().any():
            raise ValueError("La columna 'date' contiene valores con formato de fecha inválido.")
        datos["year_sale"] = fechas.dt.year
        datos["month_sale"] = fechas.dt.month
        datos = datos.drop(columns=["date"])

    # Verificar que no falte ninguna de las 16 columnas requeridas por el modelo
    faltantes = [col for col in COLUMNAS_ENTRADA if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en los datos de entrada: {faltantes}")

    # Seleccionar y ordenar las columnas exactamente como en el entrenamiento
    # (esto descarta automáticamente 'street', 'price_per_sqft' u otras columnas extra)
    datos = datos[COLUMNAS_ENTRADA].copy()

    # Validar que las columnas numéricas tengan tipos compatibles
    for col in COLUMNAS_NUMERICAS:
        try:
            datos[col] = pd.to_numeric(datos[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"La columna '{col}' contiene valores que no son numéricos.") from exc

    return datos


def limpiar_datos_entrenamiento(
    df: pd.DataFrame, target_col: str, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reproduce la limpieza de la Fase 1: verifica la columna objetivo,
    introduce el 1% de nulos en sqft_living, filtra precios <= 0 y separa X e y.
    """
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo '{target_col}' en el CSV.")

    datos = df.copy()

    # Introducir 1% de nulos reproducibles en sqft_living como en la Fase 1
    if "sqft_living" in datos.columns and datos["sqft_living"].isna().sum() == 0:
        rng = np.random.default_rng(seed)
        cantidad_nulos = int(len(datos) * 0.01)
        if cantidad_nulos > 0:
            filas_nulas = rng.choice(datos.index, size=cantidad_nulos, replace=False)
            # Usar numpy random state igual al notebook para consistencia
            np.random.seed(seed)
            filas_nulas = np.random.choice(datos.index, cantidad_nulos, replace=False)
            datos.loc[filas_nulas, "sqft_living"] = np.nan

    # Filtrar viviendas con precio igual o menor a cero
    datos = datos[datos[target_col] > 0].copy()
    if datos.empty:
        raise ValueError(f"No quedaron registros válidos con '{target_col}' > 0.")

    y = pd.to_numeric(datos[target_col], errors="raise")
    X = preparar_variables_entrada(datos.drop(columns=[target_col]))

    return X, y


def construir_preprocesador() -> ColumnTransformer:
    """
    Crea el ColumnTransformer con imputación por mediana para variables numéricas
    y OneHotEncoder para variables categóricas, evitando fuga de información.
    """
    pipeline_numerico = Pipeline(
        steps=[("imputacion", SimpleImputer(strategy="median"))]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
            ("num", pipeline_numerico, COLUMNAS_NUMERICAS),
        ]
    )
    return preprocesador