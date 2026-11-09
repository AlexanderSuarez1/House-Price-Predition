import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocesamiento import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_ENTRADA,
    COLUMNAS_NUMERICAS,
    construir_preprocesador,
    limpiar_datos_entrenamiento,
)


def obtener_commit_git() -> str:
    """Obtiene el hash corto del commit actual de Git."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return resultado.stdout.strip()
    except Exception:
        return "desconocido"


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE y R2."""
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    r2 = float(r2_score(y_real, y_pred))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def entrenar_modelo(
    ruta_datos: Path,
    target_col: str,
    output_dir: Path,
    version: str,
    seed: int = 42,
    test_size: float = 0.20,
    n_estimators: int = 100,
) -> dict:
    """
    Ejecuta el flujo completo de entrenamiento, evaluación y almacenamiento
    de una versión del modelo sin sobrescribir versiones previas.
    """
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_datos}")

    if not (0.0 < test_size < 1.0):
        raise ValueError("El parámetro --test-size debe estar entre 0 y 1.")

    # Verificar que la versión no exista para evitar sobrescritura
    carpeta_version = output_dir / version
    if carpeta_version.exists() and any(carpeta_version.iterdir()):
        raise FileExistsError(
            f"La versión '{version}' ya existe en '{carpeta_version}'. "
            "Use un identificador de versión diferente para no sobrescribir archivos."
        )

    # Leer y limpiar datos
    df_original = pd.read_csv(ruta_datos)
    X, y, grupos = limpiar_datos_entrenamiento(
        df_original, target_col=target_col, seed=seed
    )

    # Reservar la prueba final por grupos antes de seleccionar modelos.
    separador = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices_desarrollo, indices_prueba = next(
        separador.split(X, y, groups=grupos)
    )
    X_desarrollo = X.iloc[indices_desarrollo]
    X_prueba = X.iloc[indices_prueba]
    y_desarrollo = y.iloc[indices_desarrollo]
    y_prueba = y.iloc[indices_prueba]
    grupos_desarrollo = grupos.iloc[indices_desarrollo]
    grupos_prueba = grupos.iloc[indices_prueba]
    grupos_compartidos = set(grupos_desarrollo).intersection(set(grupos_prueba))
    if grupos_compartidos:
        raise RuntimeError("La prueba final contiene grupos presentes en desarrollo.")

    candidatos = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=n_estimators, random_state=seed
        ),
    }
    cv = GroupKFold(n_splits=5)
    resultados_cv = {}
    pipelines = {}
    for nombre, estimador in candidatos.items():
        pipeline = Pipeline(
            steps=[
                ("preprocesamiento", construir_preprocesador()),
                ("modelo", estimador),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_desarrollo,
            y_desarrollo,
            groups=grupos_desarrollo,
            cv=cv,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
                "R2": "r2",
            },
            n_jobs=-1,
        )
        resultados_cv[nombre] = {
            "MAE_promedio": round(float(-scores["test_MAE"].mean()), 2),
            "MAE_std": round(float(scores["test_MAE"].std()), 2),
            "RMSE_promedio": round(float(-scores["test_RMSE"].mean()), 2),
            "R2_promedio": round(float(scores["test_R2"].mean()), 4),
        }
        pipelines[nombre] = pipeline

    nombre_seleccionado = min(
        resultados_cv, key=lambda nombre: resultados_cv[nombre]["MAE_promedio"]
    )
    pipeline_final = pipelines[nombre_seleccionado]
    pipeline_final.fit(X_desarrollo, y_desarrollo)
    pred_final = pipeline_final.predict(X_prueba)
    metricas_final = calcular_metricas(y_prueba, pred_final)

    modelo_base = DummyRegressor(strategy="median")
    modelo_base.fit(X_desarrollo, y_desarrollo)
    pred_base = modelo_base.predict(X_prueba)
    metricas_base = calcular_metricas(y_prueba, pred_base)

    # Guardar artefactos en la carpeta de la versión
    carpeta_version.mkdir(parents=True, exist_ok=True)
    ruta_modelo = carpeta_version / "modelo.joblib"
    ruta_metricas = carpeta_version / "metrics.json"
    ruta_metadata = carpeta_version / "metadata.json"

    joblib.dump(pipeline_final, ruta_modelo)

    contenido_metricas = {
        "version": version,
        "metrica_principal": "MAE",
        "modelo_final": {
            "nombre": nombre_seleccionado,
            "metricas": metricas_final,
        },
        "modelo_base": {
            "nombre": "DummyRegressor (median)",
            "metricas": metricas_base,
        },
        "mejora_mae_vs_base": round(metricas_base["MAE"] - metricas_final["MAE"], 2),
        "validacion_interna": resultados_cv,
    }

    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(contenido_metricas, f, indent=4, ensure_ascii=False)

    esquema_variables = {
        col: ("categorica (str)" if col in COLUMNAS_CATEGORICAS else "numerica (float/int)")
        for col in COLUMNAS_ENTRADA
    }

    contenido_metadata = {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(),
        "tipo_problema": "Regresión supervisada",
        "variable_objetivo": target_col,
        "algoritmo": nombre_seleccionado,
        "parametros_modelo": {
            "n_estimators": n_estimators,
            "random_state": seed,
        },
        "semilla_aleatoria": seed,
        "variables_entrada": {
            "orden_esperado": COLUMNAS_ENTRADA,
            "tipos_esperados": esquema_variables,
            "categoricas": COLUMNAS_CATEGORICAS,
            "numericas": COLUMNAS_NUMERICAS,
        },
        "datos_utilizados": {
            "fuente": "Kaggle - House Price Prediction",
            "archivo": str(ruta_datos),
            "observaciones_originales": int(len(df_original)),
            "observaciones_validas": int(len(X)),
        },
        "particion": {
            "estrategia": "GroupShuffleSplit por street-city",
            "test_size": test_size,
            "tamano_desarrollo": int(len(X_desarrollo)),
            "tamano_prueba_final": int(len(X_prueba)),
            "grupos_compartidos": len(grupos_compartidos),
            "validacion_interna": "GroupKFold(n_splits=5) sobre desarrollo",
            "metrica_evaluacion": "MAE (complementada con RMSE y R2)",
        },
        "entorno": {
            "python_version": platform.python_version(),
            "archivo_dependencias": "requirements.txt",
            "git_commit": obtener_commit_git(),
        },
    }

    with open(ruta_metadata, "w", encoding="utf-8") as f:
        json.dump(contenido_metadata, f, indent=4, ensure_ascii=False)

    return {
        "ruta_modelo": str(ruta_modelo),
        "ruta_metricas": str(ruta_metricas),
        "ruta_metadata": str(ruta_metadata),
        "metricas": contenido_metricas,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Entrena y guarda una versión reproducible del modelo de precios de viviendas."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Ruta del archivo CSV con datos de entrenamiento.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="Nombre de la columna objetivo (por defecto: price).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directorio base donde se guardarán las versiones (por defecto: models).",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Identificador único de la versión (ej. v1, v2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporción del conjunto de prueba entre 0 y 1 (por defecto: 0.20).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Número de árboles de Gradient Boosting (por defecto: 100).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    try:
        resultado = entrenar_modelo(
            ruta_datos=args.data,
            target_col=args.target,
            output_dir=args.output_dir,
            version=args.version,
            seed=args.seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
        )
        metricas_mod = resultado["metricas"]["modelo_final"]["metricas"]
        metricas_base = resultado["metricas"]["modelo_base"]["metricas"]

        print(f"=== Entrenamiento completado exitosamente (Versión: {args.version}) ===")
        print(f"Modelo final ({resultado['metricas']['modelo_final']['nombre']}) -> MAE: {metricas_mod['MAE']} | RMSE: {metricas_mod['RMSE']} | R2: {metricas_mod['R2']}")
        print(f"Modelo base (Dummy Mediana)      -> MAE: {metricas_base['MAE']} | RMSE: {metricas_base['RMSE']} | R2: {metricas_base['R2']}")
        print(f"Archivos generados:")
        print(f"  - Modelo:    {resultado['ruta_modelo']}")
        print(f"  - Métricas:  {resultado['ruta_metricas']}")
        print(f"  - Metadatos: {resultado['ruta_metadata']}")
    except Exception as exc:
        print(f"Error durante el entrenamiento: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()