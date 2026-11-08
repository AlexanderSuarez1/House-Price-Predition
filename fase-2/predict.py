import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.preprocesamiento import preparar_variables_entrada


def resolver_ruta_modelo(ruta_modelo: Path) -> Path:
    """
    Permite recibir tanto la ruta directa al archivo .joblib como el directorio
    de la versión (ej. models/v1 o models/v1/modelo.joblib).
    """
    if ruta_modelo.is_dir():
        candidato = ruta_modelo / "modelo.joblib"
        if candidato.exists():
            return candidato
        raise FileNotFoundError(
            f"No se encontró 'modelo.joblib' dentro del directorio '{ruta_modelo}'."
        )
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe el archivo de modelo: '{ruta_modelo}'.")
    return ruta_modelo


def generar_predicciones(
    ruta_modelo: Path,
    ruta_entrada: Path,
    ruta_salida: Path,
) -> pd.DataFrame:
    """
    Carga una versión almacenada del pipeline, valida los datos de entrada
    y guarda las predicciones en un archivo CSV conservando el orden original.
    """
    archivo_modelo = resolver_ruta_modelo(ruta_modelo)

    if not ruta_entrada.exists():
        raise FileNotFoundError(
            f"No existe el archivo CSV de entrada: '{ruta_entrada}'."
        )

    try:
        df_entrada = pd.read_csv(ruta_entrada)
    except Exception as exc:
        raise ValueError(
            f"No se pudo leer el archivo CSV '{ruta_entrada}': {exc}"
        ) from exc

    if df_entrada.empty:
        raise ValueError("El archivo CSV de entrada está vacío.")

    # Validar columnas, tipos de datos y ordenar según el esquema de entrenamiento
    X_preparado = preparar_variables_entrada(df_entrada)

    try:
        pipeline = joblib.load(archivo_modelo)
    except Exception as exc:
        raise RuntimeError(
            f"Error al cargar el artefacto del modelo '{archivo_modelo}': {exc}"
        ) from exc

    # Generar predicciones sin reajustar transformaciones
    predicciones = pipeline.predict(X_preparado)

    # Conservar los datos originales y su orden, añadiendo la columna 'prediccion'
    df_salida = df_entrada.copy()
    if "price" in df_salida.columns:
        df_salida = df_salida.drop(columns=["price"])
    df_salida["prediccion"] = predicciones.round(2)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_salida.to_csv(ruta_salida, index=False)

    return df_salida


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Genera predicciones de precios de viviendas usando una versión guardada del modelo."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Ruta al archivo modelo.joblib o al directorio de la versión (ej. models/v1/modelo.joblib).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ruta del archivo CSV con los datos de entrada (sin necesidad de la columna objetivo).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ruta donde se guardará el archivo CSV con la columna 'prediccion'.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    try:
        df_resultado = generar_predicciones(
            ruta_modelo=args.model,
            ruta_entrada=args.input,
            ruta_salida=args.output,
        )
        print("=== Predicción completada exitosamente ===")
        print(f"Modelo utilizado:        {args.model}")
        print(f"Observaciones predichas: {len(df_resultado)}")
        print(f"Archivo generado:        {args.output}")
    except Exception as exc:
        print(f"Error durante la predicción: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
