from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from predict import generar_predicciones
from src.preprocesamiento import preparar_variables_entrada
from train import entrenar_modelo


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "DataSet_House_Price_Prediction.csv"
EXAMPLE = ROOT / "data" / "ejemplo_prediccion.csv"


def test_preparar_variables_rechaza_columnas_faltantes():
    datos = pd.read_csv(EXAMPLE).drop(columns=["city"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        preparar_variables_entrada(datos)


def test_entrenamiento_genera_version_y_no_sobrescribe(tmp_path):
    salida = entrenar_modelo(
        ruta_datos=DATA,
        target_col="price",
        output_dir=tmp_path,
        version="test-v1",
        n_estimators=5,
    )
    assert Path(salida["ruta_modelo"]).exists()
    assert Path(salida["ruta_metricas"]).exists()
    assert Path(salida["ruta_metadata"]).exists()
    with pytest.raises(FileExistsError):
        entrenar_modelo(
            ruta_datos=DATA,
            target_col="price",
            output_dir=tmp_path,
            version="test-v1",
            n_estimators=5,
        )


def test_prediccion_conserva_filas_y_agrega_resultado(tmp_path):
    entrenamiento = entrenar_modelo(
        ruta_datos=DATA,
        target_col="price",
        output_dir=tmp_path,
        version="test-v2",
        n_estimators=5,
    )
    entrada = pd.read_csv(EXAMPLE)
    destino = tmp_path / "predicciones.csv"
    resultado = generar_predicciones(
        Path(entrenamiento["ruta_modelo"]), EXAMPLE, destino
    )
    assert len(resultado) == len(entrada)
    assert "prediccion" in resultado.columns
    assert destino.exists()
