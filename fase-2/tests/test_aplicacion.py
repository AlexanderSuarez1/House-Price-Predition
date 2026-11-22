from pathlib import Path
import json
import subprocess
import sys

import joblib 
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from predict import generar_predicciones
from src.preprocesamiento import preparar_variables_entrada
from train import entrenar_modelo


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "DataSet_House_Price_Prediction.csv"
EXAMPLE = ROOT / "data" / "ejemplo_prediccion.csv"


# fixture con muestra pequena
@pytest.fixture(scope="module")
def muestra_datos_csv(tmp_path_factory) -> Path:
    carpeta = tmp_path_factory.mktemp("datos_prueba")
    df = pd.read_csv(DATA)
    muestra = df[df["price"] > 0].head(100).copy()
    ruta_muestra = carpeta / "muestra_entrenamiento.csv"
    muestra.to_csv(ruta_muestra, index=False)
    return ruta_muestra

def test_entrenamiento_y_almacenamiento(muestra_datos_csv, tmp_path):
    """
    Prueba 1 - Entrenamiento y almacenamiento (artefacto cargable, metricas y metadatos)
    """
    salida = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="test-v1",
        seed=42,
        n_estimators=5,
    )
    ruta_modelo = Path(salida["ruta_modelo"])
    ruta_metricas = Path(salida["ruta_metricas"])
    ruta_metadata = Path(salida["ruta_metadata"])

    assert ruta_modelo.exists()
    assert ruta_metricas.exists()
    assert ruta_metadata.exists()

    pipeline_cargado = joblib.load(ruta_modelo)
    assert hasattr(pipeline_cargado, "predict")

    with open(ruta_metricas, "r", encoding="utf-8") as f:
        metricas = json.load(f)
    assert "modelo_final" in metricas and "modelo_base" in metricas

    with open(ruta_metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["version"] == "test-v1"
    assert "variables_entrada" in metadata
    assert "entorno" in metadata


def test_carga_y_prediccion_equivalente(muestra_datos_csv, tmp_path):
    """
    Prueba 2 - Carga y prediccion (1 prediccion por fila y equivalencia antes/despues de guardar)
    """
    entrenamiento = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="test-v2",
        seed=42,
        n_estimators=5,
    )
    entrada = pd.read_csv(EXAMPLE)
    X_preparado = preparar_variables_entrada(entrada)

    # Prediccion del pipeline en memoria antes de recargarlo del disco
    pred_memoria = entrenamiento["pipeline"].predict(X_preparado)

    # Prediccion tras cargar el artefacto guardado
    pipeline_recargado = joblib.load(Path(entrenamiento["ruta_modelo"]))
    pred_disco = pipeline_recargado.predict(X_preparado)
    assert np.allclose(pred_memoria, pred_disco, rtol=1e-7, atol=1e-7)

    # Prediccion mediante la funcion completa de predict.py
    destino = tmp_path / "predicciones.csv"
    resultado = generar_predicciones(
        Path(entrenamiento["ruta_modelo"]), EXAMPLE, destino
    )
    assert destino.exists()
    assert len(resultado) == len(entrada)
    assert "prediccion" in resultado.columns
    assert np.allclose(resultado["prediccion"].to_numpy(), np.round(pred_memoria, 2), atol=1e-2)


def test_valores_faltantes_se_imputan(muestra_datos_csv, tmp_path):
    """
    Prueba 3 - Valores faltantes (celdas nulas en variables numericas se imputan correctamente)
    """
    entrenamiento = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="test-nulos",
        seed=42,
        n_estimators=5,
    )
    datos_con_nulos = pd.read_csv(EXAMPLE)
    datos_con_nulos.loc[0, "sqft_living"] = np.nan
    datos_con_nulos.loc[1, "bedrooms"] = np.nan

    ruta_entrada_nulos = tmp_path / "entrada_con_nulos.csv"
    datos_con_nulos.to_csv(ruta_entrada_nulos, index=False)
    destino = tmp_path / "predicciones_nulos.csv"

    resultado = generar_predicciones(
        Path(entrenamiento["ruta_modelo"]), ruta_entrada_nulos, destino
    )
    assert len(resultado) == len(datos_con_nulos)
    assert resultado["prediccion"].isna().sum() == 0
    assert np.isfinite(resultado["prediccion"]).all()


def test_entrada_invalida_columna_ausente(muestra_datos_csv, tmp_path):
    """
    Prueba 4 - Entrada invalida (columna ausente rechazada y codigo de salida != 0 en CLI)
    """
    datos_invalidos = pd.read_csv(EXAMPLE).drop(columns=["city"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        preparar_variables_entrada(datos_invalidos)

    entrenamiento = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="test-invalido",
        seed=42,
        n_estimators=5,
    )
    ruta_invalido = tmp_path / "invalido.csv"
    datos_invalidos.to_csv(ruta_invalido, index=False)
    ruta_salida = tmp_path / "no_debe_crearse.csv"

    proceso = subprocess.run(
        [
            sys.executable,
            str(ROOT / "predict.py"),
            "--model",
            entrenamiento["ruta_modelo"],
            "--input",
            str(ruta_invalido),
            "--output",
            str(ruta_salida),
        ],
        capture_output=True,
        text=True,
    )
    assert proceso.returncode != 0
    assert "city" in proceso.stderr
    assert not ruta_salida.exists()


def test_versionamiento_conserva_versiones_y_no_sobrescribe(muestra_datos_csv, tmp_path):
    """
    Prueba 5 - Versionamiento (crea dos versiones, conserva la primera intacta y bloquea repetidos)
    """
    v1 = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="ver-1",
        seed=42,
        n_estimators=5,
    )
    metadata_v1_antes = Path(v1["ruta_metadata"]).read_text(encoding="utf-8")
    modelo_v1_bytes_antes = Path(v1["ruta_modelo"]).read_bytes()

    v2 = entrenar_modelo(
        ruta_datos=muestra_datos_csv,
        target_col="price",
        output_dir=tmp_path,
        version="ver-2",
        seed=42,
        n_estimators=10,
    )

    # Verificar que la primera version sigue intacta tras crear la segunda
    assert Path(v1["ruta_metadata"]).read_text(encoding="utf-8") == metadata_v1_antes
    assert Path(v1["ruta_modelo"]).read_bytes() == modelo_v1_bytes_antes

    # Verificar que ambas versiones pueden cargarse y predecir
    pred_v1 = generar_predicciones(Path(v1["ruta_modelo"]), EXAMPLE, tmp_path / "out_v1.csv")
    pred_v2 = generar_predicciones(Path(v2["ruta_modelo"]), EXAMPLE, tmp_path / "out_v2.csv")
    assert len(pred_v1) == 10 and len(pred_v2) == 10

    # Verificar que un identificador repetido se rechaza sin sobrescribir
    with pytest.raises(FileExistsError):
        entrenar_modelo(
            ruta_datos=muestra_datos_csv,
            target_col="price",
            output_dir=tmp_path,
            version="ver-1",
            seed=42,
            n_estimators=5,
        )
    assert Path(v1["ruta_metadata"]).read_text(encoding="utf-8") == metadata_v1_antes