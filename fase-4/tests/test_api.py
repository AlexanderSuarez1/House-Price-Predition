import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_FASE3 = Path(__file__).resolve().parents[1]
if str(ROOT_FASE3) not in sys.path:
    sys.path.insert(0, str(ROOT_FASE3))

from api import app
from train import entrenar_modelo


# Registro valido base
REGISTRO_VALIDO = {
    "date": "2014-05-02",
    "bedrooms": 3.0,
    "bathrooms": 1.5,
    "sqft_living": 1340,
    "sqft_lot": 7912,
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1340,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 2005,
    "city": "Shoreline",
    "statezip": "WA 98133",
}


def _localizar_dataset() -> Path:
    candidatos = [
        ROOT_FASE3 / "data" / "DataSet_House_Price_Prediction.csv",
        ROOT_FASE3.parent / "fase-2" / "data" / "DataSet_House_Price_Prediction.csv",
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError("No se encontró DataSet_House_Price_Prediction.csv para las pruebas.")


@pytest.fixture(scope="module")
def modelo_ligero(tmp_path_factory) -> dict:
    """Entrena un modelo pequeño (100 filas, 5 árboles) en una carpeta temporal para pytest."""
    carpeta = tmp_path_factory.mktemp("api_test_models")
    df = pd.read_csv(_localizar_dataset())
    muestra = df[df["price"] > 0].head(100).copy()
    ruta_muestra = carpeta / "muestra.csv"
    muestra.to_csv(ruta_muestra, index=False)

    directorio_modelos = carpeta / "models"
    entrenar_modelo(
        ruta_datos=ruta_muestra,
        target_col="price",
        output_dir=directorio_modelos,
        version="v-test",
        seed=42,
        n_estimators=5,
    )
    return {"models_dir": directorio_modelos, "version": "v-test"}


@pytest.fixture()
def cliente_api(modelo_ligero, monkeypatch) -> TestClient:
    monkeypatch.setenv("MODELS_DIR", str(modelo_ligero["models_dir"]))
    monkeypatch.setenv("MODEL_VERSION", modelo_ligero["version"])
    ruta_registro = modelo_ligero["models_dir"].parent / "activity_log.jsonl"
    if ruta_registro.exists():
        ruta_registro.unlink()
    monkeypatch.setenv(
        "ACTIVITY_LOG_PATH",
        str(ruta_registro),
    )
    return TestClient(app)


# Prueba GET /health responde 200 y estado activo
def test_health_responde_200(cliente_api):
    respuesta = cliente_api.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["status"] == "ok"
    assert datos["version_activa"] == "v-test"
    assert datos["modelo_disponible"] is True


# Prueba entrada valida devuelve 200, prediccion numerica y version
def test_predict_entrada_valida_devuelve_prediccion_y_version(cliente_api):
    respuesta = cliente_api.post("/predict", json=REGISTRO_VALIDO)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert "prediccion" in datos
    assert "version" in datos
    assert isinstance(datos["prediccion"], (int, float))
    assert math.isfinite(datos["prediccion"]) and datos["prediccion"] > 0
    assert datos["version"] == "v-test"


# Prueba entrada sin campo obligatorio se rechaza con mensaje claro
def test_predict_campo_obligatorio_ausente_se_rechaza(cliente_api):
    entrada_incompleta = dict(REGISTRO_VALIDO)
    entrada_incompleta.pop("city")

    respuesta = cliente_api.post("/predict", json=entrada_incompleta)
    assert respuesta.status_code in (400, 422)
    datos = respuesta.json()
    assert "detail" in datos
    assert "city" in datos["detail"]


# Prueba valor null en columna numerica se imputa con la regla de Fase 2
def test_predict_admite_nulo_en_variable_numerica(cliente_api):
    entrada_con_nulo = dict(REGISTRO_VALIDO)
    entrada_con_nulo["sqft_living"] = None

    respuesta = cliente_api.post("/predict", json=entrada_con_nulo)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert isinstance(datos["prediccion"], (int, float))
    assert math.isfinite(datos["prediccion"])
    assert datos["version"] == "v-test"


# Prueba modelo configurado no disponible devuelve HTTP 503
def test_predict_modelo_no_disponible_devuelve_503(cliente_api, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v-inexistente")

    respuesta = cliente_api.post("/predict", json=REGISTRO_VALIDO)
    assert respuesta.status_code == 503
    datos = respuesta.json()
    assert "v-inexistente" in datos["detail"]


def test_predict_registra_una_fila_por_intento(cliente_api, modelo_ligero):
    respuesta_valida_1 = cliente_api.post("/predict", json=REGISTRO_VALIDO)
    respuesta_valida_2 = cliente_api.post("/predict", json=REGISTRO_VALIDO)
    entrada_incompleta = dict(REGISTRO_VALIDO)
    entrada_incompleta.pop("city")
    respuesta_invalida = cliente_api.post("/predict", json=entrada_incompleta)

    assert respuesta_valida_1.status_code == 200
    assert respuesta_valida_2.status_code == 200
    assert respuesta_invalida.status_code == 422

    ruta_registro = modelo_ligero["models_dir"].parent / "activity_log.jsonl"
    eventos = [linea for linea in ruta_registro.read_text(encoding="utf-8").splitlines() if linea]
    assert len(eventos) == 3