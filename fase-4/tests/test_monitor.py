import json

from metricas import calcular_metricas
from monitor import registrar_solicitud


def test_registro_conserva_eventos_y_metricas(tmp_path, monkeypatch):
    ruta = tmp_path / "actividad.jsonl"
    monkeypatch.setenv("ACTIVITY_LOG_PATH", str(ruta))

    registrar_solicitud(200, "v3", 100.5)
    registrar_solicitud(422)

    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 2
    assert json.loads(lineas[0])["estado_http"] == 200
    assert json.loads(lineas[0])["version"] == "v3"
    assert calcular_metricas(ruta) == (2, 50.0)


def test_registro_vacio_no_tiene_porcentaje(tmp_path):
    assert calcular_metricas(tmp_path / "inexistente.jsonl") == (0, None)
