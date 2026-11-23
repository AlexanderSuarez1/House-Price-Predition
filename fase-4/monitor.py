import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def obtener_ruta_registro() -> Path:
    """Devuelve la ruta configurable del registro persistente de actividad."""
    ruta_configurada = os.environ.get("ACTIVITY_LOG_PATH", "").strip()
    return Path(ruta_configurada) if ruta_configurada else BASE_DIR / "activity_log.jsonl"


def registrar_solicitud(
    estado_http: int,
    version: str = "",
    prediccion: Any = "",
) -> None:
    """Añade un evento JSONL sin borrar los eventos de ejecuciones anteriores."""
    evento = {
        "fecha_hora": datetime.now(timezone.utc).isoformat(),
        "estado_http": estado_http,
        "version": version,
        "prediccion": prediccion,
    }
    ruta = obtener_ruta_registro()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
