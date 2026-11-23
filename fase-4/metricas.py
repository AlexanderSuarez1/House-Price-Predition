import argparse
import json
from pathlib import Path


def calcular_metricas(ruta_registro: Path) -> tuple[int, float | None]:
    """Calcula solicitudes totales y porcentaje fallido (HTTP 4xx/5xx)."""
    total = 0
    fallidas = 0
    if ruta_registro.exists():
        with ruta_registro.open(encoding="utf-8") as archivo:
            for numero_linea, linea in enumerate(archivo, start=1):
                if not linea.strip():
                    continue
                try:
                    evento = json.loads(linea)
                    codigo = int(evento["estado_http"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Registro inválido en {ruta_registro}, línea {numero_linea}."
                    ) from exc
                total += 1
                if 400 <= codigo <= 599:
                    fallidas += 1
    if total == 0:
        return 0, None
    return total, fallidas * 100 / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula métricas del registro de la API.")
    parser.add_argument("registro", type=Path, help="Ruta del archivo JSONL.")
    args = parser.parse_args()
    total, porcentaje_fallidas = calcular_metricas(args.registro)
    print(f"Solicitudes totales: {total}")
    if porcentaje_fallidas is None:
        print("Porcentaje de solicitudes fallidas: sin solicitudes")
    else:
        print(f"Porcentaje de solicitudes fallidas: {porcentaje_fallidas:.1f} %")


if __name__ == "__main__":
    main()
