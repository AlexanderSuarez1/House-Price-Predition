import argparse
import json
import sys
import urllib.error
import urllib.request


# Primer registro real de fase 2 (sin price ni street)
REGISTRO_EJEMPLO = {
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


def enviar_solicitud(url: str, metodo: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    datos = None
    cabeceras = {"Accept": "application/json"}
    if payload is not None:
        datos = json.dumps(payload).encode("utf-8")
        cabeceras["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=datos, headers=cabeceras, method=metodo)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            cuerpo = json.loads(resp.read().decode("utf-8"))
            return resp.status, cuerpo
    except urllib.error.HTTPError as err:
        cuerpo = json.loads(err.read().decode("utf-8"))
        return err.code, cuerpo


def main():
    parser = argparse.ArgumentParser(
        description="Cliente de prueba para consultar /health y /predict de la API de precios de viviendas."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL base de la API (por defecto: http://localhost:8000).",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    try:
        codigo_health, resp_health = enviar_solicitud(f"{base_url}/health", metodo="GET")
        print(f"GET {base_url}/health -> HTTP {codigo_health}")
        print(json.dumps(resp_health, indent=2, ensure_ascii=False))

        print("\nEnviando registro de ejemplo a POST /predict:")
        print(json.dumps(REGISTRO_EJEMPLO, indent=2, ensure_ascii=False))

        codigo_pred, resp_pred = enviar_solicitud(
            f"{base_url}/predict", metodo="POST", payload=REGISTRO_EJEMPLO
        )
        print(f"\nPOST {base_url}/predict -> HTTP {codigo_pred}")
        print(json.dumps(resp_pred, indent=2, ensure_ascii=False))

        if codigo_health != 200 or codigo_pred != 200:
            sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"No se pudo conectar con la API en '{base_url}': {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()