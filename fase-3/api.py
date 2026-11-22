import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

# Asegurar importaciones locales
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict import resolver_ruta_modelo
from src.preprocesamiento import preparar_variables_entrada
from train import entrenar_modelo

app = FastAPI(
    title="API de Predicción de Precios de Viviendas",
    description="Fase 3 - Proyecto Integrador: consulta de predicciones y entrenamiento versionado.",
    version="1.0.0",
)

# Cache en memoria para evitar recargar el .joblib en cada solicitud
_CACHE_MODELOS: dict[tuple[str, float], object] = {}


"""
Variables de entorno
"""
def obtener_directorio_modelos() -> Path:
    """Devuelve el directorio de modelos (MODELS_DIR, ./models o ../fase-2/models)."""
    env_dir = os.environ.get("MODELS_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    local_models = BASE_DIR / "models"
    if local_models.exists() and any(local_models.iterdir()):
        return local_models
    return BASE_DIR.parent / "fase-2" / "models"


def obtener_ruta_datos() -> Path:
    """Devuelve la ruta del CSV de entrenamiento (DATA_PATH, ./data o ../fase-2/data)."""
    env_data = os.environ.get("DATA_PATH", "").strip()
    if env_data:
        return Path(env_data)
    local_data = BASE_DIR / "data" / "DataSet_House_Price_Prediction.csv"
    if local_data.exists():
        return local_data
    return BASE_DIR.parent / "fase-2" / "data" / "DataSet_House_Price_Prediction.csv"


def obtener_version_activa() -> str:
    """Version del modelo usada por /predict (seleccionable mediante MODEL_VERSION)."""
    return os.environ.get("MODEL_VERSION", "v3").strip() or "v3"


def cargar_pipeline_version(version: str):
    """Carga el artefacto de una version guardada; lanza HTTP 503 si no esta disponible."""
    directorio_modelos = obtener_directorio_modelos()
    ruta_objetivo = directorio_modelos / version

    try:
        archivo_modelo = resolver_ruta_modelo(ruta_objetivo)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"El modelo configurado (versión '{version}') no está disponible "
                f"en '{directorio_modelos}'. Verifique la variable MODEL_VERSION o entrene una versión."
            ),
        ) from exc

    clave_cache = (str(archivo_modelo.resolve()), archivo_modelo.stat().st_mtime)
    if clave_cache not in _CACHE_MODELOS:
        try:
            _CACHE_MODELOS[clave_cache] = joblib.load(archivo_modelo)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No se pudo cargar el modelo de la versión '{version}': {exc}",
            ) from exc

    return _CACHE_MODELOS[clave_cache]


def generar_siguiente_version(directorio_modelos: Path) -> str:
    """Genera un identificador incremental disponible (v1, v2, v3, v4...) sin sobrescribir."""
    max_num = 0
    if directorio_modelos.exists():
        for elemento in directorio_modelos.iterdir():
            if elemento.is_dir():
                coincidencia = re.fullmatch(r"v(\d+)", elemento.name)
                if coincidencia:
                    max_num = max(max_num, int(coincidencia.group(1)))
    candidato = f"v{max_num + 1}"
    while (directorio_modelos / candidato).exists():
        max_num += 1
        candidato = f"v{max_num + 1}"
    return candidato

"""
Esquemas de entrada y salida (Pydantic)
"""
NumeroOpcional = Union[StrictInt, StrictFloat, None]


class ViviendaEntrada(BaseModel):
    # Ignora columnas extra del CSV como 'street', 'price_per_sqft' o 'price'
    model_config = ConfigDict(extra="ignore")

    bedrooms: NumeroOpcional = Field(..., description="Número de habitaciones (admite null).")
    bathrooms: NumeroOpcional = Field(..., description="Número de baños (admite null).")
    sqft_living: NumeroOpcional = Field(..., description="Área habitable en pies cuadrados (admite null).")
    sqft_lot: NumeroOpcional = Field(..., description="Área del lote en pies cuadrados (admite null).")
    floors: NumeroOpcional = Field(..., description="Número de pisos (admite null).")
    waterfront: NumeroOpcional = Field(..., description="Indicador frente al agua 0/1 (admite null).")
    view: NumeroOpcional = Field(..., description="Calificación de la vista (admite null).")
    condition: NumeroOpcional = Field(..., description="Condición de la vivienda (admite null).")
    sqft_above: NumeroOpcional = Field(..., description="Área sobre el nivel del suelo (admite null).")
    sqft_basement: NumeroOpcional = Field(..., description="Área del sótano (admite null).")
    yr_built: NumeroOpcional = Field(..., description="Año de construcción (admite null).")
    yr_renovated: NumeroOpcional = Field(..., description="Año de renovación o 0 (admite null).")
    city: str = Field(..., min_length=1, description="Ciudad de la vivienda.")
    statezip: str = Field(..., min_length=1, description="Estado y código postal (ej. 'WA 98133').")

    # Alternativa de fecha: enviar 'date' O enviar 'year_sale' y 'month_sale'
    date: Optional[str] = Field(
        default=None,
        description="Fecha de venta (ej. '2014-05-02'). Alternativa a year_sale y month_sale.",
    )
    year_sale: Optional[StrictInt] = Field(
        default=None,
        description="Año de venta (obligatorio si no se envía 'date').",
    )
    month_sale: Optional[StrictInt] = Field(
        default=None,
        description="Mes de venta entre 1 y 12 (obligatorio si no se envía 'date').",
    )

    @model_validator(mode="after")
    def validar_variables_temporales(self):
        if self.date is not None:
            fecha_limpia = self.date.strip()
            if not fecha_limpia or pd.isna(pd.to_datetime(fecha_limpia, errors="coerce")):
                raise ValueError(
                    "El campo 'date' tiene un formato de fecha inválido (ejemplo válido: '2014-05-02')."
                )
        else:
            faltantes = []
            if self.year_sale is None:
                faltantes.append("year_sale")
            if self.month_sale is None:
                faltantes.append("month_sale")
            if faltantes:
                raise ValueError(
                    f"Faltan campos temporales obligatorios: {', '.join(faltantes)} "
                    "(o envíe el campo 'date')."
                )
            if not (1 <= self.month_sale <= 12):
                raise ValueError("El campo 'month_sale' debe estar entre 1 y 12.")
        return self


class EntrenamientoSolicitud(BaseModel):
    version: Optional[str] = Field(
        default=None,
        description="Identificador único de versión (si se omite, se genera automáticamente).",
    )
    seed: int = Field(default=42, description="Semilla aleatoria de reproducibilidad.")
    test_size: float = Field(default=0.20, gt=0.0, lt=1.0, description="Proporción de prueba.")
    n_estimators: int = Field(default=100, ge=1, description="Número de árboles para ensambles.")


"""
Validacion (HTTP 422)
"""
@app.exception_handler(RequestValidationError)
async def manejador_validacion(request: Request, exc: RequestValidationError):
    detalles = []
    for err in exc.errors():
        ubicacion = [str(parte) for parte in err.get("loc", []) if parte != "body"]
        campo = ".".join(ubicacion) if ubicacion else "entrada"
        tipo_error = err.get("type", "")
        if tipo_error == "missing":
            detalles.append(f"Falta el campo obligatorio '{campo}'.")
        else:
            detalles.append(f"Error en '{campo}': {err.get('msg', 'valor inválido')}.")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,  # <--- CAMBIO 1: reemplazar HTTP_422_UNPROCESSABLE_ENTITY
        content={
            "detail": " ".join(detalles),
            "errores": detalles,
        },
    )

"""
Endpoints GET /health, POST /predict, POST /train
"""
@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Indica que la API está en ejecución y reporta la versión configurada."""
    version_activa = obtener_version_activa()
    directorio_modelos = obtener_directorio_modelos()
    modelo_disponible = (directorio_modelos / version_activa / "modelo.joblib").exists()
    return {
        "status": "ok",
        "mensaje": "API en ejecución",
        "version_activa": version_activa,
        "modelo_disponible": modelo_disponible,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(entrada: ViviendaEntrada):
    """Recibe una vivienda en JSON, aplica el preprocesamiento guardado y devuelve predicción y versión."""
    version_activa = obtener_version_activa()
    pipeline = cargar_pipeline_version(version_activa)

    registro = entrada.model_dump()
    if registro.get("date") is None:
        registro.pop("date", None)
    else:
        registro.pop("year_sale", None)
        registro.pop("month_sale", None)

    df_entrada = pd.DataFrame([registro])

    try:
        X_preparado = preparar_variables_entrada(df_entrada)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,  # <--- CAMBIO 2: reemplazar HTTP_422_UNPROCESSABLE_ENTITY
            detail=str(exc),
        ) from exc

    valor_estimado = float(pipeline.predict(X_preparado)[0])
    return {
        "prediccion": round(valor_estimado, 2),
        "version": version_activa,
    }


@app.post("/train", status_code=status.HTTP_201_CREATED)
def train(solicitud: Optional[EntrenamientoSolicitud] = None):
    """Entrena y guarda una nueva versión reutilizando train.py de Fase 2 sin sobrescribir anteriores."""
    parametros = solicitud or EntrenamientoSolicitud()
    ruta_datos = obtener_ruta_datos()
    directorio_modelos = obtener_directorio_modelos()
    target_col = os.environ.get("TARGET_COL", "price").strip() or "price"

    version_objetivo = (
        parametros.version.strip()
        if parametros.version and parametros.version.strip()
        else generar_siguiente_version(directorio_modelos)
    )

    try:
        resultado = entrenar_modelo(
            ruta_datos=ruta_datos,
            target_col=target_col,
            output_dir=directorio_modelos,
            version=version_objetivo,
            seed=parametros.seed,
            test_size=parametros.test_size,
            n_estimators=parametros.n_estimators,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,  # <--- CAMBIO 3: reemplazar HTTP_422_UNPROCESSABLE_ENTITY
            detail=str(exc),
        ) from exc

    return {
        "mensaje": f"Versión '{version_objetivo}' entrenada y guardada correctamente.",
        "version": version_objetivo,
        "algoritmo": resultado["metricas"]["modelo_final"]["nombre"],
        "metricas": resultado["metricas"]["modelo_final"]["metricas"],
        "ruta_modelo": resultado["ruta_modelo"],
    }