import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==========================================
logging.basicConfig(
    filename='logging_motor_prediccion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger()

app = FastAPI(
    title="API MOTHER - Motor Operativo de Triage Hospitalario",
    description="API binaria para predecir Riesgo de Vida en siniestros viales con registro MLOps.",
    version="3.1 (Modelo Binario + MongoDB)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ==========================================
# 1.5 CONEXIÓN A LA "CAJA NEGRA" (MONGODB)
# ==========================================
try:
    # Tiempo de espera corto (2 segundos) para no bloquear la API si Mongo está apagado
    cliente_mongo = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    cliente_mongo.server_info() # Fuerza la validación de la conexión
    db = cliente_mongo["mother_mlops"]
    coleccion_logs = db["registro_operativo_911"]
    logger.info("✅ Conectado a MongoDB: Caja Negra Operativa Activa.")
except Exception as e:
    logger.warning(f"⚠️ No se pudo conectar a MongoDB. El triage funcionará, pero no se guardarán registros. Detalle: {e}")
    coleccion_logs = None

# ==========================================
# 2. CARGA DEL ARTEFACTO (CEREBRO BINARIO)
# ==========================================
try:
    # Ruta relativa a donde levantes el servidor
    MODEL_PATH = Path("models/motor_mother_produccion.pkl")
    artefacto = joblib.load(MODEL_PATH)
    
    # Extraemos las piezas exactas
    modelo = artefacto['modelo_predictivo']
    label_encoders = artefacto['label_encoders']
    umbral_mortal = artefacto['umbral_mortal']
    columnas_requeridas = artefacto['columnas_entrenamiento']
    
    logger.info(f"✅ Artefacto Binario cargado. Umbral táctico operativo: {umbral_mortal}")
except Exception as e:
    logger.error(f"❌ Error crítico al cargar el modelo: {e}")
    modelo = None

# ==========================================
# 3. ESQUEMA DE DATOS DE ENTRADA (PYDANTIC)
# ==========================================
class DatosSiniestro(BaseModel):
    rol_victima: str
    modo_desplazamiento_victima: str
    sexo_victima: str
    edad_victima: int
    franja_horaria: str
    tipo_de_via_siniestro: str
    comuna_siniestro: str
    contraparte_siniestro: str
    participantes_siniestro: str
    anio_siniestro: int
    mes_siniestro: int
    dia_siniestro: int

# ==========================================
# 4. ENDPOINT PRINCIPAL DE TRIAGE
# ==========================================
@app.post("/predecir")
async def realizar_triage(datos: DatosSiniestro):
    if modelo is None:
        raise HTTPException(status_code=500, detail="El motor analítico no está disponible.")

    try:
        # 1. Convertir JSON a DataFrame
        datos_dict = datos.dict()
        df_entrada = pd.DataFrame([datos_dict])
        
        # 2. Ordenar columnas exactamente como espera el modelo
        df_entrada = df_entrada[columnas_requeridas]
        
        # 3. Aplicar los LabelEncoders guardados
        for col, le in label_encoders.items():
            valor_ingresado = str(df_entrada.loc[0, col])
            if valor_ingresado in le.classes_:
                df_entrada[col] = le.transform([valor_ingresado])
            else:
                # Manejo de valores desconocidos para que la API no se caiga
                df_entrada[col] = le.transform(['MISSING']) if 'MISSING' in le.classes_ else 0

        # 4. Lógica de predicción binaria
        proba = modelo.predict_proba(df_entrada)[0]
        prob_mortal = float(proba[1]) # Posición 1 es la clase 'MORTAL'
        
        # 5. Aplicar umbral táctico dinámico del artefacto
        if prob_mortal >= umbral_mortal:
            resultado = 'MORTAL'
            recomendacion = "Triage Rojo/Negro: Activación de Código Trauma Máximo. Riesgo de vida inminente."
        else:
            resultado = 'LEVE'
            recomendacion = "Triage Verde/Amarillo: Riesgo basal. Monitorear evento y despachar unidad básica."

        # ==========================================
        # 6. GUARDADO EN CAJA NEGRA (MONGODB)
        # ==========================================
        if coleccion_logs is not None:
            registro_auditoria = {
                "fecha_hora_consulta": datetime.now(),
                "datos_ingresados": datos_dict,
                "prediccion_cruda_probabilidad": float(prob_mortal),
                "umbral_aplicado": float(umbral_mortal),
                "decision_triage_final": resultado
            }
            try:
                coleccion_logs.insert_one(registro_auditoria)
                logger.info(f"Registro guardado en BD. Triage: {resultado}")
            except Exception as e_db:
                logger.error(f"Error guardando registro en Mongo: {e_db}")

        # 7. Respuesta estructurada al Frontend
        return {
            "triage_sugerido": resultado,
            "confianza_prediccion_porcentaje": round(prob_mortal * 100, 2),
            "protocolo_respuesta": recomendacion,
            "detalles_tecnicos": {
                "umbral_mortal_usado": round(umbral_mortal, 3)
            }
        }

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))