import papermill as pm
from pathlib import Path
import datetime
import logging

# ==========================================
# CONFIGURACIÓN DEL LOGGER (Centro de Control)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logging_reentrenamiento.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Papermill_Orchestrator")

class MotorReentrenamiento:
    # Definimos las rutas maestras absolutas
    BASE_DIR = Path.cwd().resolve()
    INPUT_NOTEBOOK = BASE_DIR / "inputs" / "3_modelado_evaluacion.ipynb"
    OUTPUT_DIR = BASE_DIR / "auditoria_modelos"
    
    @classmethod
    def ejecutar_pipeline(cls, nueva_ruta_datos: str, version_modelo: str):
        # 1. Crear carpeta de auditoría si no existe
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # 2. Generar el nombre del reporte inmutable
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_notebook = cls.OUTPUT_DIR / f"reporte_v{version_modelo}_{timestamp}.ipynb"
        
        # 3. Construir RUTAS ABSOLUTAS para inyectar y evitar que el cuaderno se pierda
        ruta_datos_absoluta = cls.BASE_DIR / nueva_ruta_datos
        ruta_modelo_absoluta = cls.BASE_DIR / 'models' / f'motor_mother_v{version_modelo}.pkl'

        logger.info("🚀 Iniciando Pipeline de Reentrenamiento Automatizado (Papermill)...")
        logger.info(f"📁 Ingestando datos desde: {ruta_datos_absoluta}")
        
        # ==========================================
        # EL PAYLOAD: Inyección de variables
        # ==========================================
        parametros_inyeccion = {
            "RUTA_DATOS": str(ruta_datos_absoluta),
            "RUTA_MODELO_SALIDA": str(ruta_modelo_absoluta),
            "SEMILLA": 2026 
        }

        try:
            # Ejecución silenciosa forzando el directorio de trabajo (cwd) a la carpeta inputs/
            pm.execute_notebook(
                input_path=str(cls.INPUT_NOTEBOOK),
                output_path=str(output_notebook),
                parameters=parametros_inyeccion,
                cwd=str(cls.INPUT_NOTEBOOK.parent) 
            )
            
            print("\n" + "="*60)
            logger.info("✅ REENTRENAMIENTO EXITOSO")
            logger.info(f"📊 Documento de Auditoría en: {output_notebook}")
            logger.info(f"🧠 Nuevo Cerebro (.pkl) disponible en: {ruta_modelo_absoluta}")
            print("="*60 + "\n")
            
        except pm.exceptions.PapermillExecutionError as e:
            logger.error(f"❌ Fallo crítico de MLOps en la celda del cuaderno. Revisa el log.")
            logger.info(f"🔍 Cuaderno de diagnóstico parcial guardado en: {output_notebook}")

if __name__ == "__main__":
    # Apuntamos a los datos simulados para la demostración
    ruta_datos_frescos = "data/processed/datos_nuevos_simulados_reentrenamiento.csv" 
    
    MotorReentrenamiento.ejecutar_pipeline(
        nueva_ruta_datos=ruta_datos_frescos,
        version_modelo="4.0" 
    )