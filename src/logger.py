# src/logger.py

# ============================
# UTILIDADES DE LOGGING
# ============================
# Funcionalidad para redirigir print statements a archivos de log
# manteniendo la salida en consola simultáneamente.
# ============================

import sys
from datetime import datetime
from pathlib import Path
from src.config import LOGS_DIR


class TeeOutput:
    """
    Clase que permite escribir simultáneamente a múltiples destinos (consola y archivo).
    Redirige la salida estándar para que los print() vayan tanto a consola como a archivo.
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Asegurar que se escriba inmediatamente

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()


def setup_logging():
    """
    Configura la redirección de salida para guardar logs en archivos.
    Crea un archivo de log único por ejecución con formato log_YYYYMMDD_HHMMSS.log
    """
    # Crear directorio de logs si no existe
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{timestamp}.log"
    log_path = LOGS_DIR / log_filename

    # Configurar redirección de stdout
    tee = TeeOutput(log_path)
    sys.stdout = tee

    # Imprimir información inicial del log
    print(f"=== INICIO DE EJECUCIÓN DEL PIPELINE ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    return tee


def close_logging(tee_output):
    """
    Cierra la redirección de logging y restaura stdout original.
    """
    if tee_output:
        print("=" * 50)
        print(f"=== FIN DE EJECUCIÓN DEL PIPELINE ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

        # Restaurar stdout original
        sys.stdout = tee_output.terminal
        tee_output.close()