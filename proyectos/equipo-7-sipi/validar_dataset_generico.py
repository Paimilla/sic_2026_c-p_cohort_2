"""
Módulo de Validación de Calidad de Datos (QA) - SIPI
====================================================
Este módulo funciona como un "Control de Calidad" automático.
Se asegura de que los datos creados por el módulo anterior sean perfectos
antes de pasárselos a la Inteligencia Artificial para que aprenda.
"""
import pandas as pd
import numpy as np

def validar_dataset(ruta_csv, nombre_contexto):
    """
    Realiza una auditoría matemática de integridad del archivo CSV generado.
    Retorna True si el archivo cumple con los estándares para entrenar XGBoost.
    """
    print(f"  [LOG] Validando estructura matemática de {nombre_contexto}...")
    
    try:
        # [PASO 1] Carga de Datos en Memoria:
        # Leemos el archivo CSV gigante generado en el paso anterior.
        df = pd.read_csv(ruta_csv)
        
        # ==============================================================================
        # [PASO 2] AUDITORÍA MATEMÁTICA (Reglas de Negocio)
        # ==============================================================================
        
        # REGLA A: Volumen Mínimo de Datos
        # La IA necesita mucha historia para aprender. Exigimos al menos 3 años completos (365 * 3).
        if len(df) < (365 * 3):
            print(f"  [WARN] {nombre_contexto}: Insuficientes datos ({len(df)} filas). Se requieren 1095.")
            return False
            
        # REGLA B: Presencia de Columnas Obligatorias
        # Verificamos que el dataset tenga todo lo que el modelo XGBoost espera recibir.
        columnas_req = ['Fecha', 'Caudal_CAMELS_m3_s', 'precipitacion_era5_mm']
        for col in columnas_req:
            if col not in df.columns:
                print(f"  [ERROR] {nombre_contexto}: Falta la columna clave '{col}'.")
                return False
                
        # REGLA C: Tolerancia de Vacíos (Missing Values)
        # Calculamos el porcentaje de "agujeros" sin datos. Si es más del 15%, rechazamos el río.
        pct_nulos = df['Caudal_CAMELS_m3_s'].isnull().mean()
        if pct_nulos > 0.15:
            print(f"  [WARN] {nombre_contexto}: Demasiados nulos en Caudal ({pct_nulos*100:.1f}%).")
            return False
            
        # REGLA D: Control de Varianza (Detectar Anomalías Físicas)
        # Calculamos la Desviación Estándar. Si es cercana a cero, significa que el río 
        # está seco todo el año, o el sensor físico se quedó "pegado" leyendo el mismo número.
        std_caudal = df['Caudal_CAMELS_m3_s'].std()
        if pd.isna(std_caudal) or std_caudal < 0.01:
            print(f"  [WARN] {nombre_contexto}: Varianza casi nula (río seco o sensor defectuoso).")
            return False
            
        # [PASO 3] Aprobación:
        print(f"  [OK] Dataset aprobado para entrenamiento ({len(df)} filas validadas).")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Falla crítica leyendo {nombre_contexto}: {e}")
        return False
