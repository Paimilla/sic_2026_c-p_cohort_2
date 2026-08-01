"""
Orquestador Nacional de Entrenamiento (Zero Interacción) - SIPI
Este script es el encargado de iterar sobre las 516 estaciones de CAMELS-CL,
descargar su historial satelital (ERA5), limpiarlo, generar los CSV y
mandar a entrenar los modelos (Archivos .pkl) a la GPU.
"""
import os
import sys
import time
import glob
import json
import re
import traceback
import pandas as pd
from datetime import datetime

# Asegurar codificación UTF-8 en consola y errores en Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Definición de Directorios Estructurados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMELS_DIR = os.path.join(BASE_DIR, "CAMELS_CL_v202201")
LOG_FILE = os.path.join(BASE_DIR, "log_ejecucion_nacional.txt")

# Nuevas carpetas organizadas
DATOS_DIR = os.path.join(BASE_DIR, "datos_entrenamiento")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos_entrenados")

# Crear carpetas si no existen
os.makedirs(DATOS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Importación de módulos accesorios de procesamiento
try:
    from procesar_cuenca_desatendido import procesar_cuenca_desatendido, consultar_era5_archive_open_meteo, slugify
    from validar_dataset_generico import validar_dataset
    from entrenar_modelo_gpu import entrenar_modelo_cuenca
except ImportError:
    # Definición de fallback para entornos donde no están todos los scripts auxiliares
    def slugify(text):
        return re.sub(r'[\W_]+', '_', str(text).lower()).strip('_')

# ============================================================================
# MASTER ORQUESTADOR NACIONAL DESATENDIDO DE CHILE (516 ESTACIONES CAMELS-CL)
# 100% Automatizado, Tolerante a Fallos, Ininterrumpido (Zero Interacción Humana)
# ============================================================================

def log_master(mensaje):
    """Escribe un mensaje de registro tanto en consola como en un archivo log."""
    now_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    linea = f"{now_str} {mensaje}"
    print(linea, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linea + "\n")

def determinar_region(lat):
    """Asigna una Macro-Región basándose en la latitud."""
    if lat > -21.5: return "Arica y Parinacota y Tarapaca"
    elif lat > -26.0: return "Antofagasta"
    elif lat > -29.0: return "Atacama"
    elif lat > -32.0: return "Coquimbo"
    elif lat > -33.8: return "Valparaiso y RM"
    elif lat > -35.0: return "OHiggins y Maule"
    elif lat > -38.5: return "Nuble Biobio y Araucania"
    elif lat > -41.5: return "Los Rios y Los Lagos"
    else: return "Aysen y Magallanes"

def procesar_todas_las_estaciones_chile():
    """
    Función principal de ejecución del Pipeline.
    1. Lee el inventario oficial CAMELS-CL
    2. Identifica estaciones ya procesadas para retomar donde quedó (Checkpoint)
    3. Descarga clima desde satélite ERA5 (1940-2023)
    4. Solicita a XGBoost que genere los modelos predictivos (Archivos .pkl)
    """
    log_master("=" * 90)
    log_master("🇨🇱 INICIANDO PROCESAMIENTO Y ENTRENAMIENTO NACIONAL DE CHILE DESATENDIDO")
    log_master("   Base de Datos: CAMELS-CL (516 Estaciones Fluviométricas en 16 Regiones)")
    log_master("=" * 90)

    # 1. Leer Atributos CAMELS (Lat, Lon, Area, Nombre)
    ruta_attr = os.path.join(CAMELS_DIR, "catchment_attributes.csv")
    if not os.path.exists(ruta_attr):
        log_master(f"[ERROR CRÍTICO] No se encontró catchment_attributes.csv en {CAMELS_DIR}")
        return

    df_attr = pd.read_csv(ruta_attr)
    df_attr['st_id'] = df_attr['gauge_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_attr['region_chile'] = df_attr['gauge_lat'].apply(determinar_region)

    # 2. Leer Caudales Históricos para pre-filtrar ríos secos
    ruta_q = os.path.join(CAMELS_DIR, "q_m3s_day.csv")
    df_q_sub = pd.DataFrame()
    if os.path.exists(ruta_q):
        df_q = pd.read_csv(ruta_q)
        if 'date' in df_q.columns:
            df_q_sub = df_q[(df_q['date'] >= '2015-01-01') & (df_q['date'] <= '2020-06-06')]

    total_estaciones = len(df_attr)
    log_master(f"Total de estaciones en CAMELS-CL a evaluar: {total_estaciones}")

    # Identificar estaciones ya procesadas para evitar trabajo duplicado (Checkpointing)
    archivos_existentes = glob.glob(os.path.join(DATOS_DIR, "datos_entrenamiento_*.csv"))
    ids_procesados = set()
    for f in archivos_existentes:
        base = os.path.basename(f)
        for part in base.split('_'):
            if part.isdigit() and len(part) >= 7:
                ids_procesados.add(part)

    log_master(f"Estaciones ya procesadas previamente: {len(ids_procesados)}")
    log_master(f"Estaciones pendientes por procesar: {total_estaciones - len(ids_procesados)}")

    exitosos, omitidos, fallidos = 0, 0, 0
    t_inicio_total = time.time()

    for idx, row in df_attr.iterrows():
        st_id = str(row['st_id']).strip()
        nombre_raw = str(row.get('gauge_name', f'estacion_{st_id}'))
        nombre_slug = slugify(nombre_raw)
        region_nombre = str(row['region_chile'])
        region_slug = slugify(region_nombre)
        lat, lon = float(row.get('gauge_lat', -32.0)), float(row.get('gauge_lon', -70.5))

        # Definir Rutas donde se guardarán los resultados
        nombre_csv_salida = f"datos_entrenamiento_{region_slug}_{st_id}_{nombre_slug}.csv"
        ruta_csv_salida = os.path.join(DATOS_DIR, nombre_csv_salida)

        m_reg_24 = os.path.join(MODELOS_DIR, f"modelo_caudal_datos_entrenamiento_{region_slug}_{st_id}_{nombre_slug}_24h.pkl")
        m_clf_24 = os.path.join(MODELOS_DIR, f"modelo_inundacion_datos_entrenamiento_{region_slug}_{st_id}_{nombre_slug}_24h.pkl")

        # CHECKPOINT 1: Si ya existe el CSV y los modelos 24h en las carpetas, omitir
        if os.path.exists(ruta_csv_salida) and os.path.exists(m_reg_24) and os.path.exists(m_clf_24):
            log_master(f"[{idx+1}/{total_estaciones}] ⏩ Checkpoint OK: Station {st_id} ({nombre_raw}) ya entrenada. Omitiendo.")
            omitidos += 1
            continue

        # CHECKPOINT 2: Pre-filtro instantáneo de caudal activo (Omitir estaciones sin registros o cauces secos < 0.05 m3/s)
        if not df_q_sub.empty:
            if st_id not in df_q_sub.columns:
                log_master(f"[{idx+1}/{total_estaciones}] ⏩ Filtro Caudal: Station {st_id} ({nombre_raw}) sin registros fluviométricos. Omitiendo al instante.")
                fallidos += 1
                continue
            s_cau = df_q_sub[st_id].dropna()
            if len(s_cau) < 200 or s_cau.mean() < 0.05:
                mean_val = round(float(s_cau.mean()), 3) if len(s_cau) > 0 else 0.0
                log_master(f"[{idx+1}/{total_estaciones}] ⏩ Filtro Caudal: Station {st_id} ({nombre_raw}) es cauce seco/inactivo (Media {mean_val} m³/s). Omitiendo al instante.")
                fallidos += 1
                continue

        log_master(f"\n[{idx+1}/{total_estaciones}] 🌊 Procesando Estación ID {st_id}: '{nombre_raw}' ({region_nombre})")

        try:
            # ====================================================================
            # [PASO 1] EXTRACCIÓN Y CREACIÓN DE DATASET
            # Definir un área geográfica (Bounding Box) alrededor de la estación 
            # y calcular automáticamente un punto virtual en la Cordillera de los Andes (+0.3 lat/lon)
            # ====================================================================
            area_bbox = [lat + 0.3, lon - 0.4, lat - 0.3, lon + 0.4]
            cordillera_lon = lon + 0.3

            if 'procesar_cuenca_desatendido' in globals():
                # Llama al Módulo 1 (ETL) que descarga Open-Meteo y cruza con CAMELS
                csv_generado = procesar_cuenca_desatendido(
                    nombre_cuenca=f"{region_nombre} - {nombre_raw}",
                    id_estacion=st_id,
                    area_bbox=area_bbox,
                    cordillera_lon_limit=cordillera_lon,
                    archivo_salida_csv=ruta_csv_salida
                )
            else:
                log_master("  ⚠️ procesar_cuenca_desatendido.py no está presente en este entorno. Se omite.")
                fallidos += 1
                continue

            if not os.path.exists(ruta_csv_salida) or os.path.getsize(ruta_csv_salida) < 1000:
                log_master(f"  ⚠️ No se pudieron generar datos suficientes para {st_id}. Continuando con la siguiente.")
                fallidos += 1
                continue

            # ====================================================================
            # [PASO 2] AUDITORÍA DE CALIDAD (QA)
            # Llama al Módulo de Validación para asegurar que los datos no tengan nulos,
            # tengan suficientes años de historia y el caudal varíe físicamente.
            # ====================================================================
            if 'validar_dataset' in globals():
                if not validar_dataset(ruta_csv_salida, f"{region_nombre} - {nombre_raw}"):
                    log_master(f"  ⚠️ Dataset de estación {st_id} no superó la validación. Omitiendo entrenamiento.")
                    fallidos += 1
                    continue

            # ====================================================================
            # [PASO 3] ENTRENAMIENTO DE IA EN GPU (XGBoost)
            # Mandamos el CSV perfecto a entrenar. Se crean dos modelos por separado:
            # Uno que predice a 24 horas y otro que predice a 48 horas.
            # ====================================================================
            log_master(f"  🤖 Entrenando Modelos de Predicción (24h y 48h)...")
            if 'entrenar_modelo_cuenca' in globals():
                ok_24 = entrenar_modelo_cuenca(ruta_csv_salida, horizonte_horas=24, modo_pronostico="pronostico_con_ruido")
                ok_48 = entrenar_modelo_cuenca(ruta_csv_salida, horizonte_horas=48, modo_pronostico="pronostico_con_ruido")

                if ok_24 and ok_48:
                    log_master(f"  ✅ Estación ID {st_id} ({nombre_raw}) COMPLETADA CON ÉXITO Y CONECTADA A SIPI AI.")
                    exitosos += 1
                else:
                    log_master(f"  ⚠️ Entrenamiento parcial para {st_id}.")
                    fallidos += 1
            else:
                log_master("  ⚠️ entrenar_modelo_gpu.py no está presente. Simulación finalizada para este nodo.")

            # Pausa ligera de cortesía (Rate Limiting) para evitar saturar la API Satelital
            time.sleep(1.2)

        except Exception as e:
            # [TOLERANCIA A FALLOS] Si una estación falla, el sistema lo registra y sigue con la siguiente sin colapsar.
            log_master(f"  ❌ Error inesperado procesando estación {st_id} ({nombre_raw}): {e}")
            log_master(traceback.format_exc())
            fallidos += 1
            time.sleep(2.0)

    t_total = (time.time() - t_inicio_total) / 60.0

    log_master("\n" + "=" * 90)
    log_master(f"🎉 EJECUCIÓN NACIONAL COMPLETA EN {t_total:.2f} MINUTOS")
    log_master(f"   Estaciones Procesadas Exitosamente : {exitosos}")
    log_master(f"   Estaciones Previamente Listas    : {omitidos}")
    log_master(f"   Estaciones Omitidas por Faltantes : {fallidos}")
    log_master(f"   Total Cobertura Nacional          : {exitosos + omitidos} / {total_estaciones} estaciones")
    log_master("=" * 90)

    # ====================================================================
    # [PASO 4] EVALUACIÓN GENERAL DE RENDIMIENTO
    # Al terminar todo el país, llamamos al módulo de métricas para
    # generar el archivo Markdown con el RMSE y el NSE a nivel nacional.
    # ====================================================================
    log_master("\n📊 INICIANDO AUDITORÍA AUTOMÁTICA Y EVALUACIÓN EN DATOS NO VISTOS (TEST DATA)...")
    try:
        from evaluar_rendimiento_nacional import evaluar_modelos_nacionales
        evaluar_modelos_nacionales()
        log_master("✅ AUDITORÍA DE DATOS Y EVALUACIÓN FINAL DE RENDIMIENTO COMPLETADA CON ÉXITO.")
    except Exception as e_eval:
        log_master(f"⚠️ Error durante la evaluación final: {e_eval}")

if __name__ == "__main__":
    procesar_todas_las_estaciones_chile()

