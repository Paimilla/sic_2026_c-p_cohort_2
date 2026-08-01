"""
Módulo Backend de Predicción (Cron Job) - SIPI
Este script es el corazón del sistema. Se ejecuta en segundo plano, descarga
datos meteorológicos de los satélites (Open-Meteo), realiza el Feature Engineering
y utiliza los modelos de Machine Learning (.pkl) para generar la caché de predicciones.
"""
import os
import glob
import json
import pickle
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Suprimir advertencias molestas de Pandas
warnings.filterwarnings('ignore')

# Definición de rutas base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMELS_DIR = os.path.join(BASE_DIR, "CAMELS_CL_v202201")
CACHE_FILE = os.path.join(BASE_DIR, "cache_predicciones.pkl")
DATOS_DIR = os.path.join(BASE_DIR, "datos_entrenamiento")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos_entrenados")

def determinar_region(lat):
    """Asigna una Macro-Región basándose en la latitud GPS de la estación."""
    if lat > -21.5: return "Arica y Parinacota y Tarapacá"
    elif lat > -26.0: return "Antofagasta"
    elif lat > -29.0: return "Atacama"
    elif lat > -32.0: return "Coquimbo"
    elif lat > -33.8: return "Valparaíso y RM"
    elif lat > -35.0: return "O'Higgins y Maule"
    elif lat > -38.5: return "Ñuble, Biobío y Araucanía"
    elif lat > -41.5: return "Los Ríos y Los Lagos"
    else: return "Aysén y Magallanes"

def calcular_alerta(caudal_pred, umbral_p95):
    """
    Calcula el nivel de alerta basándose en el caudal predicho y el Percentil 95 histórico.
    Retorna: Categoría de color, Título, Código Hexadecimal, Clase CSS, Valor del termómetro
    """
    # Filtro de Sentido Común: Caudales muy pequeños (< 5 m3/s) no son inundación, 
    # incluso si superan el P95 de un río hiperárido (como Copiapó).
    if caudal_pred < 5.0:
        return "Verde", "Normal", "#34d399", "alert-verde", 12

    u_rojo = umbral_p95 * 1.60     # Peligro Catastrófico (>160%)
    u_naranja = umbral_p95 * 1.30  # Alerta Inundación (>130%)
    u_azul = umbral_p95 * 0.60     # Flujo Elevado (>60%)
    
    if caudal_pred >= u_rojo: return "Roja", "Inundación Severa", "#f87171", "alert-roja", 92
    elif caudal_pred >= u_naranja: return "Naranja", "Crecida Mayor", "#fb923c", "alert-naranja", 75
    elif caudal_pred >= umbral_p95: return "Amarilla", "Crecida Moderada P95", "#fbbf24", "alert-amarilla", 58
    elif caudal_pred >= u_azul: return "Azul", "Flujo Elevado", "#60a5fa", "alert-azul", 38
    else: return "Verde", "Normal", "#34d399", "alert-verde", 12

def cargar_inventario_estaciones():
    """
    Escanea la carpeta de datos de entrenamiento y modelos para deducir 
    qué cuencas están listas para predecirse, emparejándolas con los atributos de CAMELS.
    """
    ruta_attr = os.path.join(CAMELS_DIR, "catchment_attributes.csv")
    df_attr = pd.read_csv(ruta_attr) if os.path.exists(ruta_attr) else pd.DataFrame()
    if not df_attr.empty and 'gauge_id' in df_attr.columns:
        df_attr['gauge_id_clean'] = df_attr['gauge_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    else:
        df_attr['gauge_id_clean'] = pd.Series(dtype=str)

    # Buscar todos los CSVs en la nueva carpeta 'datos_entrenamiento'
    archivos_csv = sorted(glob.glob(os.path.join(DATOS_DIR, "datos_entrenamiento_*.csv")))
    estaciones_raw = {}

    for f_csv in archivos_csv:
        nombre_base = os.path.splitext(os.path.basename(f_csv))[0]
        # Extraer el ID de 7 dígitos de la estación desde el nombre del archivo
        st_id = next((p for p in nombre_base.split('_') if p.isdigit() and len(p) >= 7), None)
        if not st_id: continue

        # Construir las rutas hacia la nueva carpeta de 'modelos_entrenados'
        m_reg_24 = os.path.join(MODELOS_DIR, f"modelo_caudal_{nombre_base}_24h.pkl")
        m_clf_24 = os.path.join(MODELOS_DIR, f"modelo_inundacion_{nombre_base}_24h.pkl")
        j_feat_24 = os.path.join(MODELOS_DIR, f"features_{nombre_base}_24h.json")
        
        m_reg_48 = os.path.join(MODELOS_DIR, f"modelo_caudal_{nombre_base}_48h.pkl")
        m_clf_48 = os.path.join(MODELOS_DIR, f"modelo_inundacion_{nombre_base}_48h.pkl")
        j_feat_48 = os.path.join(MODELOS_DIR, f"features_{nombre_base}_48h.json")

        # Verificar que los modelos existan, si no, omitir la estación
        if not (os.path.exists(m_reg_24) and os.path.exists(m_clf_24)): continue

        # Extraer el valor NSE (Nash-Sutcliffe) guardado durante el entrenamiento
        nse_val = None
        if os.path.exists(j_feat_24):
            try:
                with open(j_feat_24, 'r', encoding='utf-8') as f:
                    nse_val = json.load(f).get('nse')
            except: pass

        # Cruzar con atributos de CAMELS para sacar Lat, Lon y Nombre Real
        row = df_attr[df_attr['gauge_id_clean'] == str(st_id).strip()] if not df_attr.empty else pd.DataFrame()
        if not row.empty:
            a = row.iloc[0]
            nombre, area, lat, lon = str(a.get('gauge_name', nombre_base)), float(a.get('area_km2', 1000)), float(a.get('gauge_lat', -32.0)), float(a.get('gauge_lon', -70.5))
        else:
            nombre, area, lat, lon = nombre_base.replace('datos_entrenamiento_', '').replace('_', ' ').title(), 1200, -32.5, -70.5

        entry = dict(
            id=st_id, nombre=nombre, csv=f_csv, region=determinar_region(lat),
            area_km2=area, lat=lat, lon=lon,
            reg24=m_reg_24, clf24=m_clf_24, feat24=j_feat_24,
            reg48=m_reg_48, clf48=m_clf_48, feat48=j_feat_48, nse=nse_val,
            _key=f"{nombre} ({st_id})"
        )

        # Deduplicación: Si hay dos modelos para la misma estación, quedarse con el de mejor NSE
        if st_id in estaciones_raw:
            if (nse_val or -999) > (estaciones_raw[st_id].get('nse') or -999):
                estaciones_raw[st_id] = entry
        else:
            estaciones_raw[st_id] = entry

    return {entry.pop('_key'): entry for entry in estaciones_raw.values()}

def consultar_open_meteo_forecast(lat, lon, lat_cord, lon_cord):
    """
    Consulta simultánea a los satélites de Open-Meteo para el Valle y la Alta Cordillera.
    Posee lógica de re-intento (Exponential Backoff) en caso de superar los límites de la API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    vars = "temperature_2m,dew_point_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,shortwave_radiation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm,snow_depth,evapotranspiration"
    # Petición a dos puntos GPS separados por coma (Valle y Montaña)
    params = {"latitude": f"{lat},{lat_cord}", "longitude": f"{lon},{lon_cord}", "hourly": vars, "windspeed_unit": "ms", "timezone": "UTC", "past_days": 7, "forecast_days": 3}
    
    for attempt in range(3):
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429: time.sleep(2 ** attempt); continue # Limite de peticiones API
            res.raise_for_status()
            data = res.json()
            if isinstance(data, list) and len(data) >= 2: return data[0].get("hourly"), data[1].get("hourly")
            elif isinstance(data, dict) and "hourly" in data: return data["hourly"], data["hourly"]
        except: time.sleep(1.5 ** attempt)
    return None, None

def aplicar_feature_engineering(df, shift=-1):
    """
    Genera las variables matemáticas complejas (Lags, Acumulados, Isoterma Cero).
    Esta función emula exactamente cómo se trataron los datos en la fase de entrenamiento.
    """
    cols_clima = ['precipitacion_era5_mm', 'precipitacion_zona_alta_mm', 'temperatura_c', 'temperatura_zona_alta_c', 'deshielo_zona_alta_mm', 'radiacion_solar_j_m2']
    # 1. Crear variables de 'pronóstico' (El futuro, desfasando hacia atrás)
    for col in cols_clima:
        if col in df.columns: df[f'{col}_pronostico'] = df[col].shift(shift)

    # 2. Sensor Virtual de Isoterma Cero Alta (Si llueve y hace calor en la montaña)
    if 'temperatura_zona_alta_c_pronostico' in df.columns and 'precipitacion_zona_alta_mm_pronostico' in df.columns:
        df['riesgo_isoterma0_alta_pronostico'] = ((df['temperatura_zona_alta_c_pronostico'] > 0.0) & (df['precipitacion_zona_alta_mm_pronostico'] > 2.0)).astype(int)

    excl = {'Fecha', 'Caudal_CAMELS_m3_s', 'caudal_futuro', 'inundacion_futura', 'caudal_hoy'}
    cols_lag = [c for c in df.columns if c not in excl and not c.endswith('_pronostico')]
    
    # 3. Lags (Memoria Hidrológica) de 1 a 3 días
    for lag in range(1, 4):
        for col in cols_lag: df[f'{col}_lag{lag}'] = df[col].shift(lag)
    for lag in range(1, 4): df[f'caudal_lag{lag}'] = df['Caudal_CAMELS_m3_s'].shift(lag) if 'Caudal_CAMELS_m3_s' in df.columns else df['caudal_hoy']

    # 4. Acumuladores de Saturación (Lluvia de la última semana)
    if 'precipitacion_era5_mm' in df.columns:
        df['precipitacion_acum_3d'] = df['precipitacion_era5_mm'].rolling(3).sum()
        df['precipitacion_acum_7d'] = df['precipitacion_era5_mm'].rolling(7).sum()
    if 'deshielo_zona_alta_mm' in df.columns: df['deshielo_acum_3d'] = df['deshielo_zona_alta_mm'].rolling(3).sum()
    return df

def predecir_estacion(info, horizonte=24):
    """
    Función orquestadora por estación. 
    Descarga el clima, calcula features, y aplica los Modelos de IA (.pkl).
    """
    df = pd.read_csv(info["csv"])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)

    # Cargar modelos de Inteligencia Artificial (Cerebro Híbrido)
    with open(info[f"reg{horizonte}"], 'rb') as f: reg = pickle.load(f)
    with open(info[f"clf{horizonte}"], 'rb') as f: clf = pickle.load(f)
    with open(info[f"feat{horizonte}"], 'r', encoding='utf-8') as f: fcfg = json.load(f)

    features, umbral_p95 = fcfg['features'], fcfg.get('umbral_p95', float(df['Caudal_CAMELS_m3_s'].quantile(0.95)))
    
    # Lógica de Mediana Estacional basada en el mes actual (Real)
    mes_actual_real = datetime.now().month
    mediana_val = float(df[df['Fecha'].dt.month == mes_actual_real]['Caudal_CAMELS_m3_s'].median())
    if pd.isna(mediana_val):
        mediana_val = float(df['Caudal_CAMELS_m3_s'].median())
    
    shift = -1 if horizonte == 24 else -2
    ultimo_dato_fecha = df['Fecha'].iloc[-1].strftime('%Y-%m-%d')
    
    # Consultar Sensor Valle + Sensor Virtual Cordillera (+0.35 lon)
    h_c, h_k = consultar_open_meteo_forecast(info['lat'], info['lon'], info['lat'], info['lon'] + 0.35)

    if h_c:
        try:
            # Procesamiento de JSON a DataFrame (Valle)
            df_h = pd.DataFrame(h_c); df_h["time"] = pd.to_datetime(df_h["time"]); df_h["Fecha"] = df_h["time"].dt.date
            
            # Descomposición Vectorial del Viento
            ws, wdir_rad = df_h["wind_speed_10m"], np.radians(df_h["wind_direction_10m"])
            df_h["viento_u_m_s"], df_h["viento_v_m_s"] = -1.0 * ws * np.sin(wdir_rad), -1.0 * ws * np.cos(wdir_rad)
            df_h["radiacion_solar_j_m2"], df_h["presion_superficial_pa"] = df_h["shortwave_radiation"] * 3600.0, df_h["surface_pressure"] * 100.0
            
            # Agrupación horaria a promedios/sumas diarias
            cols_prom = {"temperature_2m": "temperatura_c", "dew_point_2m": "punto_rocio_c", "relative_humidity_2m": "humedad_relativa_porcentaje", "presion_superficial_pa": "presion_superficial_pa", "viento_u_m_s": "viento_u_m_s", "viento_v_m_s": "viento_v_m_s", "soil_moisture_0_to_7cm": "humedad_suelo_capa1_vol", "soil_moisture_7_to_28cm": "humedad_suelo_capa2_vol", "soil_moisture_28_to_100cm": "humedad_suelo_capa3_vol", "soil_moisture_100_to_255cm": "humedad_suelo_capa4_vol"}
            df_d_real = pd.merge(df_h.groupby("Fecha")[list(cols_prom.keys())].mean().rename(columns=cols_prom).reset_index(), df_h.groupby("Fecha")[["precipitation", "radiacion_solar_j_m2", "evapotranspiration"]].sum().rename(columns={"precipitation": "precipitacion_era5_mm", "evapotranspiration": "evapotranspiracion_mm"}).reset_index(), on="Fecha")
            df_d_real["swe_mm"] = df_h.groupby("Fecha")["snow_depth"].mean().values * 200.0
            df_d_real["cobertura_nieve_porcentaje"] = (df_h.groupby("Fecha")["snow_depth"].mean().values > 0.01).astype(float) * 100.0
            df_d_real["deshielo_mm"] = 0.0

            # Procesamiento DataFrame (Cordillera)
            if h_k:
                df_hk = pd.DataFrame(h_k); df_hk["time"] = pd.to_datetime(df_hk["time"]); df_hk["Fecha"] = df_hk["time"].dt.date
                df_k_d = pd.merge(df_hk.groupby("Fecha")[["temperature_2m", "snow_depth"]].mean().reset_index(), df_hk.groupby("Fecha")[["precipitation"]].sum().reset_index(), on="Fecha").rename(columns={"temperature_2m": "temperatura_zona_alta_c", "precipitation": "precipitacion_zona_alta_mm"})
                df_k_d["swe_zona_alta_mm"], df_k_d["deshielo_zona_alta_mm"] = df_k_d["snow_depth"] * 250.0, 0.0
                df_d_real = pd.merge(df_d_real, df_k_d[["Fecha", "temperatura_zona_alta_c", "precipitacion_zona_alta_mm", "swe_zona_alta_mm", "deshielo_zona_alta_mm"]], on="Fecha", how="left")
            else:
                df_d_real["temperatura_zona_alta_c"], df_d_real["precipitacion_zona_alta_mm"], df_d_real["swe_zona_alta_mm"], df_d_real["deshielo_zona_alta_mm"] = df_d_real["temperatura_c"], df_d_real["precipitacion_era5_mm"], df_d_real["swe_mm"], 0.0

            for col in ["temperatura_zona_alta_c", "precipitacion_zona_alta_mm", "swe_zona_alta_mm", "deshielo_zona_alta_mm"]:
                fb = col.replace("_zona_alta", "")
                df_d_real[col] = df_d_real[col].fillna(df_d_real[fb]) if fb in df_d_real.columns else df_d_real[col].fillna(0.0)

            # Preparación del DataFrame para predicción con XGBoost (Usamos la mediana mensual como línea base estable)
            val_q_hoy = mediana_val
            df_d_real["Caudal_CAMELS_m3_s"] = df_d_real["caudal_hoy"] = val_q_hoy
            df_clean = aplicar_feature_engineering(df_d_real, shift).dropna().reset_index(drop=True)
            for c in features:
                if c not in df_clean.columns: df_clean[c] = 0.0
            
            X = df_clean[features]
            # [PREDICCIÓN DUAL] Regressor -> Caudal (m3/s) | Classifier -> Riesgo de Desborde (%)
            df_clean['pred_caudal'], df_clean['pred_prob'] = reg.predict(X), clf.predict_proba(X)[:, 1]

            # --- FILTRO ANTI-ALUCINACIONES (SHAP SENSE-CHECK) ---
            for idx in df_clean.index:
                p_c = df_clean.at[idx, 'pred_caudal']
                lluvia = df_clean.at[idx, 'precipitacion_acum_3d'] if 'precipitacion_acum_3d' in df_clean.columns else 0
                
                # Si predice cualquier nivel de alerta (> 0.6 * p95) pero la lluvia acumulada es casi nula (falsa crecida)
                if p_c > (umbral_p95 * 0.6) and lluvia < 15.0:
                    df_clean.at[idx, 'pred_caudal'] = max(val_q_hoy * 1.05, umbral_p95 * 0.1) # Forzar calma
                    df_clean.at[idx, 'pred_prob'] = 0.01 # Forzar probabilidad a 1%
            # ----------------------------------------------------

            obs, pred, prob = val_q_hoy, float(df_clean.iloc[-1]['pred_caudal']), float(df_clean.iloc[-1]['pred_prob'])
            prev = float(df_clean.iloc[-2]['pred_caudal']) if len(df_clean) > 1 else obs
            delta_obs, delta_pred = 0.0, round(pred - prev, 2)
            nivel, nota, color_hex, css_class, gauge = calcular_alerta(pred, umbral_p95)

            df_clean['Fecha_Objetivo'] = pd.to_datetime(df_clean['Fecha']) + pd.to_timedelta(-shift, unit='d')
            serie_df = df_clean[['Fecha_Objetivo', 'pred_caudal', 'pred_prob']].rename(columns={'Fecha_Objetivo': 'Fecha', 'pred_caudal': 'Predicho'}).assign(Observado=val_q_hoy)
            forecast_df = df_d_real[['Fecha', 'precipitacion_era5_mm', 'temperatura_c']].rename(columns={'precipitacion_era5_mm': 'precipitacion_mm'})

            # Retorno del diccionario completo a la memoria caché
            return dict(serie=serie_df, obs=round(obs, 2), pred=round(pred, 2), delta_obs=delta_obs, delta_pred=delta_pred, prob=round(prob*100,1), nivel=nivel, nota=nota, color=color_hex, css=css_class, gauge=gauge, p95=round(umbral_p95, 2), mediana=round(mediana_val, 2), nse=fcfg.get('nse'), r2=fcfg.get('r2'), en_vivo=True, ultimo_dato=ultimo_dato_fecha, forecast=forecast_df)
        except: pass

    # Fallback histórico (Si falla Open-Meteo)
    df['caudal_futuro'], df['inundacion_futura'], df['caudal_hoy'] = df['Caudal_CAMELS_m3_s'].shift(shift), (df['Caudal_CAMELS_m3_s'].shift(shift) >= umbral_p95).astype(int), df['Caudal_CAMELS_m3_s']
    df_c = aplicar_feature_engineering(df, shift).dropna().reset_index(drop=True)
    for c in features:
        if c not in df_c.columns: df_c[c] = 0.0
    
    X = df_c[features]
    df_c['pred_caudal'], df_c['pred_prob'] = reg.predict(X), clf.predict_proba(X)[:, 1]
    
    obs, pred, prob = float(df_c.iloc[-1]['Caudal_CAMELS_m3_s']), float(df_c.iloc[-1]['pred_caudal']), float(df_c.iloc[-1]['pred_prob'])
    prev = float(df_c.iloc[-2]['Caudal_CAMELS_m3_s']) if len(df_c) > 1 else obs
    nivel, nota, color_hex, css_class, gauge = calcular_alerta(pred, umbral_p95)
    
    df_c['Fecha_Objetivo'] = pd.to_datetime(df_c['Fecha']) + pd.to_timedelta(-shift, unit='d')
    serie_c = df_c[['Fecha_Objetivo', 'Caudal_CAMELS_m3_s', 'pred_caudal', 'pred_prob']].rename(columns={'Fecha_Objetivo': 'Fecha', 'Caudal_CAMELS_m3_s': 'Observado', 'pred_caudal': 'Predicho'})
    
    return dict(serie=serie_c, obs=round(obs, 2), pred=round(pred, 2), delta_obs=round(obs-prev,2), delta_pred=round(pred-obs,2), prob=round(prob*100,1), nivel=nivel, nota=nota, color=color_hex, css=css_class, gauge=gauge, p95=round(umbral_p95, 2), mediana=round(mediana_val, 2), nse=fcfg.get('nse'), r2=fcfg.get('r2'), en_vivo=False, ultimo_dato=ultimo_dato_fecha, forecast=None)

def generar_cache():
    """
    [EL CRON JOB MAESTRO]
    Función Orquestadora que se ejecuta automáticamente cada 12 horas.
    Recorre las 392 cuencas de Chile, descarga el clima de los satélites en tiempo real,
    le pide a la IA que prediga el futuro, y guarda todo comprimido en un archivo
    'cache_predicciones.pkl' para que el Frontend (App) lo lea al instante.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando pre-cálculo para caché global...")
    estaciones = cargar_inventario_estaciones()
    print(f"Inventario cargado: {len(estaciones)} estaciones únicas deduplicadas.")

    cache = {"metadata": {"fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, "estaciones": estaciones, "datos": {}}
    
    total = len(estaciones)
    for i, (k, info) in enumerate(estaciones.items(), 1):
        print(f"Procesando {i}/{total}: {k}...")
        try:
            r24 = predecir_estacion(info, 24)
            r48 = predecir_estacion(info, 48)
            cache["datos"][k] = {"24": r24, "48": r48}
        except Exception as e:
            print(f" Error en {k}: {e}")
            cache["datos"][k] = None
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Caché guardado exitosamente en: {CACHE_FILE}")

if __name__ == "__main__":
    generar_cache()
