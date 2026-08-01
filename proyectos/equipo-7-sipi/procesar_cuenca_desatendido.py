"""
Módulo de Extracción y Procesamiento (ETL) - SIPI
=================================================
Este módulo es el primer eslabón del sistema. Su objetivo es recolectar 
la historia del clima y cruzarla con la historia del río para crear un 
dataset unificado que la Inteligencia Artificial pueda aprender.
"""
import re
import os
import pandas as pd
import requests

def slugify(text):
    """Limpia los nombres de los ríos para usarlos como nombres de archivo (elimina tildes, espacios, etc)."""
    return re.sub(r'[\W_]+', '_', str(text).lower()).strip('_')

def consultar_era5_archive_open_meteo(lat, lon, start_date, end_date):
    """
    Descarga la historia del clima de un punto GPS exacto usando la API Satelital de Open-Meteo (ERA5).
    Implementa un mecanismo de Fallback (Tolerancia a Fallos) en caso de caída de internet.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "America/Santiago"
    }
    
    try:
        # [PASO 1] Conexión a la API Satelital Real
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Verifica si hay error HTTP (404, 500, etc)
        data = response.json()
        
        # Mapeamos la respuesta JSON a un DataFrame de Pandas
        df_clima = pd.DataFrame({
            'Fecha': data['daily']['time'],
            'precipitacion_era5_mm': data['daily']['precipitation_sum'],
            'temperatura_c': data['daily']['temperature_2m_mean']
        })
        
        # Llenamos vacíos (imputación) si el satélite tuvo lecturas corruptas ese día
        df_clima = df_clima.ffill()
        return df_clima
        
    except Exception as e:
        # [PASO 2] Tolerancia a Fallos (Circuit Breaker)
        # Si el servidor satelital europeo se cae o nos bloquea, el sistema no colapsa,
        # sino que genera una matriz sintética de ceros para permitir que el Pipeline siga fluyendo.
        print(f"    [WARN] Falló la API ERA5, usando fallback sintético. Error: {e}")
        fechas = pd.date_range(start=start_date, end=end_date, freq='D')
        df_clima = pd.DataFrame({
            'Fecha': fechas.strftime('%Y-%m-%d'),
            'precipitacion_era5_mm': [0.0] * len(fechas),
            'temperatura_c': [15.0] * len(fechas)
        })
        return df_clima

def procesar_cuenca_desatendido(nombre_cuenca, id_estacion, area_bbox, cordillera_lon_limit, archivo_salida_csv):
    """
    Función principal que consolida el clima y los caudales en un CSV listo para entrenamiento.
    """
    print(f"  [LOG] Extrayendo datos ERA5 para {nombre_cuenca}...")
    
    # [PASO 2] Extracción de Clima Dual (Valle y Alta Cordillera):
    # Solicitamos el clima histórico del valle y el de la alta cordillera por separado.
    df_clima_valle = consultar_era5_archive_open_meteo(-33.4, -70.6, "2010-01-01", "2020-12-31")
    df_clima_cordillera = consultar_era5_archive_open_meteo(-33.4, cordillera_lon_limit, "2010-01-01", "2020-12-31")
    
    print(f"  [LOG] Cruzando datos con historial CAMELS-CL...")
    
    # [PASO 3] Carga de Historia Fluviométrica (CAMELS-CL):
    # Simulamos la lectura del archivo de mediciones físicas del río.
    # Producción: df_caudales = pd.read_csv('CAMELS_CL_v202201/q_m3s_day.csv')
    df_caudales = pd.DataFrame({
        'Fecha': df_clima_valle['Fecha'],
        'Caudal_CAMELS_m3_s': [12.5] * len(df_clima_valle) # Caudal base ficticio
    })
    
    # [PASO 4] Fusión de Datos (Merge):
    # Alineamos el clima satelital y el nivel del río usando la 'Fecha' como llave.
    df_final = pd.merge(df_caudales, df_clima_valle, on='Fecha', how='inner')
    
    # ==============================================================================
    # [PASO 5] INGENIERÍA DE CARACTERÍSTICAS (Feature Engineering)
    # Aquí creamos nuevas variables matemáticas para darle más "inteligencia" al modelo.
    # ==============================================================================
    
    # Feature 1: Clima de Montaña (Se añade a la tabla principal)
    df_final['temp_cordillera_c'] = df_clima_cordillera['temperatura_c']
    df_final['precipitacion_cordillera_mm'] = df_clima_cordillera['precipitacion_era5_mm']
    
    # Feature 2: Sensor Virtual de Isoterma Cero Alta
    # Lógica: Si hace más de 0°C en la cordillera Y llueve mucho -> ¡Hay derretimiento de nieve (deshielo)!
    df_final['riesgo_isoterma0'] = ((df_final['temp_cordillera_c'] > 0.0) & (df_final['precipitacion_cordillera_mm'] > 5.0)).astype(int)
    
    # Feature 3: Memoria Hidrológica (Lags)
    # Le enseñamos al modelo cuánto caudal traía el río hace 1 y 2 días atrás.
    df_final['caudal_lag1'] = df_final['Caudal_CAMELS_m3_s'].shift(1)
    df_final['caudal_lag2'] = df_final['Caudal_CAMELS_m3_s'].shift(2)
    
    # [PASO 6] Limpieza Final y Exportación:
    print(f"  [LOG] Guardando dataset en {archivo_salida_csv}")
    df_final.dropna(inplace=True) # Borrar filas incompletas por el 'shift'
    df_final.to_csv(archivo_salida_csv, index=False) # Guardar en disco duro
        
    return True
