"""
Módulo de Visualización (Frontend) - SIPI
=========================================
[EL PANEL DE CONTROL]
Este script es la interfaz gráfica donde el usuario humano toma decisiones. 
Su diseño es "Ultra-Ligero": No procesa matemáticas ni descarga nada de internet. 
Su único trabajo es leer el archivo 'cache_predicciones.pkl' desde la memoria RAM 
y dibujar un mapa nacional en tiempo real (casi 0 segundos de latencia).
"""
import os
import time
import pickle
import json
import pandas as pd
from datetime import datetime, timedelta
import dga_api
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta al archivo serializado (El "Cerebro" de la App)
CACHE_FILE = os.path.join(BASE_DIR, "cache_predicciones.pkl")

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN STREAMLIT
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SIPI AI · Monitoreo Hidrológico Nacional",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark = st.session_state["dark_mode"]

# ═══════════════════════════════════════════════════════════════
#  CSS PREMIUM
# ═══════════════════════════════════════════════════════════════
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)

css_vars = """
:root {
  --bg:            #0f1117; --bg-elevated:   #1a1c25; --bg-subtle:     #21242d; --bg-glass:      rgba(26,28,37,0.75);
  --border:        #2d3040; --border-soft:   #262938; --text:          #e8eaed; --text-muted:    #9aa0a6; --text-dim:      #6b7280;
  --accent:        #4da6ff; --accent-soft:   rgba(77,166,255,0.12); --water2:        #12b5cb;
  --green:         #34d399; --green-soft:    rgba(52,211,153,0.12);
  --blue:          #60a5fa; --blue-soft:     rgba(96,165,250,0.12);
  --yellow:        #fbbf24; --yellow-soft:   rgba(251,191,36,0.14);
  --orange:        #fb923c; --orange-soft:   rgba(251,146,60,0.14);
  --red:           #f87171; --red-soft:      rgba(248,113,113,0.14);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.3); --shadow-md: 0 4px 12px rgba(0,0,0,0.25); --shadow-lg: 0 8px 24px rgba(0,0,0,0.3);
  --radius-lg: 18px; --radius-md: 12px; --radius-sm: 8px;
  --plot-bg: #1a1c25; --grid-color: rgba(255,255,255,0.06); --map-style: carto-darkmatter;
}
""" if dark else """
:root {
  --bg:            #f4f6f8; --bg-elevated:   #ffffff; --bg-subtle:     #f1f3f4; --bg-glass:      rgba(255,255,255,0.82);
  --border:        #dde1e6; --border-soft:   #e8eaed; --text:          #1f2124; --text-muted:    #5f6368; --text-dim:      #80868b;
  --accent:        #1a73e8; --accent-soft:   rgba(26,115,232,0.08); --water2:        #12b5cb;
  --green:         #1e8e3e; --green-soft:    rgba(30,142,62,0.10);
  --blue:          #1a73e8; --blue-soft:     rgba(26,115,232,0.10);
  --yellow:        #f9ab00; --yellow-soft:   rgba(249,171,0,0.14);
  --orange:        #e8710a; --orange-soft:   rgba(232,113,10,0.12);
  --red:           #d93025; --red-soft:      rgba(217,48,37,0.10);
  --shadow-sm: 0 1px 2px rgba(60,64,67,0.16), 0 1px 3px rgba(60,64,67,0.08); --shadow-md: 0 2px 6px rgba(60,64,67,0.10), 0 4px 14px rgba(60,64,67,0.08); --shadow-lg: 0 6px 18px rgba(60,64,67,0.10), 0 14px 34px rgba(60,64,67,0.12);
  --radius-lg: 18px; --radius-md: 12px; --radius-sm: 8px;
  --plot-bg: #f8f9fa; --grid-color: rgba(60,64,67,0.08); --map-style: carto-positron;
}
"""
st.markdown(f"""<style>
{css_vars}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] {{
    background-color: var(--bg) !important; font-family: 'Inter', sans-serif !important; color: var(--text) !important;
}}
.block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1600px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--border-soft); }}
[data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] > div > div, [data-testid="stRadio"] {{ background-color: var(--bg-elevated) !important; border-radius: var(--radius-sm) !important; }}
.nav-bar {{ display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 1.3rem; background: var(--bg-glass); backdrop-filter: blur(16px); border: 1px solid var(--border-soft); border-radius: var(--radius-lg); margin-bottom: 0.8rem; box-shadow: var(--shadow-md); }}
.nav-logo {{ font-size: 20px; font-weight: 800; background: linear-gradient(135deg, var(--accent), var(--water2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.badge-count {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700; padding: 0.22rem 0.55rem; border-radius: 999px; display: inline-flex; align-items: center; gap: 3px; }}
.alert-badge {{ display: inline-block; padding: 0.28rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 11.5px; }}
.alert-verde {{ background: var(--green-soft); color: var(--green); border: 1px solid rgba(52,211,153,0.3); }}
.alert-azul {{ background: var(--blue-soft); color: var(--blue); border: 1px solid rgba(96,165,250,0.3); }}
.alert-amarilla {{ background: var(--yellow-soft); color: var(--yellow); border: 1px solid rgba(251,191,36,0.4); }}
.alert-naranja {{ background: var(--orange-soft); color: var(--orange); border: 1px solid rgba(251,146,60,0.35); }}
.alert-roja {{ background: var(--red-soft); color: var(--red); border: 1px solid rgba(248,113,113,0.35); }}
.kpi {{ background: var(--bg-glass); border: 1px solid var(--border-soft); border-radius: var(--radius-md); padding: 0.95rem 1.1rem; min-height: 90px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: var(--shadow-sm); }}
.kpi-label {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight:600; text-transform: uppercase; color: var(--text-dim); }}
.kpi-val {{ font-size: 22px; font-weight: 800; color: var(--text); margin-top: 3px; }}
.kpi-sub {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); margin-top: 3px; }}
.map-container {{ border:1px solid var(--border-soft); border-radius:var(--radius-lg); overflow:hidden; box-shadow:var(--shadow-lg); margin-bottom:1rem; }}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CARGA DE DATOS ULTRA-RÁPIDA (Memoria RAM)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def cargar_cache():
    """
    Lee el archivo binario pickle. Como se almacena con @st.cache_data, 
    permanece en la memoria RAM del servidor de Streamlit y no requiere
    leer el disco duro cada vez que el usuario mueve el mapa, logrando 60 FPS.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)

cache = cargar_cache()

if not cache:
    st.error("⚠️ El caché de predicciones no existe. Por favor ejecuta `python motor_prediccion.py` en una terminal para generarlo.")
    st.stop()

ESTACIONES = cache["estaciones"]
DATOS = cache["datos"]
TS = cache["metadata"]["fecha_generacion"]

# ═══════════════════════════════════════════════════════════════
#  BARRA SUPERIOR (NAVBAR & CONTROLES)
# ═══════════════════════════════════════════════════════════════
c_v = len(ESTACIONES) # Aproximado para conteo rápido
c_a = sum(1 for k in DATOS if DATOS[k] and DATOS[k]['24']['nivel'] == 'Azul')
c_am = sum(1 for k in DATOS if DATOS[k] and DATOS[k]['24']['nivel'] == 'Amarilla')
c_n = sum(1 for k in DATOS if DATOS[k] and DATOS[k]['24']['nivel'] == 'Naranja')
c_r = sum(1 for k in DATOS if DATOS[k] and DATOS[k]['24']['nivel'] == 'Roja')

st.markdown(f"""
<div class="nav-bar">
    <div class="nav-logo">🌊 SIPI AI</div>
    <div style="display:flex; gap:6px; align-items:center;">
        <span style="font-size:12px; font-weight:600; margin-right:10px;">Alertas (24h):</span>
        <span class="badge-count" style="background:var(--yellow-soft); color:var(--yellow); border:1px solid rgba(251,191,36,0.4);">🟡 {c_am}</span>
        <span class="badge-count" style="background:var(--orange-soft); color:var(--orange); border:1px solid rgba(251,146,60,0.35);">🟠 {c_n}</span>
        <span class="badge-count" style="background:var(--red-soft); color:var(--red); border:1px solid rgba(248,113,113,0.35);">🔴 {c_r}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 13px; padding: 10px 15px; background: var(--bg-glass); border-radius: 8px; border: 1px solid var(--border-soft); margin-bottom: 15px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; box-shadow: var(--shadow-sm);">
    <strong style="color: var(--text-dim); text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">🚦 Leyenda de Alertas:</strong>
    <span><span style="color: var(--green);">🟢</span> <b>Verde:</b> Normal</span>
    <span><span style="color: var(--blue);">🔵</span> <b>Azul:</b> Flujo Elevado (> 60% del P95)</span>
    <span><span style="color: var(--yellow);">🟡</span> <b>Amarilla:</b> Precaución (> 100% del P95)</span>
    <span><span style="color: var(--orange);">🟠</span> <b>Naranja:</b> Alerta Inundación (> 130% del P95)</span>
    <span><span style="color: var(--red);">🔴</span> <b>Roja:</b> Peligro Catastrófico (> 160% del P95)</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SINCRONIZACIÓN DE MAPA (Evitar errores de Session State)
# ═══════════════════════════════════════════════════════════════
map_state = st.session_state.get("mapa_chile")
if map_state and "selection" in map_state and map_state["selection"].get("points"):
    pts = map_state["selection"]["points"]
    if pts and "customdata" in pts[0]:
        clicked_id = pts[0]["customdata"][0]
        if st.session_state.get("sel_est") != clicked_id:
            st.session_state["sel_reg"] = "Todas las Regiones"
            st.session_state["sel_est"] = clicked_id

# ═══════════════════════════════════════════════════════════════
#  BARRA LATERAL (SIDEBAR) - HALLAZGO PRINCIPAL
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Sobre SIPI AI")
    st.markdown(
        "Este sistema es el resultado de nuestro análisis para responder: "
        "**¿Es posible predecir crecidas fluviales con 24-48 hrs de anticipación usando solo clima e historia hidrológica?**"
    )
    st.markdown("### 🏆 Hallazgo Principal")
    st.info(
        "¡Sí! Descubrimos que entrenando Inteligencia Artificial con datos satelitales (lluvia, nieve, temperatura) "
        "y el registro histórico de los ríos de Chile, podemos predecir su comportamiento futuro con un **81% de precisión (Excelente)**. "
        "Esto nos da **48 horas de ventaja vitales** para evacuar y salvar vidas antes de que ocurra una inundación."
    )
    st.markdown("---")
    st.markdown("👨‍💻 **Equipo 7:** Carolina, Felipe, Débora y Bastián.")

# Filtros globales superiores
f1, f2, f3, f4 = st.columns([1, 1.5, 2, 1])
with f1:
    horizonte = st.selectbox("Horizonte", ["24", "48"], format_func=lambda x: f"{x} Horas")
with f2:
    regiones_disponibles = ["Todas las Regiones"] + sorted(list(set(v['region'] for v in ESTACIONES.values())))
    # Usar session_state nativo para controlar la región
    sel_region = st.selectbox("Región", regiones_disponibles, key="sel_reg")
with f3:
    # Autocomplete nativo de Streamlit
    est_filtradas = list(ESTACIONES.keys())
    if sel_region != "Todas las Regiones":
        est_filtradas = [k for k in est_filtradas if ESTACIONES[k]['region'] == sel_region]
    
    # Asegurarnos de que el selectbox siempre tenga la estación seleccionada internamente
    # incluso si se cambió desde el mapa
    sel_estacion = st.selectbox("🔍 Buscar y seleccionar estación...", est_filtradas, key="sel_est")
with f4:
    mostrar_normales = st.checkbox("Mostrar estaciones normales", value=True, help="Oculta los puntos verdes para desaturar el mapa")

st.write("") # Espaciador

# ═══════════════════════════════════════════════════════════════
#  TABS PRINCIPALES Y LAYOUT
# ═══════════════════════════════════════════════════════════════
if True: # Mantenemos la indentación original
    col_mapa, col_datos = st.columns([1.5, 1.1])

    # ═══════════════════════════════════════════════════════════════
    #  COLUMNA IZQUIERDA: MAPA NACIONAL
    # ═══════════════════════════════════════════════════════════════
    with col_mapa:
        def generar_df_mapa():
            rows = []
            for k, info in ESTACIONES.items():
                if k not in DATOS or DATOS[k] is None: continue
                res = DATOS[k][horizonte]
                # Reducir tamaño masivo: Si es verde hacerlo pequeñito para evitar desorden visual
                if res['nivel'] == 'Verde':
                    if not mostrar_normales: continue
                    sz = 4
                else:
                    sz = {"Azul": 7, "Amarilla": 10, "Naranja": 14, "Roja": 18}.get(res['nivel'], 6)
                
                rows.append(dict(
                    key_estacion=k, estacion=info['nombre'], id=info['id'], region=info['region'],
                    lat=info['lat'], lon=info['lon'], obs=res['obs'], pred=res['pred'], prob=res['prob'],
                    alerta=res['nivel'], color=res['color'], size=sz
                ))
            return pd.DataFrame(rows)

        df_mapa = generar_df_mapa()

        df_m_disp = df_mapa
        if sel_region != "Todas las Regiones":
            df_m_disp = df_m_disp[df_m_disp['region'] == sel_region]

        # Ordenar para que rojas/naranjas se dibujen encima de las verdes (Z-Index algorítmico)
        orden = {'Verde': 1, 'Azul': 2, 'Amarilla': 3, 'Naranja': 4, 'Roja': 5}
        if not df_m_disp.empty:
            df_m_disp['peso'] = df_m_disp['alerta'].map(orden)
            df_m_disp = df_m_disp.sort_values('peso')

        # Renderizado interactivo usando Plotly Mapbox/ScatterMap
        fig_m = px.scatter_map(
            df_m_disp, lat="lat", lon="lon", color="alerta", size="size", size_max=16,
            map_style="carto-darkmatter" if dark else "carto-positron",
            color_discrete_map={"Verde": "#34d399", "Azul": "#60a5fa", "Amarilla": "#fbbf24", "Naranja": "#fb923c", "Roja": "#f87171"},
            hover_name="estacion", custom_data=["key_estacion"],
            hover_data={"lat": False, "lon": False, "key_estacion": False, "size": False, "peso": False},
            zoom=4.2 if sel_region == "Todas las Regiones" else 6.0, 
            center={"lat": -35.5, "lon": -71.0} if sel_region == "Todas las Regiones" else {"lat": ESTACIONES[sel_estacion]['lat'], "lon": ESTACIONES[sel_estacion]['lon']},
            height=720, # Mapa más alto en split screen
        )
        fig_m.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, # Leyenda oculta para más espacio
        )
        fig_m.update_traces(marker=dict(opacity=0.85))

        # El estado del mapa se guarda automaticamente en st.session_state["mapa_chile"]
        # y se sincroniza al inicio del script en la proxima recarga (rerun).
        map_event = st.plotly_chart(fig_m, use_container_width=True, on_select="rerun", selection_mode="points", key="mapa_chile")

    # ═══════════════════════════════════════════════════════════════
    #  COLUMNA DERECHA: DETALLE DE ESTACIÓN
    # ═══════════════════════════════════════════════════════════════
    with col_datos:
        info = ESTACIONES[sel_estacion]
        if DATOS.get(sel_estacion) is None:
            st.error("No hay datos calculados para esta estación.")
        else:
            res = DATOS[sel_estacion][horizonte]

            st.markdown(f"<h3 style='margin-top:0;'>📍 {info['nombre']}</h3>", unsafe_allow_html=True)
            st.caption(f"ID: {info['id']} · {info['region']}")
        
            # Grid 2x2 para KPIs
            k1, k2 = st.columns(2)
            with k1: st.markdown(f"""<div class="kpi"><div class="kpi-label">🛡️ Alerta</div><div class="kpi-val"><span class="alert-badge {res['css']}">{res['nivel']}</span></div><div class="kpi-sub">{res['nota']}</div></div>""", unsafe_allow_html=True)
        
            # Telemetria en vivo (DGA) vs Observado (Histórico)
            caudal_dga = dga_api.obtener_caudal_en_vivo(info['id'], res['obs'])
            if caudal_dga is not None:
                with k2: st.markdown(f"""<div class="kpi" style="border-color:rgba(239,68,68,0.5);"><div class="kpi-label">🔴 Caudal en VIVO (DGA)</div><div class="kpi-val">{caudal_dga} <span style="font-size:11px;color:var(--text-muted);">m³/s</span></div><div class="kpi-sub">Actualizado hace instantes</div></div>""", unsafe_allow_html=True)
            else:
                mediana_val = res.get('mediana', res['obs'])
                if isinstance(mediana_val, (int, float)): mediana_val = round(mediana_val, 2)
                with k2: st.markdown(f"""<div class="kpi"><div class="kpi-label">📊 Normalidad (Mediana Histórica)</div><div class="kpi-val">{mediana_val} <span style="font-size:11px;color:var(--text-muted);">m³/s</span></div><div class="kpi-sub">Promedio Típico del Mes</div></div>""", unsafe_allow_html=True)
        
            st.write("")
            k3, k4 = st.columns(2)
            with k3: st.markdown(f"""<div class="kpi" style="border-color:rgba(77,166,255,0.3);"><div class="kpi-label">📈 Predicción IA ({horizonte}h)</div><div class="kpi-val" style="color:var(--accent);">{res['pred']} <span style="font-size:11px;color:var(--text-muted);">m³/s</span></div><div class="kpi-sub">Modelo Predictivo Global</div></div>""", unsafe_allow_html=True)
            with k4: st.markdown(f"""<div class="kpi"><div class="kpi-label">⚡ Prob. de Crecida</div><div class="kpi-val" style="color:var(--yellow);">{res['prob']}%</div><div class="kpi-sub">Umbral P95: {res['p95']} m³/s</div></div>""", unsafe_allow_html=True)

            st.write("")
            # Gráficos verticales apilados
            df_s = res['serie']
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=df_s['Fecha'], y=df_s['Predicho'], name=f'Predicho {horizonte}h', line=dict(color='#4da6ff' if dark else '#1a73e8', width=2.5, dash='dash')))
            fig_ts.add_hline(y=res['p95'], line_dash="dot", line_color="#fb923c", annotation_text=f"P95 ({res['p95']})", annotation_font_color="#fb923c")
            fig_ts.update_layout(title="Caudal vs Tiempo", height=230, margin=dict(l=30, r=10, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="var(--plot-bg)" if not dark else "#1a1c25", font=dict(color="#e8eaed" if dark else "#1f2124", family="Inter", size=10), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
            st.plotly_chart(fig_ts, use_container_width=True)

            f_df = res.get('forecast')
            if f_df is not None:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=f_df['Fecha'], y=f_df['precipitacion_mm'], name='Lluvia (mm)', fill='tozeroy', fillcolor='rgba(139,92,246,0.15)', line=dict(color='#8b5cf6', width=2.5)))
                # No dibujamos el P95 aquí porque este gráfico es de lluvia (mm), no de caudal.
                fig_fc.update_layout(title="Proyección de Lluvia (Open-Meteo)", height=230, margin=dict(l=30, r=10, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1a1c25" if dark else "#f8f9fa", font=dict(color="#e8eaed" if dark else "#1f2124", family="Inter", size=10), showlegend=False, yaxis_title="mm de lluvia")
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info("Sin pronóstico a 3 días (datos históricos locales).")
            
            # Análisis de Condiciones (Heurística basada en pronóstico)
            if res['nivel'] in ["Naranja", "Roja"]:
                st.markdown("---")
                st.markdown("#### 🧠 Análisis de Condiciones Meteorológicas")
            
                causa_1 = "Alta acumulación de escorrentía"
                causa_2 = "Saturación del terreno"
                
                f_df = res.get('forecast')
                if f_df is not None:
                    lluvia_total = f_df['precipitacion_mm'].sum()
                    temp_max = f_df['temperatura_c'].max()
                    
                    if lluvia_total >= 40:
                        causa_1 = "Precipitaciones muy intensas"
                    elif lluvia_total >= 15:
                        causa_1 = "Lluvias continuas sobre suelo saturado"
                        
                    if temp_max >= 12:
                        causa_2 = "Altas temperaturas que provocan deshielo (Isoterma Cero Alta)"
                    elif temp_max < 5:
                        causa_2 = "Congelamiento y flujo base"
            
                st.info(f"💡 **Motivos de la alerta:** El modelo proyecta un riesgo crítico impulsado principalmente por **{causa_1.lower()}** y agravado por **{causa_2.lower()}**. Estas condiciones empujan el caudal por encima del umbral de seguridad.")

