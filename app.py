import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Configuración de la página en modo ancho
st.set_page_config(
    page_title="Velocidad del Viento en Cali - SVR",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos visuales modernos (CSS) sin alterar la lógica del sistema
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 55%, #0ea5e9 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.22);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        margin: 0 0 0.4rem 0;
        font-weight: 800;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0;
    }
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-top: 4px solid #0ea5e9;
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .pred-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.14), rgba(16, 185, 129, 0.14));
        border: 1px solid rgba(14, 165, 233, 0.35);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .pred-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0ea5e9;
        line-height: 1.2;
    }
    .pred-sub {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 0.25rem;
    }
    .footer-note {
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
    }
</style>

<div class="hero-banner">
    <div class="hero-badge">🟢 DATOS ABIERTOS COLOMBIA · IDEAM · SANTIAGO DE CALI</div>
    <h1>🌬️ Predicción de Velocidad del Viento en Cali</h1>
    <p>Sistema inteligente basado en <b>Máquina de Soporte Vectorial (SVR - Kernel RBF)</b> entrenado con mediciones meteorológicas oficiales.</p>
</div>
""", unsafe_allow_html=True)

# 2. Carga y limpieza de los 5000 datos reales del IDEAM con caché (SIN CAMBIOS)
@st.cache_data
def cargar_datos():
    url = "https://www.datos.gov.co/resource/sgfv-3yp8.csv?$q=CALI&$limit=5000"
    df = pd.read_csv(url)
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    df["valorobservado"] = pd.to_numeric(df["valorobservado"], errors="coerce")
    df = df.dropna(subset=["fechaobservacion", "valorobservado"])
    df["mes"] = df["fechaobservacion"].dt.month
    df["hora"] = df["fechaobservacion"].dt.hour
    return df

with st.spinner("Cargando datos meteorológicos oficiales del IDEAM..."):
    df = cargar_datos()

# 3. Entrenamiento del modelo SVR (80% entrenamiento, 20% prueba) (SIN CAMBIOS)
X = df[["mes", "hora"]]
y = df["valorobservado"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

modelo = SVR(kernel="rbf")
modelo.fit(X_train, y_train)

# Evaluación del modelo (SIN CAMBIOS)
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 4. Panel lateral mejorado visualmente para predicción interactiva
nombres_meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

with st.sidebar:
    st.header("⚙️ Simulador de Viento")
    st.caption("Ajusta el mes y la hora para estimar la velocidad del viento en Cali.")

    mes_usuario = st.slider("📅 Selecciona el Mes", 1, 12, 6)
    hora_usuario = st.slider("🕒 Selecciona la Hora (0-23)", 0, 23, 14)

    entrada = pd.DataFrame({"mes": [mes_usuario], "hora": [hora_usuario]})
    prediccion = modelo.predict(entrada)[0]

    # Contexto visual amigable para el usuario
    nombre_mes = nombres_meses[mes_usuario - 1]
    icono_hora = "🌙" if hora_usuario < 6 else ("🌅" if hora_usuario < 12 else ("☀️" if hora_usuario < 18 else "🌆"))
    pred_kmh = prediccion * 3.6

    if prediccion < 0.5:
        nivel_viento = "🍃 Calma"
    elif prediccion < 1.6:
        nivel_viento = "🌿 Ventolina (muy suave)"
    elif prediccion < 3.4:
        nivel_viento = "🌬️ Brisa ligera"
    elif prediccion < 5.5:
        nivel_viento = "💨 Brisa suave (típica caleña)"
    else:
        nivel_viento = "🌪️ Brisa moderada a fuerte"

    st.markdown(f"""
    <div class="pred-card">
        <div style="font-size:0.85rem; font-weight:600; margin-bottom:0.3rem;">
            📅 {nombre_mes} &nbsp;|&nbsp; {icono_hora} {hora_usuario:02d}:00 hrs
        </div>
        <div class="pred-value">{prediccion:.2f} m/s</div>
        <div class="pred-sub">Equivalente a <b>{pred_kmh:.1f} km/h</b><br>{nivel_viento}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(max(float(prediccion) / 6.0, 0.05), 1.0), text="Intensidad estimada del viento")
    st.success(f"Velocidad estimada: **{prediccion:.2f} m/s**")

# 5. Tarjetas visuales con métricas en 3 columnas
st.markdown("### 📊 Desempeño del Modelo y Datos")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Total de Datos Oficiales",
    f"{len(df):,}",
    "IDEAM Cali",
    delta_color="off",
    help="Registros válidos descargados en tiempo real desde Datos Abiertos Colombia (sgfv-3yp8)."
)
col2.metric(
    "Error MAE (Promedio)",
    f"{mae:.4f} m/s",
    "Precisión media",
    delta_color="off",
    help="Error Absoluto Medio entre los valores reales y las predicciones del conjunto de prueba (20%)."
)
col3.metric(
    "Error MSE (Cuadrático)",
    f"{mse:.4f}",
    "Kernel RBF",
    delta_color="off",
    help="Error Cuadrático Medio evaluado sobre el 20% de datos de prueba."
)

st.write("")

# 6. Sección gráfica en dos paneles visuales
col_graf1, col_graf2 = st.columns([3, 2])

with col_graf1:
    st.subheader("📈 Valores Reales vs. Predichos (Primeros 100 datos de prueba)")
    st.caption("Comparación directa entre las mediciones reales del IDEAM y la estimación del modelo SVR.")
    df_grafica = pd.DataFrame({
        "Valor Real (m/s)": y_test.values[:100],
        "Predicción SVR (m/s)": y_pred[:100]
    })
    st.line_chart(df_grafica, color=["#f97316", "#0ea5e9"], height=350)

with col_graf2:
    st.subheader(f"🕒 Curva Diaria en {nombre_mes}")
    st.caption("Comportamiento estimado del viento hora por hora (00:00 a 23:00) para el mes elegido.")
    horas_dia = pd.DataFrame({"mes": [mes_usuario] * 24, "hora": list(range(24))})
    df_curva_dia = pd.DataFrame(
        {"Velocidad Estimada (m/s)": modelo.predict(horas_dia)},
        index=range(24)
    )
    df_curva_dia.index.name = "Hora del día"
    st.area_chart(df_curva_dia, color="#0ea5e9", height=350)

# 7. Tabla desplegable con los 5000 datos originales
st.write("")
with st.expander("📂 Ver 5000 datos oficiales del IDEAM"):
    st.caption("Conjunto completo de registros meteorológicos obtenidos directamente del portal gubernamental Datos Abiertos Colombia.")
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("""
<div class="footer-note">
    Fuente oficial: <b>IDEAM - Datos Abiertos Colombia (Dataset sgfv-3yp8)</b> · Motor predictivo: <b>Scikit-Learn SVR (RBF)</b> · Santiago de Cali, Colombia
</div>
""", unsafe_allow_html=True)
