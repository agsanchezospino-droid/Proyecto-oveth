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

# Cargar hoja de estilos CSS externa
with open("styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Encabezado principal
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🟢 DATOS ABIERTOS COLOMBIA · IDEAM · SANTIAGO DE CALI</div>
    <h1>🌬️ Predicción de Velocidad del Viento en Cali</h1>
    <p>Sistema inteligente basado en <b>Máquina de Soporte Vectorial (SVR - Kernel RBF)</b> entrenado con 5000 mediciones oficiales.</p>
</div>
""", unsafe_allow_html=True)

# 2. Carga y limpieza de los 5000 datos desde el archivo Excel local con caché
@st.cache_data
def cargar_datos():
    df = pd.read_excel("vientos_cali.xlsx")
    df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
    df["valorobservado"] = pd.to_numeric(df["valorobservado"], errors="coerce")
    df = df.dropna(subset=["fechaobservacion", "valorobservado"])
    df["mes"] = df["fechaobservacion"].dt.month
    df["hora"] = df["fechaobservacion"].dt.hour
    return df

df = cargar_datos()

# 3. Entrenamiento del modelo SVR (80% entrenamiento, 20% prueba)
X = df[["mes", "hora"]]
y = df["valorobservado"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

modelo = SVR(kernel="rbf")
modelo.fit(X_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 4. Panel lateral interactivo y fácil de entender
nombres_meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

with st.sidebar:
    st.header("⚙️ Parámetros de Predicción")
    st.caption("Elige el mes y la hora para estimar la velocidad del viento.")

    # Lista desplegable con nombres de los meses y conversión interna a número (1-12)
    mes_nombre = st.selectbox("📅 Selecciona el Mes", nombres_meses, index=5)
    mes_usuario = nombres_meses.index(mes_nombre) + 1

    hora_usuario = st.slider("🕒 Selecciona la Hora (0-23)", 0, 23, 14)

    # Predicción en m/s y conversión a km/h
    entrada = pd.DataFrame({"mes": [mes_usuario], "hora": [hora_usuario]})
    prediccion = modelo.predict(entrada)[0]
    prediccion_kmh = prediccion * 3.6

    st.subheader("🎯 Resultado Estimado")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("En m/s", f"{prediccion:.2f} m/s")
    col_s2.metric("En km/h", f"{prediccion_kmh:.2f} km/h")

    # Aviso sencillo explicando qué significa esa velocidad
    if prediccion < 3:
        st.success("🍃 **Brisa suave:** Viento ligero y agradable (menor a 3 m/s).")
    elif prediccion <= 6:
        st.info("🌬️ **Viento moderado:** Brisa fresca típica de la tarde caleña (entre 3 y 6 m/s).")
    else:
        st.info("💨 **Viento fuerte:** Corrientes de aire intensas (mayor a 6 m/s).")

# 5. Vista principal organizada en pestañas
tab1, tab2 = st.tabs(["Predicción y Gráfica", "Datos del Excel (IDEAM)"])

with tab1:
    # Tarjetas visuales con métricas del modelo en 3 columnas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Datos Oficiales", len(df))
    col2.metric("Error MAE", f"{mae:.4f} m/s")
    col3.metric("Error MSE", f"{mse:.4f}")

    st.write("")

    # Gráficas visuales
    col_graf1, col_graf2 = st.columns([3, 2])

    with col_graf1:
        st.subheader("📈 Valores Reales vs. Predichos (Primeros 100 datos de prueba)")
        df_grafica = pd.DataFrame({
            "Valor Real (m/s)": y_test.values[:100],
            "Predicción SVR (m/s)": y_pred[:100]
        })
        st.line_chart(df_grafica, color=["#f97316", "#0ea5e9"], height=350)

    with col_graf2:
        st.subheader(f"🕒 Curva Diaria en {mes_nombre}")
        horas_dia = pd.DataFrame({"mes": [mes_usuario] * 24, "hora": list(range(24))})
        df_curva_dia = pd.DataFrame(
            {"Velocidad Estimada (m/s)": modelo.predict(horas_dia)},
            index=range(24)
        )
        df_curva_dia.index.name = "Hora del día"
        st.area_chart(df_curva_dia, color="#0ea5e9", height=350)

with tab2:
    st.subheader("📂 5000 Registros Oficiales del IDEAM (Más reciente a más antiguo)")
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("""
<div class="footer-note">
    Fuente oficial: <b>IDEAM - Datos Abiertos Colombia (Archivo local vientos_cali.xlsx)</b> · Motor predictivo: <b>Scikit-Learn SVR (RBF)</b>
</div>
""", unsafe_allow_html=True)
