import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Configuración de la página en modo ancho
st.set_page_config(page_title="Velocidad del Viento en Cali - SVR", layout="wide")

st.title("🌬️ Predicción de Velocidad del Viento en Cali (IDEAM)")
st.write("Modelo de **Máquina de Soporte Vectorial (SVR)** entrenado con datos reales de Datos Abiertos Colombia.")

# 2. Carga y limpieza de los 5000 datos reales del IDEAM con caché
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

# 4. Panel lateral para predicción interactiva
with st.sidebar:
    st.header("⚙️ Parámetros de Predicción")
    mes_usuario = st.slider("Selecciona el Mes", 1, 12, 6)
    hora_usuario = st.slider("Selecciona la Hora (0-23)", 0, 23, 14)
    
    entrada = pd.DataFrame({"mes": [mes_usuario], "hora": [hora_usuario]})
    prediccion = modelo.predict(entrada)[0]
    
    st.success(f"Velocidad estimada: **{prediccion:.2f} m/s**")

# 5. Tarjetas visuales con métricas en 3 columnas
col1, col2, col3 = st.columns(3)
col1.metric("Total de Datos Oficiales", len(df))
col2.metric("Error MAE", f"{mae:.4f} m/s")
col3.metric("Error MSE", f"{mse:.4f}")

# 6. Gráfica comparativa (primeros 100 datos de prueba)
st.subheader("📈 Valores Reales vs. Predichos (Primeros 100 datos de prueba)")
df_grafica = pd.DataFrame({
    "Valor Real (m/s)": y_test.values[:100],
    "Predicción SVR (m/s)": y_pred[:100]
})
st.line_chart(df_grafica)

# 7. Tabla desplegable con los 5000 datos originales
with st.expander("Ver 5000 datos oficiales del IDEAM"):
    st.dataframe(df, use_container_width=True)
