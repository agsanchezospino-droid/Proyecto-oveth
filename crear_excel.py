import pandas as pd

# Descargar 5000 registros reales del IDEAM (Cali) ordenados del más reciente al más antiguo
url = "https://www.datos.gov.co/resource/sgfv-3yp8.csv?$q=CALI&$limit=5000&$order=fechaobservacion%20DESC"
df = pd.read_csv(url)

# Guardar en archivo Excel local
df.to_excel("vientos_cali.xlsx", index=False)
print(f"Archivo 'vientos_cali.xlsx' creado exitosamente con {len(df)} registros.")
