# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Título de la app
st.title('Clustering Jerárquico - Ejemplo')

# 1️⃣ Cargar archivo CSV desde el usuario
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer archivo
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:", df.head())

    # Opcional: seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_cols = st.multiselect("Selecciona variables numéricas:", numeric_cols, default=numeric_cols)

    if selected_cols:
        data = df[selected_cols]

        # 2️⃣ Estandarizar datos
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # 3️⃣ Clustering jerárquico
        Z = linkage(scaled_data, method='ward')

        # 4️⃣ Graficar dendrograma
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, labels=df.index.tolist(), ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Selecciona al menos una columna numérica.")
else:
    st.info("Por favor, sube un archivo CSV.")
