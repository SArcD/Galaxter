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
    # expansor
    # Expander con explicaciones
    with st.expander("**Ver explicación de las columnas**"):
        # 🔑 Aquí defines tus descripciones
        descripcion_columnas = {
        'SDSS': 'Nombre del catálogo Sloan Digital Sky Survey del objeto.',
        'ID': 'Identificador único del objeto, usualmente coordenadas codificadas.',
        'RA': 'Ascensión recta (Right Ascension) en grados (coordenada celeste este-oeste).',
        'Dec': 'Declinación (Declination) en grados (coordenada celeste norte-sur).',
        'Vel': 'Velocidad radial del objeto en km/s, indica movimiento relativo al observador.',
        'Rf': 'Magnitud absoluta o relativa (posiblemente magnitud fotométrica corregida).',
        'Cl_d': 'Distancia al centro del cúmulo en Mpc o arcmin (según convención).',
        'Delta': 'Desviación estadística o parámetro de subestructura local (p.ej. parámetro δ de D-S).',
        'plt.mjd.fiber': 'Placa, fecha modificada juliana (MJD) y número de fibra del SDSS, o nota de espectro.',
        'C(index)': 'Índice de concentración de luz (Petrosian u otro).',
        'M(C)': 'Clasificación morfológica basada en C(index) (p.ej. E/S0/Sa).',
        '(u-g)': 'Color fotométrico entre bandas u y g.',
        'M(u-g)': 'Clasificación morfológica asociada al color (u-g).',
        '(g-r)': 'Color fotométrico entre bandas g y r.',
        'M(g-r)': 'Clasificación morfológica asociada al color (g-r).',
        '(r-i)': 'Color fotométrico entre bandas r y i.',
        'M(r-i)': 'Clasificación morfológica asociada al color (r-i).',
        '(i-z)': 'Color fotométrico entre bandas i y z.',
        'M(i-z)': 'Clasificación morfológica asociada al color (i-z).',
        'M(parn)': 'Clasificación morfológica paramétrica (posible resultado de un modelo).',
        'M(ave)': 'Clasificación morfológica promedio de distintos métodos.',
        'M(IPn)': 'Clasificación morfológica de acuerdo a Plauchu (con claves numéricas).',
        'M(IP)': 'Clasificación morfológica de acuerdo a Plauchu.',
        'Act': 'Clasificación de la actividad nuclear: TO (Transition Object), SFG (Star-Forming Galaxy), '
            'LLA (Low-Luminosity AGN), UNK (Desconocido), NoE (Sin emisión).'
        }

        for col, desc in descripcion_columnas.items():
            st.markdown(f"**{col}**: {desc}")

    
    # Buscador de variables
    import difflib

    # 🕵️‍♂️ Barra de búsqueda para encontrar columnas por nombre
    st.subheader("🔍 Buscar variable por nombre")

    search_query = st.text_input("Escribe parte del nombre de la variable:")

    if search_query:
        # Lista de nombres de columnas
        cols = list(descripcion_columnas.keys())

        # Encuentra la coincidencia más cercana usando difflib
        best_match = difflib.get_close_matches(search_query, cols, n=1, cutoff=0.1)

        if best_match:
            col_name = best_match[0]
            description = descripcion_columnas[col_name]
            st.success(f"**{col_name}**: {description}")
        else:
            st.warning("No se encontró ninguna coincidencia.")
    else:
        st.info("Empieza a escribir para buscar una variable.")

    
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
