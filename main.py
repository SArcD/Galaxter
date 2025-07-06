# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

    search_query = st.text_input("Escribe parte del nombre de la variable:", key="var_search_desc")

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

    
    import plotly.express as px
    import plotly.figure_factory as ff

    with st.expander("📊 Análisis exploratorio: Distribuciones, Pair Plot y Correlación"):
        st.subheader("1️⃣ Distribución univariada de una variable numérica")

        # Lista de columnas numéricas en tu DataFrame
        numeric_colss = df.select_dtypes(include='number').columns.tolist()

        # Caja de búsqueda para variable numérica
        search_var = st.text_input("🔍 Busca una variable numérica para graficar su distribución:", key="var_search_dist")

        if search_var:
            best_match_var = difflib.get_close_matches(search_var, numeric_colss, n=1, cutoff=0.1)
            if best_match_var:
                col = best_match_var[0]
                st.success(f"Mostrando distribución para: **{col}**")
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribución de {col}")
                st.plotly_chart(fig)
            else:
                st.warning("No se encontró ninguna variable numérica similar.")
        else:
            st.info("Empieza a escribir para buscar la variable numérica.")

        st.divider()

        st.subheader("2️⃣ Pair Plot de variables numéricas")

        # Multiselect para elegir variables para el pair plot
        selected_pair_cols = st.multiselect(
            "Selecciona dos variables para el pair plot (o selecciona más para todos)",
            options=numeric_colss,
            default=numeric_colss[:2] if len(numeric_colss) >= 2 else numeric_colss
        )

        if len(selected_pair_cols) >= 2:
            fig_pair = px.scatter_matrix(
                df[selected_pair_cols],
                dimensions=selected_pair_cols,
                title="Pair Plot"
            )
            st.plotly_chart(fig_pair)
        else:
            st.info("Selecciona al menos dos variables para el pair plot.")

        st.divider()

        st.subheader("3️⃣ Matriz de correlación")

        # Calcular y graficar matriz de correlación
        if numeric_colss:
            corr_matrix = df[numeric_colss].corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matriz de correlación",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig_corr)
        else:
            st.warning("No hay variables numéricas para calcular correlación.")






    # Expansor cpm el mapa de Abell 85
    with st.expander("🌌 Ver mapa interactivo del cúmulo Abell 85"):
        num_vars = ['Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)']
        cat_vars = ['M(parn)', 'Act']
        all_vars = num_vars + cat_vars

        selected_var = st.selectbox(
            "Variable para filtrar y colorear puntos:",
            options=all_vars
        )

        df_filtered = df.copy()

        # Filtrado según tipo
        if selected_var in num_vars:
            try:
                df_filtered['range_label'] = pd.qcut(df_filtered[selected_var], 5, duplicates='drop')
                labels = df_filtered['range_label'].cat.categories.astype(str).tolist()
                #labels = fcluster(Z, t=num_clusters, criterion=criterion)
                #df.loc[data.index, 'Subcluster'] = labels
                selected_labels = st.multiselect(
                    "Selecciona uno o varios rangos:",
                    options=labels,
                    default=labels
                )
                df_filtered = df_filtered[df_filtered['range_label'].astype(str).isin(selected_labels)]
            except Exception as e:
                st.warning(f"No se pudieron crear rangos para esta variable. Detalle: {e}")
        elif selected_var in cat_vars:
            labels = df_filtered[selected_var].dropna().unique().tolist()
            selected_labels = st.multiselect(
                "Selecciona una o varias categorías:",
                options=labels,
                default=labels
            )
            df_filtered = df_filtered[df_filtered[selected_var].isin(selected_labels)]

        # Hover enriquecido
        hover_data = {
            "RA": True, "Dec": True,
            "Vel": True, "Cl_d": True, "Delta": True,
            "(u-g)": True, "(g-r)": True, "(r-i)": True, "(i-z)": True,
            "M(IP)": True, "M(ave)": True, "Act": True
        }

        # Columnas necesarias para asegurar que existan
        required_cols = {'RA', 'Dec', 'ID'} | set(hover_data.keys())

        if required_cols.issubset(df.columns):
            fig = px.scatter(
                df_filtered,
                x="RA",
                y="Dec",
                color=selected_var,
                hover_name="ID",
                hover_data=hover_data,
                title=f"Mapa filtrado por: {selected_var}"
            )

            fig.add_trace(
                go.Histogram2dContour(
                    x=df_filtered['RA'],
                    y=df_filtered['Dec'],
                    ncontours=10,
                    colorscale='Viridis',
                    contours_coloring='lines',
                    line_width=2,
                    opacity=0.5,
                    showscale=False,
                    hoverinfo='skip'
                )
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                height=700,
                width=900
            )

            st.plotly_chart(fig)

            # Botón para descargar tabla filtrada
            st.download_button(
                "💾 Descargar tabla filtrada",
                df_filtered.to_csv(index=False).encode('utf-8'),
                file_name="galaxias_filtradas.csv",
                mime="text/csv"
            )
        else:
            st.warning(
                f"Faltan columnas necesarias para el mapa interactivo: "
                f"{required_cols - set(df.columns)}"
            )

    with st.expander("🔍 Buscar subestructuras"):
        st.subheader("🧬 Clustering Jerárquico")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_cols = st.multiselect(
            "Selecciona variables numéricas para clustering:",
            options=numeric_cols,
            default=numeric_cols
        )

        if selected_cols:
            data = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()

            if data.shape[0] < 2:
                st.warning("No hay suficientes datos después de limpiar filas para clustering.")
            else:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)

                Z = linkage(scaled_data, method='ward')

            #     🟢 NUEVO: Parámetros interactivos
                num_clusters = st.slider(
                    "Número de subestructuras:",
                    min_value=2, max_value=10, value=4
                )
                criterion = st.selectbox(
                    "Criterio de corte para fcluster:",
                    options=['maxclust', 'distance'],
                    index=0
                )

                from scipy.cluster.hierarchy import fcluster
                #if criterion == 'maxclust':
                #    df['Subcluster'] = fcluster(Z, t=num_clusters, criterion=criterion)
                #else:
                #    # Para 'distance', podrías ajustar t= umbral de distancia
                #    distance_threshold = st.number_input(
                #        "Umbral de distancia:", min_value=0.0, value=10.0, step=0.5
                #    )
                #    df['Subcluster'] = fcluster(Z, t=distance_threshold, criterion=criterion)
                if criterion == 'maxclust':
                    labels = fcluster(Z, t=num_clusters, criterion=criterion)
                else:
                    labels = fcluster(Z, t=distance_threshold, criterion=criterion)

                df.loc[data.index, 'Subcluster'] = labels

                
                fig_dendro, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, labels=data.index.tolist(), ax=ax)
                ax.set_title("Dendrograma de Clustering Jerárquico")
                ax.set_xlabel("Índices de galaxias")
                ax.set_ylabel("Distancia")
                st.pyplot(fig_dendro)

            #     🟢 PANEL t-SNE + Boxplots
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                vars_phys = selected_cols
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

                if 'Subcluster' in df.columns and 'TSNE1' in df.columns and 'TSNE2' in df.columns:
                    unique_clusters = sorted(df['Subcluster'].dropna().unique())

                    n_cols = 3
                    n_rows = (len(vars_phys) + n_cols - 1) // n_cols
                    total_rows = n_rows + 1

                    specs = [[{"colspan": n_cols}] + [None]*(n_cols-1)]
                    for _ in range(n_rows):
                        specs.append([{} for _ in range(n_cols)])

                    subplot_titles = ["PCA + t-SNE Clustering"] + vars_phys

                    fig = make_subplots(
                        rows=total_rows, cols=n_cols,
                        specs=specs,
                        subplot_titles=subplot_titles
                    )

                    for i, cluster in enumerate(unique_clusters):
                        cluster_data = df[df['Subcluster'] == cluster]
                        hover_text = (
                            "<b>ID:</b> " + cluster_data['ID'].astype(str) +
                            "<br><b>RA:</b> " + cluster_data['RA'].round(4).astype(str) +
                            "<br><b>Dec:</b> " + cluster_data['Dec'].round(4).astype(str) +
                            "<br><b>Vel:</b> " + cluster_data['Vel'].round(1).astype(str) +
                            "<br><b>Cl_d:</b> " + cluster_data['Cl_d'].round(3).astype(str) +
                            "<br><b>Delta:</b> " + cluster_data['Delta'].round(3).astype(str) +
                            "<br><b>M(IPn):</b> " + cluster_data['M(IPn)'].astype(str) +
                            "<br><b>TSNE1:</b> " + cluster_data['TSNE1'].round(3).astype(str) +
                            "<br><b>TSNE2:</b> " + cluster_data['TSNE2'].round(3).astype(str)
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=cluster_data['TSNE1'],
                                y=cluster_data['TSNE2'],
                                mode='markers',
                                name=f'Subcluster {cluster}',
                                legendgroup=f'Subcluster {cluster}',
                                showlegend=True,
                                marker=dict(
                                    size=6,
                                    color=colors[i % len(colors)],
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                text=hover_text,
                                hoverinfo='text'
                            ),
                            row=1, col=1
                        )    

                    fig.add_trace(
                        go.Histogram2dContour(
                            x=df['TSNE1'],
                            y=df['TSNE2'],
                            colorscale='Greys',
                            reversescale=True,
                            opacity=0.2,
                            showscale=False,
                            hoverinfo='skip',
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                    for idx, var in enumerate(vars_phys):
                        row = (idx // n_cols) + 2
                        col = (idx % n_cols) + 1
                        for j, cluster in enumerate(unique_clusters):
                            cluster_data = df[df['Subcluster'] == cluster]
                            hover_text = (
                                "<b>ID:</b> " + cluster_data['ID'].astype(str) +
                                f"<br><b>{var}:</b> " + cluster_data[var].astype(str) +
                                "<br><b>Subcluster:</b> " + cluster_data['Subcluster'].astype(str)
                            )

                            fig.add_trace(
                                go.Box(
                                    y=cluster_data[var],
                                    x=[f'Subcluster {cluster}'] * len(cluster_data),
                                    name=f'Subcluster {cluster}',
                                    legendgroup=f'Subcluster {cluster}',
                                    showlegend=False,
                                    boxpoints='all',
                                    jitter=0.5,
                                    pointpos=-1.8,
                                    marker_color=colors[j % len(colors)],
                                    notched=True,
                                    text=hover_text,
                                    hoverinfo='text',
                                    width=0.6
                                ),
                                row=row, col=col
                            )

                    fig.update_layout(
                        title="Panel interactivo: PCA + t-SNE + Boxplots",
                        template='plotly_white',
                        width=400 * n_cols,
                        height=400 * total_rows,
                        boxmode='group',
                        legend_title="Subclusters"
                    )

                    fig.update_xaxes(title="t-SNE 1", row=1, col=1)
                    fig.update_yaxes(title="t-SNE 2", row=1, col=1)

                    for idx, var in enumerate(vars_phys):
                        row = (idx // n_cols) + 2
                        col = (idx % n_cols) + 1
                        fig.update_xaxes(title="Subcluster", row=row, col=col)
                        fig.update_yaxes(title=var, row=row, col=col)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Faltan columnas 'Subcluster', 'TSNE1' o 'TSNE2' para el panel interactivo."
                    )
        else:
            st.info("Selecciona al menos una variable numérica para generar el dendrograma y panel t-SNE.")





    
    #else:
    #    st.warning("Selecciona al menos una columna numérica.")
else:
    st.info("Por favor, sube un archivo CSV.")
