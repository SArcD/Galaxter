# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Título de la app
#st.image("GCexplorer.PNG", use_column_width=True)
#st.image("Designer.png")

#import streamlit as st

# Crea dos columnas: una para el logo, otra para el texto
col1, col2 = st.columns([1, 3])

with col1:
    st.image("Designer.png", width=400)  # Más pequeño, ajusta el ancho como prefieras

with col2:
    st.markdown(
        """
        <div style='
            background-color: #001F3F;   /* Azul marino profundo */
            padding: 25px 90px;
            border-radius: 8px;
            display: inline-block;
        '>
            <h1 style='
                color: #FFD700;          /* Dorado */
                font-family: sans-serif;
                margin: 0;
                font-size: 35px;
            '>
                Galaxy Cluster Explorer
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )





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
                width=1500,   # Cambia el ancho aquí
                height=1500,   # Cambia la altura aquí
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
        num_vars = ['Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)', 'Delta']
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
            fig.update_xaxes(autorange="reversed")

            
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




        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import gaussian_kde

        # ✅ Sub-sección para suavizar
        st.subheader("🗺️ Mapa suavizado de densidad para variable seleccionada")

        # Ofrece solo numéricas que tengan sentido
        smooth_var = st.selectbox(
            "Variable para mapa suavizado:",
            options=['Delta', 'Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)'],
            index=0
        )

        # Filtrar puntos con datos válidos
        df_smooth = df_filtered[df_filtered[smooth_var].notna()]

        if df_smooth.empty:
            st.warning("No hay datos válidos para suavizar.")
        else:
            # Crear malla
            ra = df_smooth['RA'].values
            dec = df_smooth['Dec'].values
            z = df_smooth[smooth_var].values

            # Grid para interpolar
            xi, yi = np.mgrid[ra.min():ra.max():200j, dec.min():dec.max():200j]

            # KDE 2D ponderada por variable
            positions = np.vstack([ra, dec])
            kernel = gaussian_kde(positions, weights=z, bw_method=0.3)

            zi = np.reshape(kernel(np.vstack([xi.ravel(), yi.ravel()])), xi.shape)

            fig_smooth, ax = plt.subplots(figsize=(8, 6))
            cf = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
            scatter = ax.scatter(ra, dec, c=z, cmap='viridis', edgecolor='k', s=20)

            ax.set_title(f"Mapa suavizado (KDE) — {smooth_var}")
            ax.set_xlabel("Ascensión Recta (RA, grados)")
            ax.set_ylabel("Declinación (Dec, grados)")
            ax.invert_xaxis()  # Invertir RA como en tus otros mapas
            fig_smooth.colorbar(cf, ax=ax, label=smooth_var)

            st.pyplot(fig_smooth)

    
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

            #     ✅ Controles interactivos
                num_clusters = st.slider(
                    "Número de subestructuras (clusters):",
                    min_value=2, max_value=10, value=4
                )
                criterion = st.selectbox(
                    "Criterio de corte para fcluster:",
                    options=['maxclust', 'distance'],
                    index=0
                )

                from scipy.cluster.hierarchy import fcluster

                if criterion == 'maxclust':
                    labels = fcluster(Z, t=num_clusters, criterion=criterion)
                else:
                    distance_threshold = st.number_input(
                        "Umbral de distancia:", min_value=0.0, value=10.0, step=0.5
                    )
                    labels = fcluster(Z, t=distance_threshold, criterion=criterion)

                df.loc[data.index, 'Subcluster'] = labels

                # ✅ Dendrograma
                fig_dendro, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, labels=data.index.tolist(), ax=ax)
                ax.set_title("Dendrograma de Clustering Jerárquico")
                ax.set_xlabel("Índices de galaxias")
                ax.set_ylabel("Distancia")
                st.pyplot(fig_dendro)
    
                # ✅ Generar TSNE1 y TSNE2 si faltan
                if 'TSNE1' not in df.columns or 'TSNE2' not in df.columns:
                    st.info("Generando TSNE1 y TSNE2 dinámicamente con PCA + t-SNE...")
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE

                    pca = PCA(n_components=min(20, scaled_data.shape[1])).fit_transform(scaled_data)
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(pca)

                    df.loc[data.index, 'TSNE1'] = tsne_result[:, 0]
                    df.loc[data.index, 'TSNE2'] = tsne_result[:, 1]

                # ✅ PANEL t-SNE + Boxplots
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                #vars_phys = selected_cols
                vars_phys = numeric_cols
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

                if 'Subcluster' in df.columns and 'TSNE1' in df.columns and 'TSNE2' in df.columns:
                    unique_clusters = sorted(df.loc[data.index, 'Subcluster'].dropna().unique())

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

                # Scatter t-SNE
                    for i, cluster in enumerate(unique_clusters):
                        cluster_data = df.loc[data.index][df.loc[data.index, 'Subcluster'] == cluster]
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
                            x=df.loc[data.index, 'TSNE1'],
                            y=df.loc[data.index, 'TSNE2'],
                            colorscale='Greys',
                            reversescale=True,
                            opacity=0.2,
                            showscale=False,
                            hoverinfo='skip',
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                    # Boxplots
                    for idx, var in enumerate(vars_phys):
                        row = (idx // n_cols) + 2
                        col = (idx % n_cols) + 1
                        for j, cluster in enumerate(unique_clusters):
                            cluster_data = df.loc[data.index][df.loc[data.index, 'Subcluster'] == cluster]
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
                    st.warning("Faltan columnas 'Subcluster', 'TSNE1' o 'TSNE2' para el panel interactivo.")
        else:
            st.info("Selecciona al menos una variable numérica para generar el dendrograma y panel t-SNE.")


        # ✅ Visualizar tabla filtrada por subcluster específico
        st.subheader("🔍 Explora los datos de un subcluster específico")

        if 'Subcluster' in df.columns:
            unique_subclusters = sorted(df.loc[data.index, 'Subcluster'].dropna().unique())
            selected_sub = st.selectbox(
                "Selecciona un Subcluster para mostrar sus filas:",
                options=unique_subclusters
            )

            filtered_df = df.loc[
                (df.index.isin(data.index)) & (df['Subcluster'] == selected_sub)
            ]

            st.dataframe(filtered_df)

            # Botón para descargar CSV de este subcluster
            st.download_button(
                "💾 Descargar CSV de este Subcluster",
                filtered_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Subcluster_{selected_sub}.csv",
                mime="text/csv"
            )
        else:
            st.info("No se ha generado la columna 'Subcluster'.")


        import plotly.graph_objects as go
        import plotly.express as px

    with st.expander("🗺️ Mapa final Abell 85 por Subestructura"):
        import plotly.graph_objects as go
        import plotly.express as px

        df_plot = df[df['Subcluster'].notna()].copy()  # ✅ Solo puntos con Subcluster

        if not df_plot.empty:
            unique_clusters = sorted(df_plot['Subcluster'].unique())
            colores = px.colors.qualitative.Set2

            fig = go.Figure()

            for i, cluster in enumerate(unique_clusters):
                cl_data = df_plot[df_plot['Subcluster'] == cluster]
                color = colores[i % len(colores)]
                cl_str = f"Subestructura {cluster}"

                # Curvas de contorno
                fig.add_trace(go.Histogram2dContour(
                    x=cl_data['RA'],
                    y=cl_data['Dec'],
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, color]],
                    showscale=False,
                    opacity=0.3,
                    ncontours=8,
                    line=dict(width=1, color=color),
                    name=cl_str,
                    legendgroup=cl_str,
                    hoverinfo="skip",
                    showlegend=False
                ))

                # Puntos
                fig.add_trace(go.Scatter(
                    x=cl_data['RA'],
                    y=cl_data['Dec'],
                    mode='markers',
                    marker=dict(size=6, color=color),
                    name=cl_str,
                    legendgroup=cl_str,
                    hovertemplate="<br>".join([
                        "ID: %{customdata[0]}",
                        "RA: %{x}",
                        "Dec: %{y}",
                        "Vel: %{customdata[1]}",
                        "Delta: %{customdata[2]}"
                    ]),
                    customdata=cl_data[['ID', 'Vel', 'Delta']],
                    showlegend=True
                ))

                # Estrella centro
                ra_centro = cl_data['RA'].mean()
                dec_centro = cl_data['Dec'].mean()

                fig.add_trace(go.Scatter(
                    x=[ra_centro],
                    y=[dec_centro],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color=color, line=dict(width=1, color="black")),
                    name=f"{cl_str} (Centro)",
                    legendgroup=cl_str,
                    showlegend=False,
                    hoverinfo="skip"
                ))

            fig.update_layout(
                title="Mapa RA–Dec por Subcluster con Curvas QS y Centros",
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                legend_title="Subestructura",
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showgrid=False,
                    autorange="reversed"  # 🔥 Invierte RA
                ),
                yaxis=dict(showgrid=False),
                font=dict(color="black")
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos con Subcluster asignado para mostrar el mapa.")


#with st.expander("📊 Diagramas de caja por Subestructura"):
        import plotly.express as px
        import pandas as pd

        # 1️⃣ Variables candidatas
        vars_candidates = [
            'RA', 'Dec', 'Vel', 'Delta', 'Cl_d',
            'Rf', 'C(index)', '(u-g)', '(g-r)', 'M(IPn)', 'Act'
        ]

        selected_vars = st.multiselect(
            "Selecciona variables para analizar evidencias de subestructuras:",
            options=vars_candidates,
            default=['RA', 'Dec', 'Vel', 'Delta']
        )

        # 2️⃣ Tabla de interpretación
        st.markdown("**🔍 Sugerencias para interpretar cada variable:**")

        table_data = [
            ["RA, Dec", "Distribución espacial: busca agrupaciones locales."],
            ["Vel", "Picos secundarios o colas: indica grupos cinemáticamente distintos."],
            ["Delta", "Desviación local: zonas con dinámica diferente."],
            ["Cl_d", "Distancia al centro: grupos externos o desplazados radialmente."],
            ["Rf", "Magnitud: galaxias brillantes dominantes en subcúmulos."],
            ["C(index)", "Concentración de luz: relación con morfología."],
            ["(u-g), (g-r)", "Colores: poblaciones estelares jóvenes/viejas."],
            ["M(IPn)", "Morfología interna: coherencia morfológica."],
            ["Act", "Actividad nuclear: AGN/starburst asociados a interacción."]
        ]
        df_tips = pd.DataFrame(table_data, columns=["Variable", "¿Qué observar?"])
        st.table(df_tips)

        # 3️⃣ Boxplots dinámicos
        if 'Subcluster' in df.columns:
            df_box = df[df['Subcluster'].notna()].copy()

            if selected_vars and not df_box.empty:
                st.info("Observa si los diagramas muestran distribuciones diferenciadas entre subestructuras.")
                for var in selected_vars:
                    fig = px.box(
                        df_box,
                        x='Subcluster',
                        y=var,
                        color='Subcluster',
                        points='all',
                        notched=True,
                        title=f"Distribución de {var} por Subestructura",
                        labels={'Subcluster': 'Subestructura', var: var},
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig.update_traces(width=0.6, jitter=0.3)
                    fig.update_layout(
                        yaxis_title=var,
                        xaxis_title='Subestructura',
                        legend_title='Subestructura',
                        boxmode='group',
                        boxgap=0.1
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se seleccionaron variables o no hay datos para graficar.")
        else:
            st.info("No se ha generado la columna 'Subcluster'.")

#with st.expander("📊 Barras estratificadas por Subestructura"):
        import plotly.express as px

        # Variables categóricas posibles
        cat_candidates = ['Act', 'M(IPn)', 'M(IP)', 'M(ave)', 'M(C)']

        selected_cats = st.multiselect(
            "Selecciona variables categóricas para ver proporciones por Subestructura:",
            options=cat_candidates,
            default=['Act']
        )

        if 'Subcluster' in df.columns and selected_cats:
            df_cat = df[df['Subcluster'].notna()].copy()

            for cat_var in selected_cats:
                if cat_var in df_cat.columns:
                    # Asegúrate de tratar como categórica
                    cat_name = f"{cat_var}_cat"
                    df_cat[cat_name] = df_cat[cat_var].astype(str)

                    # Proporciones
                    df_prop = (
                        df_cat.groupby(['Subcluster', cat_name])
                        .size()
                        .reset_index(name='Count')
                    )

                    totals = df_prop.groupby('Subcluster')['Count'].transform('sum')
                    df_prop['Proportion'] = df_prop['Count'] / totals
                    df_prop['Percent'] = (df_prop['Proportion'] * 100).round(1)

                    fig_bar = px.bar(
                        df_prop,
                        x='Subcluster',
                        y='Proportion',
                        color=cat_name,
                        text=df_prop['Percent'].astype(str) + '%',
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        title=f'Proporción por categoría {cat_var} en cada Subestructura',
                        barmode='stack',
                        height=600
                    )

                    fig_bar.update_layout(
                        yaxis=dict(title='Proporción'),
                        xaxis=dict(title='Subestructura'),
                        legend_title=cat_var,
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(f"La columna '{cat_var}' no existe en tu DataFrame.")
        else:
            st.info("No se encuentran las columnas necesarias o no se seleccionó ninguna variable.")

    import numpy as np
    from scipy.spatial import KDTree
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    import numpy as np
    from scipy.spatial import KDTree
    import plotly.graph_objects as go
    from tqdm import tqdm

    import numpy as np
    from scipy.spatial import KDTree
    import plotly.graph_objects as go
    import pandas as pd
    from tqdm import tqdm

    with st.expander("📊 Prueba Dressler–Shectman Interactiva + Bootstrapping"):
        st.subheader("🧪 Análisis DS + Densidad Local + p-valor (Monte Carlo)")

        if 'Subcluster' in df.columns:
            unique_subclusters = sorted(df['Subcluster'].dropna().unique())
            selected_sub = st.selectbox(
                "Selecciona una Subestructura para la prueba DS:",
                options=unique_subclusters
            )

            df_sub = df[df['Subcluster'] == selected_sub].copy()

            if df_sub.empty or df_sub.shape[0] < 5:
                st.warning("No hay suficientes galaxias para aplicar la prueba.")
            else:
                st.success(f"Galaxias seleccionadas: {len(df_sub)}")

                coords = df_sub[['RA', 'Dec']].values
                velocities = df_sub['Vel'].values

                N = int(np.sqrt(len(coords)))
                tree = KDTree(coords)
                neighbors_idx = [tree.query(coords[i], k=N+1)[1][1:] for i in range(len(coords))]

                V_global = np.mean(velocities)
                sigma_global = np.std(velocities)

                delta = []
                for i, neighbors in enumerate(neighbors_idx):
                    local_vel = np.mean(velocities[neighbors])
                    local_sigma = np.std(velocities[neighbors])

                    d_i = ((N + 1) / sigma_global**2) * (
                        (local_vel - V_global)**2 + (local_sigma - sigma_global)**2
                    )
                    delta.append(np.sqrt(d_i))

                df_sub['Delta'] = delta
                DS_stat_real = np.sum(delta)
                st.write(f"**Estadístico Dressler–Shectman Δ real:** {DS_stat_real:.2f}")

                # ✅ Bootstrapping / Monte Carlo
                st.info("Calculando distribución nula con permutaciones (Monte Carlo)...")
                n_permutations = st.slider("Número de permutaciones:", 100, 2000, 500, step=100)

                DS_stats_permuted = []
                for _ in tqdm(range(n_permutations), desc="Permutando"):
                    velocities_perm = np.random.permutation(velocities)
                    delta_perm = []
                    for i, neighbors in enumerate(neighbors_idx):
                        local_vel = np.mean(velocities_perm[neighbors])
                        local_sigma = np.std(velocities_perm[neighbors])
                        d_i = ((N + 1) / sigma_global**2) * (
                            (local_vel - V_global)**2 + (local_sigma - sigma_global)**2
                        )
                        delta_perm.append(np.sqrt(d_i))
                    DS_stats_permuted.append(np.sum(delta_perm))

                DS_stats_permuted = np.array(DS_stats_permuted)
                p_value = np.sum(DS_stats_permuted >= DS_stat_real) / n_permutations
                st.write(f"**p-valor empírico:** {p_value:.4f}")

                # ✅ Histograma nulo vs. real
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=DS_stats_permuted,
                    nbinsx=30,
                    name="Δ permutado",
                    marker_color='lightgrey'
                ))

                fig_hist.add_trace(go.Scatter(
                    x=[DS_stat_real, DS_stat_real],
                    y=[0, np.histogram(DS_stats_permuted, bins=30)[0].max()],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    name='Δ real'
                ))

                fig_hist.update_layout(
                    title="Distribución nula (Δ permutado) vs. Δ real",
                    xaxis_title='Δ',
                    yaxis_title='Frecuencia',
                    template='plotly_white'
                )

                st.plotly_chart(fig_hist, use_container_width=True)

                # ✅ Clasifica Delta en rangos y hover detallado
                bins = [0, 1, 2, 3, 5]
                labels = ['Bajo', 'Medio', 'Alto', 'Muy Alto']

                df_sub['Delta_cat'] = pd.cut(df_sub['Delta'], bins=bins, labels=labels)

                color_map = {
                    'Bajo': '#1f77b4',
                    'Medio': '#2ca02c',
                    'Alto': '#ff7f0e',
                    'Muy Alto': '#d62728'
                }

                df_sub['color'] = df_sub['Delta_cat'].map(color_map)

                hover_text = df_sub.apply(
                    lambda row:
                    f"SDSS: {row['SDSS']}<br>"
                    f"ID: {row['ID']}<br>"
                    f"RA: {row['RA']:.3f}°<br>"
                    f"Dec: {row['Dec']:.3f}°<br>"
                    f"Velocidad (km/s): {row['Vel']:.1f}<br>"
                    f"Rf: {row['Rf']:.2f}<br>"
                    f"Cl_d (Dist. centro): {row['Cl_d']:.2f}<br>"
                    f"Delta: {row['Delta']:.3f} ({row['Delta_cat']})<br>"
                    f"C(index): {row['C(index)']:.2f}<br>"
                    f"Morfología M(C): {row['M(C)']}<br>"
                    f"(u-g): {row['(u-g)']:.2f}, M(u-g): {row['M(u-g)']}<br>"
                    f"(g-r): {row['(g-r)']:.2f}, M(g-r): {row['M(g-r)']}<br>"
                    f"Actividad: {row['Act']}",
                    axis=1
                )


                fig = go.Figure()

                # 1️⃣ Puntos: una traza por categoría Delta
                for cat, color in color_map.items():
                    df_cat = df_sub[df_sub['Delta_cat'] == cat]
                    if not df_cat.empty:
                        hover_text_cat = df_cat.apply(
                            lambda row:
                            f"SDSS: {row['SDSS']}<br>"
                            f"ID: {row['ID']}<br>"
                            f"RA: {row['RA']:.3f}°<br>"
                            f"Dec: {row['Dec']:.3f}°<br>"
                            f"Velocidad (km/s): {row['Vel']:.1f}<br>"
                            f"Rf: {row['Rf']:.2f}<br>"
                            f"Cl_d (Dist. centro): {row['Cl_d']:.2f}<br>"
                            f"Delta: {row['Delta']:.3f} ({row['Delta_cat']})<br>"
                            f"C(index): {row['C(index)']:.2f}<br>"
                            f"Morfología M(C): {row['M(C)']}<br>"
                            f"(u-g): {row['(u-g)']:.2f}, M(u-g): {row['M(u-g)']}<br>"
                            f"(g-r): {row['(g-r)']:.2f}, M(g-r): {row['M(g-r)']}<br>"
                            f"Actividad: {row['Act']}",
                            axis=1
                        )
        
                        fig.add_trace(go.Scatter(
                            x=df_cat['RA'],
                            y=df_cat['Dec'],
                            mode='markers',
                            name=f'Delta: {cat}',
                            marker=dict(size=8, color=color, line=dict(width=0.5, color='DarkSlateGrey')),
                            text=hover_text_cat,
                            hoverinfo='text'
                        ))

                # 2️⃣ Contorno KDE (sin hover)
                fig.add_trace(go.Histogram2dContour(
                    x=df_sub['RA'],
                    y=df_sub['Dec'],
                    colorscale='Blues',
                    reversescale=True,
                    showscale=False,
                    opacity=0.3,
                    name='Densidad Local',
                    hoverinfo='skip',
                    ncontours=15
                ))

                # 3️⃣ Layout
                fig.update_layout(
                    title=f'DS + Densidad Local — Subestructura {selected_sub}',
                    xaxis_title='Ascensión Recta (RA, grados)',
                    yaxis_title='Declinación (Dec, grados)',
                    xaxis=dict(autorange='reversed'),
                    template='plotly_white',
                    height=700,
                    width=900
                )

                st.plotly_chart(fig, use_container_width=True)

                
                # ✅ Exporta resultados
                with st.expander("📄 Ver tabla de galaxias con Delta"):
                    st.dataframe(df_sub[['SDSS', 'ID', 'RA', 'Dec', 'Vel', 'Rf', 'Cl_d',
                                     'Delta', 'Delta_cat', 'C(index)', 'M(C)',
                                     '(u-g)', 'M(u-g)', '(g-r)', 'M(g-r)', 'Act']])
                    st.download_button(
                        "💾 Descargar resultados DS",
                        df_sub.to_csv(index=False).encode('utf-8'),
                        file_name=f"DS_Subcluster_{selected_sub}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No se ha generado la columna 'Subcluster'. Ejecuta el clustering jerárquico primero.")


        st.subheader("📑 Evidencias de coherencia morfológica y dinámica")

        if 'Delta_cat' in df_sub.columns:
            unique_cats = df_sub['Delta_cat'].dropna().unique()
            selected_cat = st.selectbox(
                "Selecciona un rango de Delta:",
                options=unique_cats
            )

            df_cat = df_sub[df_sub['Delta_cat'] == selected_cat].copy()

            if df_cat.empty:
                st.warning("No hay datos para este rango.")
            else:
                st.success(f"Galaxias en rango '{selected_cat}': {len(df_cat)}")

                # 1️⃣ Barras de M(C)
                fig_morph = px.histogram(
                    df_cat,
                    x='M(C)',
                    color='M(C)',
                    text_auto=True,
                    title=f"Distribución morfológica M(C) para Delta: {selected_cat}",
                    category_orders={"M(C)": sorted(df_cat['M(C)'].unique())}
                )
                fig_morph.update_layout(
                    xaxis_title="Morfología M(C)",
                    yaxis_title="Número de galaxias",
                    showlegend=False
                )
                st.plotly_chart(fig_morph, use_container_width=True)

                # 2️⃣ Barras de Act (Actividad Nuclear)
                fig_act = px.histogram(
                    df_cat,
                    x='Act',
                    color='Act',
                    text_auto=True,
                    title=f"Distribución de Actividad Nuclear para Delta: {selected_cat}",
                    category_orders={"Act": sorted(df_cat['Act'].unique())}
                )
                fig_act.update_layout(
                    xaxis_title="Actividad Nuclear",
                    yaxis_title="Número de galaxias",
                    showlegend=False
                )
                st.plotly_chart(fig_act, use_container_width=True)

                # 3️⃣ Boxplots de Vel y Delta
                fig_vel = px.box(
                    df_cat,
                    y='Vel',
                    points='all',
                    notched=True,
                    title=f"Distribución de Velocidad (km/s) para Delta: {selected_cat}",
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_vel, use_container_width=True)

                fig_delta = px.box(
                    df_cat,
                    y='Delta',
                    points='all',
                    notched=True,
                    title=f"Distribución de Delta para Delta: {selected_cat}",
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_delta, use_container_width=True)

        else:
            st.info("No se ha generado la columna Delta_cat. Ejecuta primero la prueba DS.")


    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    def plot_conditional_panel(df):
        st.subheader("🎛️ Ajusta percentiles para bins Delta y Vel")

        n_bins_delta = st.slider("Número de bins Delta:", 2, 10, 4)
        n_bins_vel = st.slider("Número de bins Vel:", 2, 10, 4)

        df_cond = df.copy()
        df_cond['Delta_bin'] = pd.qcut(df_cond['Delta'], q=n_bins_delta, labels=[f'Δ{i+1}' for i in range(n_bins_delta)])
        df_cond['Vel_bin'] = pd.qcut(df_cond['Vel'], q=n_bins_vel, labels=[f'V{i+1}' for i in range(n_bins_vel)])

        df_cond = df_cond[df_cond['Delta_bin'].notna() & df_cond['Vel_bin'].notna()]

        color_map = {
            f'Δ{i+1}': px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            for i in range(n_bins_delta)
        }

        hover_cols = [
            "ID", "RA", "Dec", "Vel", "Delta", "Cl_d",
            "C(index)", "M(C)", "(u-g)", "(g-r)", "M(u-g)",
            "M(g-r)", "Act"
        ]
        hover_data = {col: True for col in hover_cols}

        # === ✅ 2️⃣ FACET GRID ===
        #fig_faceted = px.scatter(
        #    df_cond,
        #    x="RA", y="Dec",
        #    facet_col="Delta_bin",
        #    facet_row="Vel_bin",
        #color="Delta_bin",
        #hover_name="ID",
        #    hover_data=hover_data,
        #    opacity=0.7,
        #    color_discrete_map=color_map
        #)

        # Asegúrate de no poner hover_name
        fig_faceted = px.scatter(
            df_cond,
            x="RA",
            y="Dec",
            facet_col="Delta_bin",
            facet_row="Vel_bin",
            color="Delta_bin",
            opacity=0.7,
            color_discrete_map=color_map,
        )

        # Añade customdata (se pasa como array 2D)
        for trace in fig_faceted.data:
            trace.customdata = df_cond[hover_cols].values
            trace.hovertemplate = (
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>RA:</b> %{customdata[1]:.4f}<br>" +
                "<b>Dec:</b> %{customdata[2]:.4f}<br>" +
                "<b>Vel:</b> %{customdata[3]:.2f}<br>" +
                "<b>Delta:</b> %{customdata[4]:.2f}<br>" +
                "<b>Cl_d:</b> %{customdata[5]:.2f}<br>" +
                "<b>C(index):</b> %{customdata[6]:.2f}<br>" +
                "<b>M(C):</b> %{customdata[7]}<br>" +
                "<b>(u-g):</b> %{customdata[8]:.2f}<br>" +
                "<b>(g-r):</b> %{customdata[9]:.2f}<br>" +
                "<b>M(u-g):</b> %{customdata[10]}<br>" +
                "<b>M(g-r):</b> %{customdata[11]}<br>" +
                "<b>Act:</b> %{customdata[12]}"
            )





        
        fig_contour = px.density_contour(
            df_cond,
            x="RA", y="Dec",
            facet_col="Delta_bin",
            facet_row="Vel_bin",
            color="Delta_bin",
            color_discrete_map=color_map
        )
        for trace in fig_contour.data:
            trace.showlegend = False
            trace.hoverinfo = "skip"
            fig_faceted.add_trace(trace)

        fig_faceted.update_xaxes(autorange="reversed")
        #fig_faceted.update_yaxes(autorange="reversed")
        fig_faceted.update_layout(showlegend=True)

        for trace in fig_faceted.data:
            if hasattr(trace, 'showscale'):
                trace.showscale = False
            if trace.type == "histogram2dcontour":
                trace.showlegend = False

        st.plotly_chart(fig_faceted, use_container_width=True)

        st.markdown("""
        <div style="text-align: justify;">
        <strong>✅ Checklist para Panel RA–Dec</strong><br>
        - Agrupamientos claros<br>
        - Filamentos o elongaciones<br>
        - Coinciden en rangos Vel<br>
        - Cruza morfología/actividad
        </div>
        """, unsafe_allow_html=True)

        # === ✅ 3️⃣ HISTOGRAMAS ===
        st.subheader("Distribución global de Delta")
        fig_hist_delta = px.histogram(
            df_cond, x="Delta",
            nbins=20, color="Delta_bin",
            opacity=0.7, color_discrete_map=color_map
        )
        st.plotly_chart(fig_hist_delta, use_container_width=True)

        st.subheader("Distribución global de Vel")
        fig_hist_vel = px.histogram(
            df_cond, x="Vel",
            nbins=20, color="Vel_bin",
            opacity=0.7
        )
        st.plotly_chart(fig_hist_vel, use_container_width=True)

        # === ✅ 4️⃣ PANEL INDIVIDUAL ===
        st.subheader("🔍 Explora cada panel RA–Dec individual")

        combinaciones = [
            (delta_bin, vel_bin)
            for delta_bin in df_cond['Delta_bin'].unique()
            for vel_bin in df_cond['Vel_bin'].unique()
        ]

        selected_comb = st.selectbox(
            "Selecciona un panel (Delta_bin, Vel_bin):",
            options=combinaciones,
            format_func=lambda x: f"Δ = {x[0]}  |  V = {x[1]}"
        )

        delta_sel, vel_sel = selected_comb
        df_panel = df_cond[
            (df_cond['Delta_bin'] == delta_sel) &
            (df_cond['Vel_bin'] == vel_sel)
        ]

        ra_min, ra_max = df_cond['RA'].min(), df_cond['RA'].max()
        dec_min, dec_max = df_cond['Dec'].min(), df_cond['Dec'].max()
        panel_color = color_map.get(delta_sel, "blue")

        # Usar go.Figure para hover enriquecido
        fig_panel = go.Figure()

        fig_panel.add_trace(go.Scatter(
            x=df_panel['RA'],
            y=df_panel['Dec'],
            mode="markers",
            marker=dict(size=6, color=panel_color),
            customdata=df_panel[hover_cols].values,
            hovertemplate="<br>".join([
                "ID: %{customdata[0]}",
                "RA: %{x}",
                "Dec: %{y}",
                "Vel: %{customdata[3]}",
                "Delta: %{customdata[4]}",
                "Cl_d: %{customdata[5]}",
                "C(index): %{customdata[6]}",
                "M(C): %{customdata[7]}",
                "(u-g): %{customdata[8]}",
                "(g-r): %{customdata[9]}",
                "M(u-g): %{customdata[10]}",
                "M(g-r): %{customdata[11]}",
                "Act: %{customdata[12]}"
            ])
        ))

        fig_panel.add_trace(go.Histogram2dContour(
            x=df_panel['RA'],
            y=df_panel['Dec'],
            ncontours=10,
            xbins=dict(size=0.05),  # Controla resolución RA
            ybins=dict(size=0.05),  # Controla resolución Dec
            #ncontours=10,
            colorscale=[[0, 'white'], [1, panel_color]],
            line=dict(width=1),
            hoverinfo="skip",
            showscale=False,
            contours=dict(showlines=True)
        ))

        fig_panel.update_xaxes(autorange="reversed", range=[ra_min, ra_max])
        fig_panel.update_yaxes(autorange="True", range=[dec_min, dec_max])
        fig_panel.update_layout(
            title=f"Panel RA–Dec: Δ = {delta_sel} | V = {vel_sel}",
            height=600,
            showlegend=False
        )

        for trace in fig_panel.data:
            if trace.type == "scatter":
                trace.marker.size = 12

        
        st.plotly_chart(fig_panel, use_container_width=True)


    
    with st.expander("🔍 Panel condicional Delta × Vel con KDE"):
        plot_conditional_panel(df)

    
    #else:
    #    st.warning("Selecciona al menos una columna numérica.")
else:
    st.info("Por favor, sube un archivo CSV.")
