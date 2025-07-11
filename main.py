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

# Menú en barra lateral
opcion = st.sidebar.radio(
    "Selecciona una pestaña:",
    ["Introducción", "Proceso", "Equipo de trabajo"])

# Contenido condicional
if opcion == "Introducción":
    st.subheader("Sobre la aplicación")

    st.markdown("""
<div style="text-align: justify">
La presente aplicación reúne conceptos y técnicas de 
<a href="https://en.wikipedia.org/wiki/Machine_learning" target="_blank"><b>Machine Learning</b></a> 
para el análisis de datos de cúmulos de galaxias. Los conceptos clave incluyen el uso de 
<a href="https://scikit-learn.org/stable/modules/clustering.html" target="_blank"><b>aprendizaje no supervisado</b></a>, 
como el <a href="https://en.wikipedia.org/wiki/Hierarchical_clustering" target="_blank"><b>clustering jerárquico</b></a>, 
para la clasificación de galaxias a partir de sus características, así como la formulación de reglas de decisión mediante 
<a href="https://scikit-learn.org/stable/modules/ensemble.html#random-forests" target="_blank"><b>aprendizaje supervisado</b></a> 
con algoritmos como <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank"><b>Random Forest</b></a>.<br><br>
Además, se implementan técnicas de 
<a href="https://en.wikipedia.org/wiki/Interpolation" target="_blank"><b>interpolación</b></a> 
para el ajuste de datos y herramientas de 
<a href="https://plotly.com/python/" target="_blank"><b>visualización interactiva</b></a> 
que permiten explorar mapas espaciales, diagramas de fase y gráficos de densidad para identificar posibles subestructuras y patrones dentro de los cúmulos galácticos.
</div>
""", unsafe_allow_html=True)


    st.subheader("Motivación")
    st.markdown("""
<div style="text-align: justify">
La motivación principal es mostrar cómo estas técnicas pueden integrarse en un algoritmo que permita extraer información de datos astronómicos de forma práctica y accesible. La aplicación sigue el principio de un 
<a href="https://en.wikipedia.org/wiki/Black_box" target="_blank"><b>método de caja negra</b></a>, donde la complejidad del proceso interno se abstrae para que el usuario obtenga resultados interpretables sin necesidad de conocer los detalles matemáticos o computacionales del modelo.
<br><br>
Además, se adopta el enfoque 
<a href="https://en.wikipedia.org/wiki/No-code_development_platform" target="_blank"><b>no-code</b></a>, facilitando que cualquier persona, sin experiencia previa en programación, pueda utilizar la herramienta, analizar resultados y extraer conclusiones por sí misma.
<br><br>
Finalmente, parte del código y su documentación fueron desarrollados mediante <strong>programación asistida por Inteligencia Artificial</strong>, integrando herramientas como 
<a href="https://openai.com/blog/chatgpt" target="_blank"><b>ChatGPT</b></a> para optimizar la escritura, estructuración y depuración del proyecto.
</div>
""", unsafe_allow_html=True)

    st.subheader("Datos")
    st.markdown("""
<div style="text-align: justify">
Los datos utilizados corresponden al cúmulo de galaxias Abell 85 y forman parte del trabajo de tesis doctoral del 
<a href="https://scholar.google.com/citations?hl=es&user=dvyLfnUAAAAJ" target="_blank"><b>Dr. Juan Manuel Islas Islas</b></a>, Doctor en Ciencias (Astrofísica), realizado en el 
<a href="https://www.astro.ugto.mx/" target="_blank"><b>Departamento de Astronomía</b></a> de la 
<a href="https://www.ugto.mx/" target="_blank"><b>Universidad de Guanajuato</b></a>.<br><br>
Adicionalmente, cada sección de la aplicación se ha diseñado para que, en caso de actualizar o sustituir los datos por los de otro cúmulo de galaxias, el análisis pueda realizarse de forma automática, siempre y cuando se mantenga la misma estructura de nombres de columnas.
</div>
""", unsafe_allow_html=True)


    st.subheader("Antecedentes")

    st.markdown("""
<div style="text-align: justify">
Las técnicas de clasificación presentadas en esta aplicación fueron aplicadas previamente durante la estancia posdoctoral 
<b>"Identificación de las etapas y tipos de sarcopenia mediante modelos predictivos como herramienta de apoyo en el diagnóstico a partir de parámetros antropométricos"</b>, 
desarrollada por el <a href="https://scholar.google.com/citations?user=SFgL-gkAAAAJ&hl=es" target="_blank"><b>Dr. Santiago Arceo Díaz</b></a>, Doctor en Ciencias (Astrofísica), 
bajo la dirección de la <b>Dra. Xóchitl Angélica Rosio Trujillo Trujillo</b>. Este trabajo representa un ejemplo de cómo la <b>colaboración interdisciplinaria</b> facilita el análisis de casos en los que técnicas avanzadas de procesamiento y análisis de datos se aplican en distintos contextos, promoviendo la transferencia de conocimiento entre áreas como la medicina, las matemáticas y el análisis de datos. La estancia fue posible gracias a la colaboración entre la 
<a href="https://secihti.mx/" target="_blank"><b>Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI)</b></a> (antes <b>CONAHCYT</b>) 
y la <a href="https://portal.ucol.mx/cuib/" target="_blank"><b>Universidad de Colima (UCOL)</b></a>. Derivado de este esfuerzo, se han publicado tres artículos científicos: 
<a href="https://www.researchgate.net/profile/Elena-Bricio-Barrios/publication/378476892_Inteligencia_Artificial_para_el_diagnostico_primario_de_dependencia_funcional_en_personas_adultas_mayores/links/65dbb4a0c3b52a1170f8658d/Inteligencia-Artificial-para-el-diagnostico-primario-de-dependencia-funcional-en-personas-adultas-mayores.pdf" target="_blank"><b>
"Inteligencia Artificial para el diagnóstico primario de dependencia funcional en personas adultas mayores"</b></a>, 
<a href="https://itchihuahua.mx/revista_electro/2024/A53_18-24.html" target="_blank"><b>
"Sistema Biomédico Basado en Inteligencia Artificial para Estimar Indirectamente Sarcopenia en Personas Adultas Mayores Mexicanas"</b></a> y 
<a href="https://scholar.google.com/citations?view_op=view_citation&hl=es&user=SFgL-gkAAAAJ&cstart=20&pagesize=80&citation_for_view=SFgL-gkAAAAJ:ldfaerwXgEUC" target="_blank"><b>
"Primary Screening System for Sarcopenia in Elderly People Based on Artificial Intelligence"</b></a>.
</div>
""", unsafe_allow_html=True)

    



elif opcion == "Proceso":

    # Cargar archivo CSV desde el usuario
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
#        with st.expander("🌌 Ver mapa interactivo del cúmulo Abell 85"):
#            import plotly.express as px
#            import plotly.graph_objects as go
#            import pandas as pd
#            import streamlit as st

#            # -------------------------------
#            # 🌌 Expansor con mapa interactivo
#            # -------------------------------
#            num_vars = ['Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)', 'Delta', 'Rf']
#            cat_vars = ['M(parn)', 'Act']
#            all_vars = num_vars + cat_vars
    
#            selected_var = st.selectbox(
#                "Variable para filtrar y colorear puntos:",
#                options=all_vars
#            )

#            df_filtered = df.copy()

#            # -------------------------------
#            # 📌 Filtrado con rangos o categorías
#            # -------------------------------
#            if selected_var in num_vars:
#                try:
#                    df_filtered['range_label'] = pd.qcut(df_filtered[selected_var], 5, duplicates='drop')
#                    labels = df_filtered['range_label'].cat.categories.astype(str).tolist()
#                    selected_labels = st.multiselect(
#                        "Selecciona uno o varios rangos:",
#                        options=labels,
#                        default=labels
#                    )
#                    df_filtered = df_filtered[df_filtered['range_label'].astype(str).isin(selected_labels)]
#                except Exception as e:
#                    st.warning(f"No se pudieron crear rangos para esta variable. Detalle: {e}")
#            elif selected_var in cat_vars:
#                labels = df_filtered[selected_var].dropna().unique().tolist()
#                selected_labels = st.multiselect(
#                    "Selecciona una o varias categorías:",
#                    options=labels,
#                    default=labels
#                )
#                df_filtered = df_filtered[df_filtered[selected_var].isin(selected_labels)]

#            # -------------------------------
#            # ✅ Hover enriquecido
#            # -------------------------------
#            hover_data = {
#                "RA": True, "Dec": True,
#                "Vel": True, "Cl_d": True, "Delta": True,
#                "(u-g)": True, "(g-r)": True, "(r-i)": True, "(i-z)": True,
 #               "M(IP)": True, "M(ave)": True, "Act": True
#            }

#            required_cols = {'RA', 'Dec', 'ID'} | set(hover_data.keys())

#            if required_cols.issubset(df.columns):
#                fig = px.scatter(
#                    df_filtered,
#                    x="RA",
#                    y="Dec",
#                    color=selected_var,
#                    color_continuous_scale='viridis',
#                    hover_name="ID",
#                    hover_data=hover_data,
#                    title=f"Mapa filtrado por: {selected_var}"
#                )

#                fig.add_trace(
#                    go.Histogram2dContour(
#                        x=df_filtered['RA'],
#                        y=df_filtered['Dec'],
#                        ncontours=10,
#                        colorscale='viridis',
#                        contours_coloring='lines',
#                        line_width=2,
#                        opacity=0.5,
#                        showscale=False,
#                        hoverinfo='skip'
#                    )    
#                )

#                fig.update_xaxes(autorange="reversed")
#                #fig.update_yaxes(autorange="reversed")
#                fig.update_xaxes(showgrid=False)  # Oculta solo las líneas horizontales
#                fig.update_yaxes(showgrid=False)  # Oculta solo las líneas horizontales

#                # -------------------------------
#                # ⭐️ Destacar N galaxias más brillantes o altas en la variable
#                # -------------------------------
#                st.write("Número de galaxias a destacar (por valor más extremo de la variable seleccionada):")
#                num_highlight = st.slider("Cantidad de galaxias destacadas", min_value=1, max_value=100, value=5)

#                if selected_var in num_vars:
#                    df_stars = df_filtered.nsmallest(num_highlight, 'Rf') if selected_var == 'Rf' else df_filtered.nlargest(num_highlight, selected_var)
#                else:
#                    df_stars = df_filtered.head(num_highlight)
#
#                for i, (_, star_row) in enumerate(df_stars.iterrows()):
#                    fig.add_trace(
#                        go.Scatter(
#                            x=[star_row['RA']],
#                            y=[star_row['Dec']],
#                            mode="markers+text",
#                            marker=dict(
#                                symbol="star",
#                                size=20,
#                                color="gold",
#                                line=dict(width=1, color="black")
#                            ),
#                            text=[str(i+1)],
#                            textposition="middle center",
#                            textfont=dict(color="black", size=10),
#                            name=f"Destacado {i+1}",
#                            legendgroup="Destacadas",
#                            showlegend=False,
#                            hovertemplate="<br>".join([
#                                f"ID: {star_row['ID']}",
#                                f"RA: {star_row['RA']:.5f}",
#                                f"Dec: {star_row['Dec']:.5f}",
#                                f"Vel: {star_row['Vel']}",
#                                f"Delta: {star_row['Delta']}",
#                                f"{selected_var}: {star_row[selected_var]}"
#                            ])
#                        )
#                    )

#                fig.update_layout(
#                    xaxis_title="Ascensión Recta (RA, grados)",
#                    yaxis_title="Declinación (Dec, grados)",
#                    height=700,
#                    width=900,
#                    #plot_bgcolor='gray',
#                    #paper_bgcolor='black',
#                    font=dict(color="white")
#                )
#                #fig.update_xaxes(showgrid=False)  # Oculta solo las líneas horizontales

#                st.plotly_chart(fig)

#                # -------------------------------
#                # 💾 Botones de descarga
#                # -------------------------------
#                st.download_button(
#                    "💾 Descargar tabla filtrada",
#                    df_filtered.to_csv(index=False).encode('utf-8'),
#                    file_name="galaxias_filtradas.csv",
#                    mime="text/csv"
#                )

#                if not df_stars.empty:
#                    st.download_button(
#                        "⭐️ Descargar tabla de galaxias destacadas",
#                        df_stars.to_csv(index=False).encode('utf-8'),
#                        file_name="galaxias_destacadas.csv",
#                        mime="text/csv"
#                    )

#            else:
#                st.warning(
#                    f"Faltan columnas necesarias para el mapa interactivo: "
#                    f"{required_cols - set(df.columns)}"
#                )

        # 🌌 Expansor con mapa interactivo del cúmulo Abell 85
        with st.expander("🌌 Ver mapa interactivo del cúmulo Abell 85"):
        # 🌌 Expansor con mapa interactivo del cúmulo Abell 85
        #with st.expander("🌌 Ver mapa interactivo del cúmulo Abell 85"):
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd

            num_vars = ['Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)', 'Delta', 'Rf']
            cat_vars = ['M(parn)', 'Act']
            all_vars = num_vars + cat_vars

            selected_var = st.selectbox(
                "Variable para filtrar y colorear puntos:",
                options=all_vars
            )

            df_filtered = df.copy()

            # 📌 Filtrado
            if selected_var in num_vars:
                try:
                    df_filtered['range_label'] = pd.qcut(df_filtered[selected_var], 5, duplicates='drop')
                    labels = df_filtered['range_label'].cat.categories.astype(str).tolist()
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

            hover_data = {
                "RA": True, "Dec": True,
                "Vel": True, "Cl_d": True, "Delta": True,
                "(u-g)": True, "(g-r)": True, "(r-i)": True, "(i-z)": True,
                "M(IP)": True, "M(ave)": True, "Act": True
            }

            required_cols = {'RA', 'Dec', 'ID'} | set(hover_data.keys())

            if required_cols.issubset(df.columns):
                fig = px.scatter(
                    df_filtered,
                    x="RA",
                    y="Dec",
                    color=selected_var,
                    color_continuous_scale='plasma',
                    hover_name="ID",
                    hover_data=hover_data,
                    title=f"Mapa filtrado por: {selected_var}"
                )

                # ===============================================
                # 2️⃣ Agrega contornos KDE (Plotly density_contour)
                kde_contours = px.density_contour(
                    df_filtered,
                    x="RA",
                    y="Dec",
                    nbinsx=20,
                    nbinsy=20,
                )
                kde_contours.update_traces(
                    contours_coloring="lines",
                    line_width=2,
                    showscale=False
                )

                
                # Añade cada traza de contorno KDE al scatter principal
                for trace in kde_contours.data:
                    fig.add_trace(trace)

                
                # ⭐ y 💎 sliders
                st.write("Número de galaxias a destacar por variable seleccionada:")
                num_extreme = st.slider("Cantidad de galaxias extremas", min_value=1, max_value=100, value=5)

                st.write("Número de galaxias a destacar por brillo (Rf):")
                num_bright = st.slider("Cantidad de galaxias brillantes", min_value=1, max_value=100, value=5)

                df_extreme = (
                    df_filtered.nsmallest(num_extreme, 'Rf') if selected_var == 'Rf'
                    else df_filtered.nlargest(num_extreme, selected_var)
                ).copy()
                df_bright = df_filtered.nsmallest(num_bright, 'Rf').copy()

                # ⭐ Colores adaptativos
                custom_colors = []
                if len(df_extreme) == 1:
                    custom_colors = ['gold']
                elif len(df_extreme) == 2:
                    custom_colors = ['gold', 'silver']
                elif len(df_extreme) == 3:
                    custom_colors = ['gold', 'silver', '#cd7f32']
                elif len(df_extreme) >= 4:
                    df_extreme = df_extreme.sort_values(by=selected_var, ascending=False).reset_index(drop=True)
                    df_extreme['cuartil'] = pd.qcut(
                        df_extreme[selected_var],
                        q=4,
                        labels=[4, 3, 2, 1]
                    ).astype(int)
                    quartile_colors = {
                        1: 'gold',
                        2: 'silver',
                        3: '#cd7f32',
                        4: 'lightskyblue'
                    }
                    custom_colors = [quartile_colors[q] for q in df_extreme['cuartil']]

                # ⭐ + 💎 combinado con numeraciones dobles
                ids_extreme = df_extreme['ID'].tolist()
                ids_bright = df_bright['ID'].tolist()

                for _, row in df_filtered.iterrows():
                    is_extreme = row['ID'] in ids_extreme
                    is_bright = row['ID'] in ids_bright

                    labels = []
                    if is_extreme:
                        idx_ext = ids_extreme.index(row['ID']) + 1
                        labels.append(f"{idx_ext}")
                        color = custom_colors[idx_ext - 1] if idx_ext - 1 < len(custom_colors) else 'gold'
                        fig.add_trace(
                            go.Scatter(
                                x=[row['RA']],
                                y=[row['Dec']],
                                mode="markers+text",
                                marker=dict(symbol="star", size=24, color=color, line=dict(width=1, color="black")),
                                text=[labels[-1]],
                                textposition="middle center",
                                textfont=dict(color="black", size=10),
                                showlegend=False
                            )
                        )

                    if is_bright:
                        idx_bright = ids_bright.index(row['ID']) + 1
                        labels.append(f"B{idx_bright}")
                        fig.add_trace(
                            go.Scatter(
                                x=[row['RA']],
                                y=[row['Dec']],
                                mode="markers+text",
                                marker=dict(symbol="diamond", size=18, color="deepskyblue", line=dict(width=1, color="black")),
                                text=[labels[-1]],
                                textposition="middle center",
                                textfont=dict(color="black", size=10),
                                showlegend=False
                            )
                        )

                fig.update_layout(
                    xaxis_title="Ascensión Recta (RA, grados)",
                    yaxis_title="Declinación (Dec, grados)",
                    height=700,
                    width=900,
                    font=dict(color="black"),
                    legend_title="Destacadas"
                )
                fig.update_xaxes(autorange="reversed")   # 🗺️ Invierte RA (Ascensión Recta)

                st.plotly_chart(fig)

                st.download_button(
                    "💾 Descargar tabla filtrada",
                    df_filtered.to_csv(index=False).encode('utf-8'),
                    file_name="galaxias_filtradas.csv",
                    mime="text/csv"
                )

                if not df_extreme.empty or not df_bright.empty:
                    df_combined = pd.concat([df_extreme, df_bright]).drop_duplicates().reset_index(drop=True)
                    st.download_button(
                        "⭐️ Descargar tabla de galaxias destacadas",
                        df_combined.to_csv(index=False).encode('utf-8'),
                        file_name="galaxias_destacadas.csv",
                        mime="text/csv"
                    )
            else:
                st.warning(
                    f"Faltan columnas necesarias para el mapa interactivo: "
                    f"{required_cols - set(df.columns)}"
                )


            # ✅ Librerías necesarias
            #import numpy as np
            #import streamlit as st
            #import plotly.graph_objects as go
           # from scipy.stats import gaussian_kde
           # from statsmodels.nonparametric.kernel_density import KDEMultivariate

            # ✅ Encabezado
# ✅ Librerías
            import numpy as np
            import streamlit as st
            import plotly.graph_objects as go
            from scipy.stats import gaussian_kde
            from statsmodels.nonparametric.kernel_density import KDEMultivariate
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from pykrige.ok import OrdinaryKriging

            # ✅ Título
            st.subheader("🗺️ Mapa suavizado 2D (KDE + RF + Kriging) con corrección para magnitudes")

            # ✅ Selección de variable
            smooth_var = st.selectbox(
                "Variable para mapa suavizado:",
                options=['Delta', 'Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)', 'Rf'],
                index=0
            )

            df_smooth = df_filtered[df_filtered[smooth_var].notna()]
            if df_smooth.empty:
                st.warning("No hay datos válidos.")
                st.stop()

            # ✅ Config
            method = st.radio("Método:", ["KDE fijo", "KDE adaptativo", "Random Forest", "Kriging"])
            bw = st.slider("Ancho de banda KDE:", 0.1, 2.0, 0.3, step=0.05)
            grid_size = st.slider("Resolución malla:", 50, 500, 200, step=50)
            cmap = st.selectbox("Colormap:", ["viridis", "plasma", "magma", "cividis"])
            use_log = st.toggle("Contornos logarítmicos", value=True)

            # ✅ Datos base
            ra = df_smooth['RA'].values
            dec = df_smooth['Dec'].values
    
            # ⚡️ Corrección de signo para magnitudes
            z = df_smooth[smooth_var].values
            if smooth_var == 'Rf':
                z = -1 * z  # Invertir: más alto = más brillante

            # Asegura valores positivos para KDE
            z_weights = np.abs(z)

            # ✅ Crear malla
            xi, yi = np.mgrid[ra.min():ra.max():grid_size*1j,
                              dec.min():dec.max():grid_size*1j]

            # ✅ Calcular zi según método
            if method == "KDE fijo":
                kde = gaussian_kde(np.vstack([ra, dec]), weights=z_weights, bw_method=bw)
                zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
            elif method == "KDE adaptativo":
                kde = KDEMultivariate(data=[ra, dec], var_type='cc', bw=[bw, bw])
                zi = kde.pdf(np.vstack([xi.ravel(), yi.ravel()]))
            elif method == "Random Forest":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X = np.vstack([ra, dec]).T
                X_scaled = scaler_X.fit_transform(X)
                z_scaled = scaler_y.fit_transform(z.reshape(-1, 1)).ravel()

                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X_scaled, z_scaled)

                X_grid = np.vstack([xi.ravel(), yi.ravel()]).T
                X_grid_scaled = scaler_X.transform(X_grid)
                zi = rf.predict(X_grid_scaled)
                zi = scaler_y.inverse_transform(zi.reshape(-1, 1)).ravel()
            elif method == "Kriging":
                OK = OrdinaryKriging(ra, dec, z, variogram_model="linear")
                zi, ss = OK.execute('grid', xi[0], yi[:,0])
                zi = zi.ravel()

            # ✅ Mismo reshape
            zi = np.reshape(zi, xi.shape)

            # ✅ Log si aplica
            if use_log:
                zi = np.log1p(zi - zi.min())  # evita log(0) o negativos

            # ✅ Graficar con Plotly
            fig = go.Figure()
    
            fig.add_trace(go.Contour(
                z=zi,
                x=xi[:,0],
                y=yi[0],
                contours=dict(coloring='lines', showlabels=True),
                colorscale=cmap,
                showscale=True,
                line_width=2
            ))

#            fig.add_trace(go.Scatter(
#                x=ra,
#                y=dec,
#                mode='markers',
#                marker=dict(
#                    size=6,
#                    color=z_weights,
#                    colorscale=cmap,
#                    showscale=False,
#                    line=dict(width=0.5, color='black')
#                ),
#                hovertemplate="<br>".join([
#                    "RA: %{x:.3f}",
#                    "Dec: %{y:.3f}",
#                    f"{smooth_var}: %{marker.color:.3f}"
#                ])
#            ))

            fig.add_trace(go.Scatter(
                x=ra,
                y=dec,
                mode='markers',
                marker=dict(
                    size=6,
                    color=z_weights,
                    colorscale=cmap,
                    showscale=False,
                    line=dict(width=0.5, color='black')
                ),
                hovertemplate="<br>".join([
                    "RA: %{x:.3f}",
                    "Dec: %{y:.3f}",
                    f"{smooth_var}: %{{marker.color:.3f}}"
                ])
            ))
            
            fig.update_layout(
                title=f"{method} • {'Log' if use_log else 'Lineal'} • {smooth_var}",
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                xaxis=dict(autorange="reversed"),
                template='plotly_white',
                height=700,
                width=900
            )

            st.plotly_chart(fig, use_container_width=True)

            # ✅ Descargar tabla usada
            with st.expander("📄 Ver tabla"):
                st.dataframe(df_smooth)
                st.download_button(
                    "💾 Descargar tabla",
                    df_smooth.to_csv(index=False).encode('utf-8'),
                    file_name="suavizado.csv",
                    mime="text/csv"
                )




            




            import numpy as np
            import plotly.express as px
            from astropy.cosmology import FlatLambdaCDM

            st.subheader("🌌 Mapa 3D comóvil interactivo para Abell 85")

            # ✅ Controles para parámetros arbitrarios
            st.markdown("### ⚙️ Parámetros de cosmología y proyección")

            col1, col2, col3 = st.columns(3)

            H0 = col1.number_input("H₀ (km/s/Mpc)", value=70.0, min_value=50.0, max_value=80.0, step=0.5)
            Om0 = col2.number_input("Ωₘ", value=0.3, min_value=0.0, max_value=1.0, step=0.01)
            Ode0 = col3.number_input("ΩΛ", value=0.7, min_value=0.0, max_value=1.0, step=0.01)

            z_cluster = st.number_input("Redshift de referencia (z_cluster)", value=0.0555, step=0.001)

            # Centro de proyección
            ra0_default = float(df_filtered['RA'].mean()) if not df_filtered.empty else 0.0    
            dec0_default = float(df_filtered['Dec'].mean()) if not df_filtered.empty else 0.0

            ra0 = st.number_input("Centro RA₀ (°)", value=ra0_default, step=0.1)
            dec0 = st.number_input("Centro Dec₀ (°)", value=dec0_default, step=0.1)

            # ✅ Usar DataFrame filtrado
            df_3d = df_filtered.copy()

            if df_3d.empty:
                st.warning("No hay galaxias para mostrar en 3D con los filtros actuales.")
            else:
                # Cosmología
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                c = 3e5  # km/s

                # Calcular z_gal y distancia comóvil
                df_3d['z_gal'] = z_cluster + (df_3d['Vel'] / c) * (1 + z_cluster)
                df_3d['D_C'] = cosmo.comoving_distance(df_3d['z_gal']).value  # Mpc

                # Coordenadas comóviles
                df_3d['X'] = df_3d['D_C'] * np.cos(np.radians(df_3d['Dec'])) * np.cos(np.radians(df_3d['RA'] - ra0))
                df_3d['Y'] = df_3d['D_C'] * np.cos(np.radians(df_3d['Dec'])) * np.sin(np.radians(df_3d['RA'] - ra0))
                df_3d['Z'] = df_3d['D_C'] * np.sin(np.radians(df_3d['Dec'] - dec0))

                # Hover enriquecido
                hover_3d = ["ID", "RA", "Dec", "Vel", "Delta", "Cl_d",
                    "(u-g)", "(g-r)", "(r-i)", "(i-z)", "Act"]

                # Usar rangos si existen
                group_col = None
                if 'range_label' in df_3d.columns:
                    group_col = 'range_label'
                elif 'Subcluster' in df_3d.columns:
                    group_col = 'Subcluster'
                elif 'Delta_cat' in df_3d.columns:
                    group_col = 'Delta_cat'
                elif selected_var in cat_vars:
                    group_col = selected_var

                if group_col:
                    fig_3d = px.scatter_3d(
                        df_3d,
                        x='X', y='Y', z='Z',
                        color=group_col,
                        hover_data=hover_3d,
                        opacity=0.7,
                        title=f"Mapa 3D comóvil agrupado por {group_col}"
                    )
                else:
                    fig_3d = px.scatter_3d(
                        df_3d,
                        x='X', y='Y', z='Z',
                        color='Delta',
                        hover_data=hover_3d,
                        opacity=0.7,
                        color_continuous_scale='Viridis',
                        title="Mapa 3D comóvil de Abell 85"
                    )

                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="X [Mpc]",
                        yaxis_title="Y [Mpc]",
                        zaxis_title="Z [Mpc]"
                    ),
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=40)
                )

                st.plotly_chart(fig_3d, use_container_width=True)

                st.markdown("""
                <div style="text-align: justify;">
                <strong>Nota:</strong><br>
                Este mapa proyecta la estructura tridimensional comóvil usando los parámetros de cosmología y redshift que se elijan. Los valores pueden variar ligeramente dependiendo de la elección de H₀, Ωₘ, ΩΛ y centro de proyección RA₀/Dec₀. Los movimientos peculiares de las galaxias pueden estirar o comprimir la estructura a lo largo de la línea de visión (efecto Fingers of God).<br>
                </div>
                """, unsafe_allow_html=True)


    
    
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
        import plotly.express as px
        from astropy.cosmology import FlatLambdaCDM

        st.subheader("🌌 Mapa 3D comóvil por Subestructura y Variable")

        # ✅ Cosmología y centro interactivos
        col1, col2, col3 = st.columns(3)
        H0_3d = col1.number_input("H0 (km/s/Mpc)", value=70.0, min_value=50.0, max_value=80.0, step=0.5)
        Om0_3d = col2.number_input("Ωm", value=0.3, min_value=0.0, max_value=1.0, step=0.01)
        z_cluster_3d = col3.number_input("Redshift para referencia", value=0.0555, step=0.001)

        ra0_default_3d = float(df['RA'].mean()) if not df.empty else 0.0
        dec0_default_3d = float(df['Dec'].mean()) if not df.empty else 0.0

        ra0_3d = st.number_input("Centro RA (°)", value=ra0_default_3d, step=0.1)
        dec0_3d = st.number_input("Centro Dec (°)", value=dec0_default_3d, step=0.1)

        # ✅ Filtrado de Subestructura
        df_sub = df[df['Subcluster'].notna()].copy()
        if df_sub.empty:
            st.warning("No hay subestructuras detectadas.")
        else:
            unique_subs = sorted(df_sub['Subcluster'].unique())
            selected_subs = st.multiselect(
                "Selecciona una o varias Subestructuras:",
                options=unique_subs,
                default=unique_subs
            )

            df_3d = df_sub[df_sub['Subcluster'].isin(selected_subs)].copy()

            # ✅ Variable numérica para rangos
            num_vars_3d = ['Vel', 'Delta', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)']
            var_selected_3d = st.selectbox("Variable que agrupar rangos:", options=num_vars_3d, index=1)

            # Crear rangos dinámicos
            n_bins_3d = st.slider("Número de rangos a usar:", min_value=2, max_value=6, value=4)
            df_3d['Var_bin'] = pd.qcut(df_3d[var_selected_3d], q=n_bins_3d, duplicates='drop')
            df_3d['Var_bin_str'] = df_3d['Var_bin'].astype(str)

            # Cosmología
            cosmo = FlatLambdaCDM(H0=H0_3d, Om0=Om0_3d)
            c = 3e5  # km/s

            # Calcular z_gal y D_C
            df_3d['z_gal'] = z_cluster_3d + (df_3d['Vel'] / c) * (1 + z_cluster_3d)
            df_3d['D_C'] = cosmo.comoving_distance(df_3d['z_gal']).value  # Mpc

            # Coordenadas comóviles
            df_3d['X'] = df_3d['D_C'] * np.cos(np.radians(df_3d['Dec'])) * np.cos(np.radians(df_3d['RA'] - ra0_3d))
            df_3d['Y'] = df_3d['D_C'] * np.cos(np.radians(df_3d['Dec'])) * np.sin(np.radians(df_3d['RA'] - ra0_3d))
            df_3d['Z'] = df_3d['D_C'] * np.sin(np.radians(df_3d['Dec'] - dec0_3d))

            # Tamaño basado en rango (más alto → mayor)
            bin_map = {k: i+1 for i, k in enumerate(sorted(df_3d['Var_bin_str'].unique()))}
            df_3d['Size'] = df_3d['Var_bin_str'].map(lambda x: 4 + bin_map[x]*3)

            # Hover enriquecido
            hover_3d = ["ID", "Subcluster", "RA", "Dec", "Vel", "Delta", "Cl_d",
                    "(u-g)", "(g-r)", "(r-i)", "(i-z)", "Act"]

            fig_3d = px.scatter_3d(
                df_3d,
                x='X', y='Y', z='Z',
                color='Var_bin_str',
                size='Size',
                hover_data=hover_3d,
                opacity=0.7,
                title=f"Mapa 3D comóvil | Subestructura(s): {selected_subs} | Rango: {var_selected_3d}",
            )

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="X [Mpc]",
                    yaxis_title="Y [Mpc]",
                    zaxis_title="Z [Mpc]"
                ),
                height=800,
                margin=dict(l=0, r=0, b=0, t=50)
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown(f"""
            <div style="text-align: justify;">
            <strong> Cómo interpretarlo:</strong><br>
            - Cada punto es una galaxia en coordenadas comóviles X, Y, Z.<br>
            - El color indica el rango de {var_selected_3d}.<br>
            - El tamaño refleja su posición dentro del rango: más grande = rango más alto.<br>
            - Puedes girar, acercar y ocultar rangos usando la leyenda.<br>
            </div>
            """, unsafe_allow_html=True)


    
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
            #xbins=dict(size=0.1),  # Controla resolución RA
            #ybins=dict(size=0.1),  # Controla resolución Dec
            #ncontours=10,
                colorscale=[[0, 'white'], [1, panel_color]],
                line=dict(width=1),
                hoverinfo="skip",
                showscale=False,
                contours=dict(showlines=True)
            ))

            fig_panel.update_xaxes(autorange="reversed", range=[ra_min, ra_max])
            fig_panel.update_yaxes(autorange=True, range=[dec_min, dec_max])
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

elif opcion == "Equipo de trabajo":
    st.subheader("Equipo de Trabajo")

       # Información del equipo
    equipo = [{
               "nombre": "Dr. Santiago Arceo Díaz",
               "foto": "ArceoS.jpg",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima. Miembro del Sistema Nacional de Investigadoras e Investigadores (Nivel 1).",
               "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "santiagoarceodiaz@gmail.com"}
               ]

    for persona in equipo:
        st.image(persona["foto"], width=150)
    
        st.markdown(f"""
        <h4 style='text-align: left;'>{persona['nombre']}</h4>
        <p style='text-align: justify;'>{persona['reseña']}</p>
        <p style='text-align: left;'>
            📄 <a href="{persona['CV']}" target="_blank">Ver CV</a><br>
            📧 {persona['contacto']}
        </p>
        <hr style='margin-top: 10px; margin-bottom: 20px;'>
        """, unsafe_allow_html=True)

    

    # Información de contacto
    st.subheader("Información de Contacto")
    st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a santiagoarceodiaz@gmail.com")
