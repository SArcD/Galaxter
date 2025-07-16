
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


#    st.subheader("Antecedentes")

#    st.markdown("""
#<div style="text-align: justify">
#Las técnicas de clasificación presentadas en esta aplicación fueron aplicadas previamente durante la estancia posdoctoral 
#<b>"Identificación de las etapas y tipos de sarcopenia mediante modelos predictivos como herramienta de apoyo en el diagnóstico a partir de parámetros antropométricos"</b>, 
#desarrollada por el <a href="https://scholar.google.com/citations?user=SFgL-gkAAAAJ&hl=es" target="_blank"><b>Dr. Santiago Arceo Díaz</b></a>, Doctor en Ciencias (Astrofísica), 
#bajo la dirección de la <b>Dra. Xóchitl Angélica Rosio Trujillo Trujillo</b>. Este trabajo representa un ejemplo de cómo la <b>colaboración interdisciplinaria</b> facilita el análisis de casos en los que técnicas avanzadas de procesamiento y análisis de datos se aplican en distintos contextos, promoviendo la transferencia de conocimiento entre áreas como la medicina, las matemáticas y el análisis de datos. La estancia fue posible gracias a la colaboración entre la 
#<a href="https://secihti.mx/" target="_blank"><b>Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI)</b></a> (antes <b>CONAHCYT</b>) 
#y la <a href="https://portal.ucol.mx/cuib/" target="_blank"><b>Universidad de Colima (UCOL)</b></a>. Derivado de este esfuerzo, se han publicado tres artículos científicos: 
#<a href="https://www.researchgate.net/profile/Elena-Bricio-Barrios/publication/378476892_Inteligencia_Artificial_para_el_diagnostico_primario_de_dependencia_funcional_en_personas_adultas_mayores/links/65dbb4a0c3b52a1170f8658d/Inteligencia-Artificial-para-el-diagnostico-primario-de-dependencia-funcional-en-personas-adultas-mayores.pdf" target="_blank"><b>
#"Inteligencia Artificial para el diagnóstico primario de dependencia funcional en personas adultas mayores"</b></a>, 
#<a href="https://itchihuahua.mx/revista_electro/2024/A53_18-24.html" target="_blank"><b>
#"Sistema Biomédico Basado en Inteligencia Artificial para Estimar Indirectamente Sarcopenia en Personas Adultas Mayores Mexicanas"</b></a> y 
#<a href="https://scholar.google.com/citations?view_op=view_citation&hl=es&user=SFgL-gkAAAAJ&cstart=20&pagesize=80&citation_for_view=SFgL-gkAAAAJ:ldfaerwXgEUC" target="_blank"><b>
#"Primary Screening System for Sarcopenia in Elderly People Based on Artificial Intelligence"</b></a>.
#</div>
#""", unsafe_allow_html=True)

    



elif opcion == "Proceso":
    st.markdown("""
<div style="text-align: justify">
Por favor cargue un archivo .csv con la base de datos (versión actual del archivo: "Tabla_mainA85_b.csv".
</div>
""", unsafe_allow_html=True)
    # Cargar archivo CSV desde el usuario
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

    if uploaded_file is not None:
        # Leer archivo
        df = pd.read_csv(uploaded_file)
        st.write("Datos cargados:", df.head())
        # Expansor con explicaciones
        with st.expander("**Ver explicación de las columnas**"):
            # Descripciones
            descripcion_columnas = {
            'SDSS': 'Nombre del catálogo Sloan Digital Sky Survey del objeto.',
            'ID': 'Identificador único del objeto, usualmente coordenadas codificadas.',
            'RA': 'Ascensión recta (Right Ascension) en grados (coordenada celeste este-oeste).',
            'Dec': 'Declinación (Declination) en grados (coordenada celeste norte-sur).',
            'Vel': 'Velocidad radial del objeto en km/s, indica movimiento relativo al observador.',
            'Rf': 'Magnitud en el "rojo" (posiblemente magnitud fotométrica corregida).',
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
        st.subheader("Buscar variable por su clave")
        st.markdown("""
<div style="text-align: justify">
En esta sección puede colocar el nombre de cualquiera de las columnas de la base de datos para ver una reseña breve de su significado (nota: aún necesita revisarse) </div>
""", unsafe_allow_html=True)
        search_query = st.text_input("**Escribe parte del nombre de la variable:**", key="var_search_desc")

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

        with st.expander("**Análisis exploratorio: Distribución univariada, bivariada, correlación entre las varirables y modelos predictivos.**"):
            st.subheader("Distribución univariada de una variable numérica")

            # Lista de columnas numéricas en tu DataFrame
            numeric_colss = df.select_dtypes(include='number').columns.tolist()

            import plotly.express as px
            import streamlit as st
            import difflib
            import random


            # Paleta de colores opcional
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"
            ]

            search_vars = st.multiselect(
                "**Selecciona una o más variables numéricas para mostrar histogramas (puede manipular o guardar la gráfica con los botones que aparecen si coloca el cursor en la parte superior derecha):**",
                options=numeric_colss, default=numeric_colss[:2]
            )

            if search_vars:
                for idx, col in enumerate(search_vars):
                    st.subheader(f"Histograma de: **{col}**")

                    # Slider para definir el número de bins
                    nbins = st.slider(
                        f"Número de bins para {col}",
                        min_value=5,
                        max_value=100,
                        value=30,
                        key=f"slider_{col}"
                    )

                    # Escoge color único para este histograma
                    color = colors[idx % len(colors)]

                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=nbins,
                        title=f"Distribución de {col}"
                    )

                    # Aplica color único + borde negro
                    fig.update_traces(marker=dict(
                        color=color,
                        line=dict(color="black", width=1)
                    ))

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Selecciona al menos una variable numérica para visualizar su histograma.")

            st.divider()

            st.subheader("Pair Plot de variables numéricas")

            # Multiselect para elegir variables para el pair plot
            selected_pair_cols = st.multiselect(
                "**Selecciona al menos dos variables para ver su gráfico de pares**",
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
                st.info("Selecciona al menos dos variables.")


            import plotly.express as px
            import plotly.graph_objects as go
            import statsmodels.api as sm
            import streamlit as st

            import streamlit as st
            import plotly.graph_objects as go
            import plotly.subplots as sp
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            import numpy as np

            import streamlit as st
            import plotly.graph_objects as go
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            import numpy as np

            st.markdown("**Seleccione una variable predictora y al menos una variable a predecir. Debajo verá un dos gráficos de dispersión en los que se muestran dos ajustes: lineal y Random Forest. En cada caso se despliega también el coeficiente de determinación**")
            # Solo variables numéricas para los selectores
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            x_var = st.selectbox("Variable X", numeric_cols)
            y_vars = st.multiselect("Variables Y", numeric_cols)

            if x_var and y_vars:
                for idx, y_var in enumerate(y_vars):

                    X = df[[x_var]].values.astype(float)
                    Y = df[y_var].values.astype(float)

                    # Elimina NaNs
                    mask = ~np.isnan(X).flatten() & ~np.isnan(Y)
                    X = X[mask].reshape(-1, 1)
                    Y = Y[mask]

                    if len(Y) < 2:
                        st.warning(f"Sin datos suficientes para {y_var}")
                        continue

                    # Ajuste lineal
                    lin_model = LinearRegression().fit(X, Y)
                    Y_pred = lin_model.predict(X)

                    # Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100).fit(X, Y)
                    Y_rf_pred = rf_model.predict(X)

                    # Colores únicos
                    color = f"hsl({idx * 60 % 360}, 70%, 50%)"

                    # Hover: todas las columnas
                    hover_texts = []
                    df_valid = df[mask]  # Filtrado por mask
                    for i in range(len(df_valid)):
                        row_values = [f"<b>{col}:</b> {df_valid.iloc[i][col]}" for col in df.columns]
                        hover_texts.append("<br>".join(row_values))

                    # Subplots
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ajuste Lineal", "Random Forest"))

                    # Scatter
                    fig.add_trace(go.Scatter(
                        x=X.flatten(),
                        y=Y,
                        mode='markers',
                        marker=dict(color=color, size=7, line=dict(width=0.5, color='black')),
                        text=hover_texts,
                        hovertemplate='%{text}',
                        name=f'{y_var} vs {x_var}'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=X.flatten(),
                        y=Y_pred,
                        mode='lines',
                        line=dict(color='black', dash='dash'),
                        name='Ajuste Lineal',
                        hoverinfo='skip'
                    ), row=1, col=1)


                    # RF Scatter
                    fig.add_trace(go.Scatter(
                        x=X.flatten(),
                        y=Y,
                        mode='markers',
                        marker=dict(color=color, size=7, line=dict(width=0.5, color='black')),
                        text=hover_texts,
                        hovertemplate='%{text}',
                        name=f'{y_var} vs {x_var} (RF)'
                    ), row=1, col=2)

                    fig.add_trace(go.Scatter(
                        x=X.flatten(),
                        y=Y_rf_pred,
                        mode='lines',
                        line=dict(color='green', dash='dot'),
                        name='Random Forest',
                        hoverinfo='skip'
                    ), row=1, col=2)



                    # Ecuación y R² para lineal
                    slope = lin_model.coef_[0]
                    intercept = lin_model.intercept_
                    r_squared = lin_model.score(X, Y)
                    eq_text = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.3f}"

                    eq_text = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.3f}"
                    fig.add_annotation(
                        xref="x", yref="y",
                        x=0.95, y=0.95,
                        text=eq_text,
                        showarrow=False,
                        align="right",
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        row=1, col=1   # ☑️ Solo para el primer subplot
                    )

                    from sklearn.metrics import mean_squared_error
                    r2_rf = rf_model.score(X, Y)
                    from sklearn.metrics import mean_squared_error
                    import numpy as np

                    # MSE normal:
                    mse_rf = mean_squared_error(Y, Y_rf_pred)

                    # RMSE es la raíz cuadrada:
                    rmse_rf = np.sqrt(mse_rf)
                    #rmse_rf = mean_squared_error(Y, Y_rf_pred, squared=False)
                    rf_text = f"R² = {r2_rf:.3f}<br>RMSE = {rmse_rf:.3f}"
                    #rf_text = f"R² = {r2_rf:.3f}<br>RMSE = {rmse_rf:.3f}"

                    fig.add_annotation(
                        xref="x2", yref="y2",
                        x=0.95, y=0.95,
                        text=rf_text,
                        showarrow=False,
                        align="right",
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        row=1, col=2   # Solo para el subplot RF
                    )
                    
                    fig.update_layout(height=500, title=f"{y_var} vs {x_var} - Lineal vs RF")
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()


            import streamlit as st
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            import numpy as np

            # Solo variables numéricas para selector
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            st.header("Ajuste por rango de variable X")
            st.markdown("**Seleccione una variable predictora y al menos una variable a predecir. Debajo verá un dos gráficos de dispersión en los que se muestran dos ajustes: lineal y Random Forest. En cada caso se despliega también el coeficiente de determinación**")
            # Selectores de variable X e Y
            x_var_range = st.selectbox("Variable X para filtrar rango", numeric_cols, key="x_var_range")
            y_vars_range = st.multiselect("Variables Y", numeric_cols, key="y_vars_range")

            if x_var_range and y_vars_range:
                x_min, x_max = float(df[x_var_range].min()), float(df[x_var_range].max())

                # 🎚️ Slider de rango
                range_values = st.slider(
                    f"Selecciona rango de {x_var_range}",
                    min_value=x_min, max_value=x_max,
                    value=(x_min, x_max),
                    step=(x_max - x_min) / 100
                )

                mask_range = (df[x_var_range] >= range_values[0]) & (df[x_var_range] <= range_values[1])

                df_range = df[mask_range].copy()

                if df_range.empty:
                    st.warning("No hay datos en este rango.")
                else:
                    for idx, y_var in enumerate(y_vars_range):
                        X = df_range[[x_var_range]].values.astype(float)
                        Y = df_range[y_var].values.astype(float)

                        # Ajuste lineal
                        lin_model = LinearRegression().fit(X, Y)
                        Y_pred = lin_model.predict(X)

                        # Random Forest
                        rf_model = RandomForestRegressor(n_estimators=100).fit(X, Y)
                        Y_rf_pred = rf_model.predict(X)

                        # Hover con TODAS las columnas
                        hover_texts = []
                        for i in range(len(df_range)):
                            row_values = [f"<b>{col}:</b> {df_range.iloc[i][col]}" for col in df.columns]
                            hover_texts.append("<br>".join(row_values))

                        # Subplots lado a lado
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("Ajuste Lineal", "Random Forest"))

                        color = f"hsl({idx * 60 % 360}, 70%, 50%)"

                        # Scatter y línea Lineal
                        fig.add_trace(go.Scatter(
                            x=X.flatten(), y=Y,
                            mode='markers',
                            marker=dict(color=color, size=7, line=dict(width=0.5, color='black')),
                            text=hover_texts,
                            hovertemplate='%{text}',
                            name=f'{y_var} vs {x_var_range}'
                        ), row=1, col=1)

                        fig.add_trace(go.Scatter(
                            x=X.flatten(), y=Y_pred,
                            mode='lines',
                            line=dict(color='black', dash='dash'),
                            name='Ajuste Lineal',
                            hoverinfo='skip'
                        ), row=1, col=1)

                        # Scatter y línea RF
                        fig.add_trace(go.Scatter(
                            x=X.flatten(), y=Y,
                            mode='markers',
                            marker=dict(color=color, size=7, line=dict(width=0.5, color='black')),
                            text=hover_texts,
                            hovertemplate='%{text}',
                            name=f'{y_var} vs {x_var_range} (RF)'
                        ), row=1, col=2)

                        fig.add_trace(go.Scatter(
                            x=X.flatten(), y=Y_rf_pred,
                            mode='lines',
                            line=dict(color='green', dash='dot'),
                            name='Random Forest',
                            hoverinfo='skip'
                        ), row=1, col=2)

                        # Métricas Lineal
                        slope = lin_model.coef_[0]
                        intercept = lin_model.intercept_
                        r_squared = lin_model.score(X, Y)
                        eq_text = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.3f}"

                        fig.add_annotation(
                            xref="paper", yref="paper",
                            x=0.99, y=0.99,
                            text=eq_text,
                            showarrow=False,
                            align="right",
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            xanchor="right",
                            row=1, col=1
                        )

                        # Métricas RF
                        r2_rf = rf_model.score(X, Y)
                        #rmse_rf = mean_squared_error(Y, Y_rf_pred, squared=False)
                        import numpy as np
                        mse_rf = mean_squared_error(Y, Y_rf_pred)
                        rmse_rf = np.sqrt(mse_rf)
                        rf_text = f"R² = {r2_rf:.3f}<br>RMSE = {rmse_rf:.3f}"

                        fig.add_annotation(
                            xref="paper", yref="paper",
                            x=0.99, y=0.99,
                            text=rf_text,
                            showarrow=False,
                            align="right",
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            xanchor="right",
                            row=1, col=2
                        )

                        fig.update_layout(height=500, title=f"{y_var} vs {x_var_range} en rango [{range_values[0]:.2f}, {range_values[1]:.2f}]")
                        st.plotly_chart(fig, use_container_width=True)

            

            
            st.divider()

            import streamlit as st    
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            # -------------------------------------------------------------
            # 2 variables predictoras -> 1 target + RF + superficie 3D
            # -------------------------------------------------------------

            st.header("Random Forest: Modelos con dos variables predictoras")
            st.markdown("**Seleccione dos variables predictoras y una variable a predecir, debajo verá desplegado, a partir de un modelo de Random Forest: un gráfico tridimensional con los puntos de scater y la superficie de predicción y, debajo de eso, una superficie de decisión (debajo del selector de variables pude definir la resolución de la malla).**")
            # Variables numéricas
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            # Selectores
            x1_var = st.selectbox("Variable X1", numeric_cols, key="rf_x1")
            x2_var = st.selectbox("Variable X2", numeric_cols, key="rf_x2")
            y_var = st.selectbox("Variable Target (Y)", numeric_cols, key="rf_y")

            if x1_var and x2_var and y_var:
                X1 = df[[x1_var]].values.flatten()
                X2 = df[[x2_var]].values.flatten()
                Y = df[[y_var]].values.flatten()

                # Quita NaNs
                mask = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(Y)
                X1, X2, Y = X1[mask], X2[mask], Y[mask]

                X = np.column_stack((X1, X2))

                if len(Y) < 5:
                    st.warning("No hay suficientes datos después del filtrado.")
                else:
                    # Random Forest
                    rf_model = RandomForestRegressor(n_estimators=200)
                    rf_model.fit(X, Y)
                    Y_pred = rf_model.predict(X)

                    # Métricas
                    r2_rf = rf_model.score(X, Y)
                    from sklearn.metrics import mean_squared_error
                    rmse_rf = np.sqrt(mean_squared_error(Y, Y_pred))

                    # Malla para superficie
#                    grid_size = 30
                    grid_size = st.slider("Resolución de la malla", 10, 100, 30, step=5, key="rf_grid_size")

                    x1_range = np.linspace(X1.min(), X1.max(), grid_size)
                    x2_range = np.linspace(X2.min(), X2.max(), grid_size)
                    xx1, xx2 = np.meshgrid(x1_range, x2_range)
                    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
                    zz_pred = rf_model.predict(grid_points).reshape(xx1.shape)

                    # Hover personalizado con TODAS las columnas        
                    hover_texts = []
                    df_valid = df[mask]
                    for i in range(len(df_valid)):
                        row_values = [f"<b>{col}:</b> {df_valid.iloc[i][col]}" for col in df.columns]
                        hover_texts.append("<br>".join(row_values))

                    # 🎥 Gráfico Plotly 3D
                    fig = go.Figure()

                    # Puntos reales
                    fig.add_trace(go.Scatter3d(
                        x=X1,
                        y=X2,
                        z=Y,
                        mode='markers',
                        marker=dict(size=5, color='black'),
                        text=hover_texts,
                        hovertemplate='%{text}',
                        name='Datos Reales'
                    ))

                    # Superficie RF
                    fig.add_trace(go.Surface(
                        x=x1_range,
                        y=x2_range,
                        z=zz_pred,
                        colorscale='viridis',
                        opacity=0.7,
                        name='Superficie RF'
                    ))

                    fig.update_layout(
                        title=f"RF Superficie de Predicción 3D<br>R² = {r2_rf:.3f} | RMSE = {rmse_rf:.3f}",
                        scene=dict(
                            xaxis_title=x1_var,
                            yaxis_title=x2_var,
                            zaxis_title=y_var
                        ),
                        height=700
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Controles interactivos para invertir ejes
            invert_x = st.checkbox(f"Invertir eje {x1_var}", value=False, key="invert_x_rf")
            invert_y = st.checkbox(f"Invertir eje {x2_var}", value=False, key="invert_y_rf")

            # Gráfico Plotly 2D    
            fig2d = go.Figure()

            # Heatmap de la predicción
            fig2d.add_trace(go.Heatmap(
                x=x1_range,
                y=x2_range,
                z=zz_pred,
                colorscale='viridis',
                colorbar=dict(title=y_var),
                name='Predicción RF',
                showscale=True
            ))

            # ⚫ Puntos reales encima
            fig2d.add_trace(go.Scatter(
                x=X1,
                y=X2,
                mode='markers',
                marker=dict(size=6, color='black', line=dict(width=0.5, color='white')),
                text=hover_texts,
                hovertemplate='%{text}',
                name='Datos Reales'
            ))

            # ➕ Layout básico
            fig2d.update_layout(
                title=f"RF Superficie de Predicción 2D<br>R² = {r2_rf:.3f} | RMSE = {rmse_rf:.3f}",
                xaxis_title=x1_var,
                yaxis_title=x2_var,
                height=600
            )

            # Aplica inversión si corresponde    
            if invert_x:
                fig2d.update_xaxes(autorange='reversed')

            if invert_y:
                fig2d.update_yaxes(autorange='reversed')

            # Despliega
            st.plotly_chart(fig2d, use_container_width=True)


            # ----------------------------------------------------------
            # Formulario para predicción puntual con rangos definidos
            # ----------------------------------------------------------
            st.subheader("Predicción puntual con Random Forest")
            st.markdown("**Use los deslizadores para definir los valores de los predictores y debajo verá el valor estimado por el modelo de Random Forest.**")
            # Rango dinámico basado en los datos filtrados
            x1_min, x1_max = X1.min(), X1.max()
            x2_min, x2_max = X2.min(), X2.max()
    
            # 🕹️ Sliders con keys únicos y contextuales
            input_x1 = st.slider(
                f"Seleccione {x1_var}",
                float(x1_min), float(x1_max), float(np.mean(X1)),
                key=f"slider_{x1_var}_rf_input_s"
            )

            input_x2 = st.slider(
                f"Seleccion {x2_var}",
                float(x2_min), float(x2_max), float(np.mean(X2)),
                key=f"slider_{x2_var}_random"
            )

            # Construye input para predicción    
            input_array = np.array([[input_x1, input_x2]])
            predicted_y = rf_model.predict(input_array)[0]

            st.success(f"Predicción de **{y_var}** para {x1_var} = {input_x1:.2f} y {x2_var} = {input_x2:.2f}: **{predicted_y:.2f}**")

            # Opcional: guarda en lista, descarga o agrega historial
            if st.button("**Guardar predicción**"):
                st.write("Guardado: ", {
                    f"{x1_var}": input_x1,
                    f"{x2_var}": input_x2,
                    f"{y_var}_predicho": predicted_y
                })

            st.divider()

            import streamlit as st
            import numpy as np
            import pandas as pd
            from collections import Counter
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, learning_curve
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.utils import resample
            from imblearn.over_sampling import SMOTE
            import plotly.figure_factory as ff
            import plotly.graph_objects as go

            st.header("Clasificación de morfología con Random Forest + SMOTE")


            st.markdown("**Aquí puede usar las variables numéricas para predecir morfología usando modelos de Random Forest (se recomienda activar el botón de SMOTE, si considera que las clases morfolóficas están desbalanceadas). Use los botones y deslizadores para definir los predictores, la profundidad de los modelos de árbol en Random Forest y el número de vecinos (si usa SMOTE).**")
            
            # Variables numéricas como predictores
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            # Variable objetivo
            target_var = st.selectbox(
                "Variable objetivo categórica",
                ["M(ave)", "M(C)"],
                key="rf_class_target"
            )

            # Variables predictoras
            feature_vars = st.multiselect(
                "Variables numéricas predictoras",
                numeric_cols,
                default=numeric_cols
            )

            # Hiperparámetros
            max_depth = st.slider("Profundidad máxima del árbol", 1, 20, 5)
            n_estimators = st.slider("Número de árboles", 10, 500, 200, step=10)

            # Opción SMOTE
            use_smote = st.checkbox("Aplicar SMOTE para balancear clases", value=False)

            if target_var and feature_vars:
                X = df[feature_vars].values
                y = df[target_var].astype(str).values

                mask = ~np.isnan(X).any(axis=1) & pd.notna(y)
                X, y = X[mask], y[mask]

                if len(y) < 5:
                    st.warning("No hay suficientes datos después del filtrado.")
                else:
                    # SMOTE con filtro de clases demasiado pequeñas
                    if use_smote:
                        class_counts = Counter(y)
                        too_small_classes = [cls for cls, cnt in class_counts.items() if cnt < 2]

                        if too_small_classes:
                            st.warning(f"⚠️ Clases ignoradas porque tienen <2 muestras: {too_small_classes}")
                            mask_keep = ~np.isin(y, too_small_classes)
                            X, y = X[mask_keep], y[mask_keep]

                        class_counts = Counter(y)
                        min_samples = min(class_counts.values())

                        if min_samples < 2:
                            st.warning("⚠️ No hay clases con suficientes muestras para aplicar SMOTE.")
                        else:
                            safe_k = max(1, min_samples - 1)
                            k_neighbors = st.slider("k_neighbors para SMOTE", 1, safe_k, min(5, safe_k))
                            sm = SMOTE(k_neighbors=k_neighbors, random_state=42)
                            X, y = sm.fit_resample(X, y)
                            st.success(f"✔️ Clases balanceadas con SMOTE: {dict(pd.Series(y).value_counts())}")

                    # Random Forest
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
                    clf.fit(X, y)
                    y_pred = clf.predict(X)
                    accuracy = accuracy_score(y, y_pred)

                    st.write(f"✅ **Exactitud (Entrenamiento):** {accuracy:.3f}")
                    st.text(classification_report(y, y_pred))

                    # Validación cruzada
                    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                    st.success(f"📊 **Accuracy CV promedio:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

                    # Matriz de confusión
                    cm = confusion_matrix(y, y_pred)
                    cm_fig = ff.create_annotated_heatmap(
                        z=cm,
                        x=list(clf.classes_),
                        y=list(clf.classes_),
                        colorscale="blues"
                    )
                    cm_fig.update_layout(title="Matriz de Confusión (entrenamiento)")
                    st.plotly_chart(cm_fig, use_container_width=True)
                    st.markdown("**Muestra la cantidad de predicciones correctas (en la diagonal) vs las incorrectas (celdas fuera de la diagonal).**")
                    # Curva de aprendizaje
                    train_sizes, train_scores, test_scores = learning_curve(
                        clf, X, y, cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 5)
                    )
                    train_mean = np.mean(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)

                    curve_fig = go.Figure()
                    curve_fig.add_trace(go.Scatter(
                        x=train_sizes, y=train_mean,
                        mode='lines+markers', name='Training Score'
                    ))
                    curve_fig.add_trace(go.Scatter(
                        x=train_sizes, y=test_mean,
                        mode='lines+markers', name='Cross-Validation Score'
                    ))
                    curve_fig.update_layout(
                        title="📈 Curva de Aprendizaje Random Forest",
                        xaxis_title="Tamaño del conjunto de entrenamiento",
                        yaxis_title="Accuracy",
                        height=400
                    )
                    st.plotly_chart(curve_fig, use_container_width=True)

                    st.markdown("**Muestra la diferencia de precisión entre el modelo de Random Forest y la validadción cruzada.**")
                    
                    st.info("Revisa la curva: Si hay brecha grande entre entrenamiento y validación, puede haber sobreajuste.")

                    # Formulario de predicción
                    st.subheader("Hacer una predicción")
                    input_vals = []
                    for feat in feature_vars:
                        val = st.slider(
                            f"{feat}",
                            float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()),
                            key=f"slider_{feat}_class"
                        )
                        input_vals.append(val)

                    if st.button("Predecir clase"):
                        pred_class = clf.predict([input_vals])[0]
                        proba = clf.predict_proba([input_vals])[0]
                        proba_dict = dict(zip(clf.classes_, proba))

                        st.write(f"**Clase predicha:** {pred_class}")
                        st.write("**Probabilidades (modelo base):**")
                        st.write(proba_dict)

                        # Bootstrap para incertidumbre
                        st.subheader("Incertidumbre con Bootstrap")
                        num_bootstrap = st.slider("Número de bootstraps", 50, 500, 100, 50, key="bootstrap_rf_class")
                        bootstrap_probas = []

                        for _ in range(num_bootstrap):
                            X_res, y_res = resample(X, y, replace=True)
                            clf_b = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=None
                            )
                            clf_b.fit(X_res, y_res)
                            proba_b = clf_b.predict_proba([input_vals])[0]
                            classes_b = clf_b.classes_

                            proba_full = []
                            for cls in clf.classes_:
                                if cls in classes_b:
                                    idx = np.where(classes_b == cls)[0][0]
                                    proba_full.append(proba_b[idx])
                                else:
                                    proba_full.append(0.0)

                            bootstrap_probas.append(proba_full)

                        bootstrap_probas = np.array(bootstrap_probas)
                        proba_mean = bootstrap_probas.mean(axis=0)
                        proba_std = bootstrap_probas.std(axis=0)

                        results = pd.DataFrame({
                            "Clase": clf.classes_,
                            "Probabilidad media": proba_mean,
                            "Desviación estándar": proba_std
                        })
                        st.write("**Distribución bootstrap de la predicción:**")
                        st.dataframe(results)

                        st.subheader("Distribución Bootstrap de Probabilidades")
                        for idx, class_label in enumerate(clf.classes_):
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=bootstrap_probas[:, idx],
                                nbinsx=20,
                                marker_color='blue'
                            ))
                            fig.update_layout(
                                title=f"Distribución Bootstrap para clase: {class_label}",
                                xaxis_title="Probabilidad predicha",
                                yaxis_title="Frecuencia",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)


            st.divider()
            
            st.subheader("Matriz de correlación")
            st.markdown("**Muestra la matriz de correlación entre las variables consideradas en la base de datos.**")
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


        # Expansor con mapa interactivo del cúmulo Abell 85
        with st.expander("**Mapas interactivos**"):
            import plotly.express as px
            import plotly.graph_objects as go
            import pandas as pd
            st.markdown("Seleccione una de las variables numéricas para grear un mapa de calor de esta variable. Adicionalmente, puede seleccionar solo galaxias en alguno de los quintiles, definir el número de galaxias que quiere marcar (hasta 100 con los valores mas grandes para la variable seleccionada representadas por las estrellas numeradas) y si desea mostrar las galaxias mas brillantes (hasta 100, representadas por los diamantes numerados).")
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
                    color_continuous_scale='viridis',
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

                
                # sliders
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
                    unique_vals = df_extreme[selected_var].nunique()

                    if unique_vals >= 4:
                        df_extreme['cuartil'] = pd.qcut(
                            df_extreme[selected_var],
                            q=4,
                            labels=[4, 3, 2, 1]
                        ).astype(int)
                    else:
                        # Si no hay suficientes valores únicos para 4 cuartiles, asigna 1 a todos
                        df_extreme['cuartil'] = 1

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
                        "Descargar tabla de galaxias destacadas",
                        df_combined.to_csv(index=False).encode('utf-8'),
                        file_name="galaxias_destacadas.csv",
                        mime="text/csv"
                    )
            else:
                st.warning(
                    f"Faltan columnas necesarias para el mapa interactivo: "
                    f"{required_cols - set(df.columns)}"
                )

            import numpy as np        
            import streamlit as st
            import plotly.graph_objects as go
            from scipy.stats import gaussian_kde
            from statsmodels.nonparametric.kernel_density import KDEMultivariate

            # ✅ Encabezado
            st.subheader("Mapa KDE Gaussiano")
            st.markdown("Se muestra un mapa continuo generado a partir de un perfil de densidad Gaussiano. Puede seleccionar el ancho de banda, el mapa de color y la resolución de la malla  (**nota: aún necesita revisarse**)")
            
            # ✅ Variables disponibles
            smooth_var = st.selectbox(
                "Variable para mapa suavizado:",
                options=['Delta', 'Vel', 'Cl_d', '(u-g)', '(g-r)', '(r-i)', '(i-z)'],
                index=0
            )

            # ✅ Datos filtrados válidos
            df_smooth = df_filtered[df_filtered[smooth_var].notna()]
            if df_smooth.empty:
                st.warning("No hay datos válidos para suavizar.")
                st.stop()

            # ✅ Configuración interactiva
            kde_type = st.radio("Tipo de KDE:", ["Fijo (gaussian_kde)", "Adaptativo (KDEMultivariate)"])
            bw = st.slider("Ajuste de ancho de banda:", 0.1, 2.0, 0.3, step=0.05)
            use_log = st.toggle("Usar escala logarítmica para contornos", value=True)
            cmap = st.selectbox("Colormap:", ["viridis", "plasma", "magma", "cividis"])
            grid_size = st.slider("Resolución de la malla:", 50, 500, 200, step=50)

            # ✅ Variables para malla
            ra = df_smooth['RA'].values
            dec = df_smooth['Dec'].values
            weights = df_smooth[smooth_var].values

            # Peso inteligente corregido
            if smooth_var == 'Rf':
                weights = np.max(weights) - weights + np.min(weights) + 1e-6
            elif smooth_var == 'Cl_d':
                pass  # NO invertir: pesos tal cual
            else:
                pass  # Otros índices, igual

            weights = np.clip(weights, 1e-6, None)  # Evita ceros
            weights = weights / np.sum(weights)     # Normaliza robusto     
            
            xi, yi = np.mgrid[ra.min():ra.max():grid_size*1j, dec.min():dec.max():grid_size*1j]

            # KDE fijo o adaptativo
            if kde_type.startswith("Fijo"):
                kde = gaussian_kde(np.vstack([ra, dec]), weights=weights, bw_method=bw)
                zi = kde(np.vstack([xi.ravel(), yi.ravel()]))    
            else:
                kde = KDEMultivariate(data=[ra, dec], var_type='cc', bw=[bw, bw])
                zi = kde.pdf(np.vstack([xi.ravel(), yi.ravel()]))

            zi = np.reshape(zi, xi.shape)
            if use_log:
                zi = np.log1p(zi)

            # Gráfico interactivo
            fig = go.Figure()


            fig.add_trace(go.Contour(
                z=zi,
                x=xi[:,0],
                y=yi[0],
                contours=dict(
                    coloring='heatmap',   # Superficie coloreada
                    showlabels=True
                ),
                colorscale=cmap,
                showscale=True,
                line_width=1
            ))

            # Puntos originales con hover robusto
            fig.add_trace(go.Scatter(
                x=ra,
                y=dec,
                mode='markers',
                marker=dict(
                    size=6,
                    color=weights,
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
                title=f"KDE {'Adaptativo' if kde_type.startswith('Adaptativo') else 'Fijo'} • Escala {'Log' if use_log else 'Lineal'} • {smooth_var}",
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                xaxis=dict(autorange="reversed"),
                template='plotly_white',
                height=700,
                width=900
            )

            st.plotly_chart(fig, use_container_width=True)

            # Tabla y exportación
            with st.expander("**Ver datos suavizados**"):
                st.dataframe(df_smooth)
                st.download_button(
                    "Descargar tabla usada",
                    df_smooth.to_csv(index=False).encode('utf-8'),
                    file_name="datos_suavizados.csv",
                    mime="text/csv"
                )

            import numpy as np
            import plotly.express as px
            from astropy.cosmology import FlatLambdaCDM

            st.subheader("Mapa 3D comóvil interactivo")

            # Controles para parámetros arbitrarios
            st.markdown("## Parámetros de cosmología y proyección")

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






        import numpy as np
        import pandas as pd
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial import KDTree
        from tqdm import tqdm

        # ================================================
        # ✅ FUNCIÓN: Clustering jerárquico
        # ================================================
        def run_hierarchical_clustering(df, selected_cols, num_clusters):
            scaler = StandardScaler()
            data_clean = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
            scaled_data = scaler.fit_transform(data_clean)
            Z = linkage(scaled_data, method='ward')
            labels = fcluster(Z, t=num_clusters, criterion='maxclust')
            df.loc[data_clean.index, 'Subcluster'] = labels
            return df, Z, scaled_data, data_clean.index


        # ================================================    
        # ✅ 3️⃣ FUNCIÓN: Gráfico t-SNE + Boxplots por Subcluster
        # ================================================
        def plot_tsne_and_boxplots(df, data_idx, selected_cols):
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            unique_clusters = sorted(df.loc[data_idx, 'Subcluster'].dropna().unique())
            vars_phys = selected_cols
            n_cols = 3
            n_rows = (len(vars_phys) + n_cols - 1) // n_cols
            total_rows = n_rows + 1

            specs = [[{"colspan": n_cols}] + [None]*(n_cols-1)]
            for _ in range(n_rows):
                specs.append([{} for _ in range(n_cols)])

            subplot_titles = ["PCA + t-SNE Clustering"] + vars_phys
            fig = make_subplots(rows=total_rows, cols=n_cols, specs=specs, subplot_titles=subplot_titles)
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

            for i, cluster in enumerate(unique_clusters):
                cluster_data = df.loc[data_idx][df.loc[data_idx, 'Subcluster'] == cluster]
                hover_text = ("<b>ID:</b> " + cluster_data['ID'].astype(str) +
                              "<br><b>RA:</b> " + cluster_data['RA'].round(4).astype(str) +
                              "<br><b>Dec:</b> " + cluster_data['Dec'].round(4).astype(str))
                fig.add_trace(
                    go.Scatter(x=cluster_data['TSNE1'], y=cluster_data['TSNE2'],
                               mode='markers', name=f'Subcluster {cluster}',
                               legendgroup=f'Subcluster {cluster}',
                               marker=dict(size=6, color=colors[i % len(colors)],
                                           line=dict(width=1, color='DarkSlateGrey')),
                               text=hover_text, hoverinfo='text'),
                    row=1, col=1)

            fig.add_trace(
                go.Histogram2dContour(
                    x=df.loc[data_idx, 'TSNE1'],
                    y=df.loc[data_idx, 'TSNE2'],
                    colorscale='Greys',
                    reversescale=True,
                    opacity=0.2,
                    showscale=False,
                    hoverinfo='skip'),
                row=1, col=1)

            for idx, var in enumerate(vars_phys):
                row = (idx // n_cols) + 2
                col = (idx % n_cols) + 1
                for j, cluster in enumerate(unique_clusters):
                    cluster_data = df.loc[data_idx][df.loc[data_idx, 'Subcluster'] == cluster]
                    fig.add_trace(
                        go.Box(y=cluster_data[var],
                               x=[f'Subcluster {cluster}'] * len(cluster_data),
                               name=f'Subcluster {cluster}',
                               legendgroup=f'Subcluster {cluster}',
                               showlegend=False, boxpoints='all',
                               jitter=0.5, pointpos=-1.8,
                               marker_color=colors[j % len(colors)],
                               notched=True, width=0.6),
                        row=row, col=col)

            fig.update_layout(
                title="Panel interactivo: PCA + t-SNE + Boxplots",
                template='plotly_white',
                width=400 * n_cols,
                height=400 * total_rows,
                boxmode='group',
                legend_title="Subclusters"
            )

            return fig




        
        # ================================================    
        # ✅ FUNCIÓN: t-SNE
        # ================================================
        def add_tsne(df, scaled_data, idx):
            pca = PCA(n_components=min(20, scaled_data.shape[1])).fit_transform(scaled_data)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(pca)
            df.loc[idx, 'TSNE1'] = tsne_result[:, 0]
            df.loc[idx, 'TSNE2'] = tsne_result[:, 1]
            return df

        # ================================================
        # ✅ FUNCIÓN: DS TEST para subcluster
        # ================================================
        def run_ds(df_sub, n_permutations=500):
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
                d_i = ((N + 1) / sigma_global**2) * ((local_vel - V_global)**2 + (local_sigma - sigma_global)**2)
                delta.append(np.sqrt(d_i))
            df_sub['Delta'] = delta
            DS_stat_real = np.sum(delta)
            DS_stats_permuted = []
            for _ in range(n_permutations):
                velocities_perm = np.random.permutation(velocities)
                delta_perm = []
                for i, neighbors in enumerate(neighbors_idx):
                    local_vel = np.mean(velocities_perm[neighbors])
                    local_sigma = np.std(velocities_perm[neighbors])
                    d_i = ((N + 1) / sigma_global**2) * ((local_vel - V_global)**2 + (local_sigma - sigma_global)**2)
                    delta_perm.append(np.sqrt(d_i))
                DS_stats_permuted.append(np.sum(delta_perm))
            p_value = np.sum(np.array(DS_stats_permuted) >= DS_stat_real) / n_permutations
            return df_sub, DS_stats_permuted, DS_stat_real, p_value

        # ================================================
        # ✅ FUNCIÓN: Histograma DS
        # ================================================
        def plot_ds_histogram(DS_stats_permuted, DS_stat_real):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=DS_stats_permuted, nbinsx=30, marker_color='lightgrey'))
            fig.add_trace(go.Scatter(x=[DS_stat_real, DS_stat_real],
                                     y=[0, np.histogram(DS_stats_permuted, bins=30)[0].max()],
                                     mode='lines', line=dict(color='red', dash='dash', width=3)))
            fig.update_layout(title="DS: Distribución nula vs. Δ real",
                              xaxis_title='Δ', yaxis_title='Frecuencia',
                              template='plotly_white')
            return fig

        # ================================================
        # ✅ FUNCIÓN: Mapas RA-Dec + DS
        # ================================================
        def plot_ra_dec_ds(df_sub):
            color_map = {
                'Bajo': '#1f77b4',
                'Medio': '#2ca02c',
                'Alto': '#ff7f0e',
                'Muy Alto': '#d62728'
            }
            bins = [0, 1, 2, 3, 5]
            labels = ['Bajo', 'Medio', 'Alto', 'Muy Alto']
            df_sub['Delta_cat'] = pd.cut(df_sub['Delta'], bins=bins, labels=labels)
            df_sub['color'] = df_sub['Delta_cat'].map(color_map)
            fig = go.Figure()
            for cat, color in color_map.items():
                sub_cat = df_sub[df_sub['Delta_cat'] == cat]
                fig.add_trace(go.Scatter(
                    x=sub_cat['RA'],
                    y=sub_cat['Dec'],
                    mode='markers',
                    name=f'Delta: {cat}',
                    marker=dict(size=8, color=color),
                    hovertext=sub_cat.apply(
                        lambda row: f"ID: {row['ID']}<br>RA: {row['RA']:.3f}<br>Dec: {row['Dec']:.3f}<br>Vel: {row['Vel']:.1f}<br>Δ: {row['Delta']:.3f} ({row['Delta_cat']})",
                        axis=1),
                    hoverinfo='text'))
            fig.add_trace(go.Histogram2dContour(
                x=df_sub['RA'],
                y=df_sub['Dec'],
                colorscale='plasma',
                reversescale=True,
                showscale=False,
                opacity=0.3,
                ncontours=15))
            fig.update_layout(
                title="Mapa RA-Dec por Delta",
                xaxis_title="RA",
                yaxis_title="Dec",
                xaxis=dict(autorange='reversed'),
                template='plotly_white')
            return fig



        
        import plotly.graph_objects as go
        import plotly.express as px

        def plot_final_map_subclusters(df):
            """
            Muestra mapa RA-Dec para todos los subclusters con KDE + hover completo.
            """
            if 'Subcluster' not in df.columns:
                st.warning("No se encontró la columna 'Subcluster'. Ejecuta primero el clustering.")
                return

            df_plot = df[df['Subcluster'].notna()].copy()
            if df_plot.empty:
                st.info("No hay subclusters asignados para mostrar.")
                return

            unique_clusters = sorted(df_plot['Subcluster'].unique())
            colores = px.colors.qualitative.Set2

            fig = go.Figure()

            for i, cluster in enumerate(unique_clusters):
                cl_data = df_plot[df_plot['Subcluster'] == cluster].copy()
                color = colores[i % len(colores)]
                cl_str = f"Subcluster {cluster}"

                # Genera hover detallado con TODAS las columnas
                hover_text = cl_data.apply(
                    lambda row: "<br>".join([f"{col}: {row[col]}" for col in cl_data.columns]),
                    axis=1
                )

                # Puntos
                fig.add_trace(go.Scatter(
                    x=cl_data['RA'],
                    y=cl_data['Dec'],
                    mode='markers',
                    marker=dict(size=6, color=color, line=dict(width=0.5, color='DarkSlateGrey')),
                    name=cl_str,
                    legendgroup=cl_str,
                    text=hover_text,
                    hoverinfo='text'
                ))

                # Contorno KDE
                fig.add_trace(go.Histogram2dContour(
                    x=cl_data['RA'],
                    y=cl_data['Dec'],
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, color]],
                    showscale=False,
                    opacity=0.3,
                    ncontours=10,
                    line=dict(width=1),
                    name=f"{cl_str} Contorno",
                    legendgroup=cl_str,
                    hoverinfo='skip',
                    showlegend=False  # Para que solo la traza principal aparezca en leyenda
                ))

            fig.update_layout(
                title="Mapa Abell 85 por Subestructura • RA–Dec + KDE",
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                legend_title="Subclusters",
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(autorange="reversed"),
                template='plotly_white',
                height=800, width=1000
            )

            st.plotly_chart(fig, use_container_width=True)
 
        
        # ================================================
        # ✅ FUNCIÓN: Procesar TODO AUTOMÁTICO
        # ================================================
        def full_pipeline(df, selected_cols, num_clusters, n_permutations=500):
            df, Z, scaled_data, idx = run_hierarchical_clustering(df, selected_cols, num_clusters)
            df = add_tsne(df, scaled_data, idx)
            tsne_fig = plot_tsne_and_boxplots(df, idx, selected_cols)
            st.plotly_chart(tsne_fig)
            unique_clusters = sorted(df.loc[idx, 'Subcluster'].dropna().unique())
            for sub in unique_clusters:
                df_sub = df[df['Subcluster'] == sub].copy()
                if df_sub.shape[0] < 5:
                    continue
                df_sub, DS_stats_permuted, DS_stat_real, p_value = run_ds(df_sub, n_permutations)
                st.write(f"Subcluster {sub}: p-valor DS = {p_value:.4f}")
                hist_fig = plot_ds_histogram(DS_stats_permuted, DS_stat_real)
                st.plotly_chart(hist_fig)
                map_fig = plot_ra_dec_ds(df_sub)
                st.plotly_chart(map_fig)
                st.download_button(f"Descargar tabla Subcluster {sub}",
                                   df_sub.to_csv(index=False).encode('utf-8'),
                                   file_name=f"DS_Subcluster_{sub}.csv",
                                   mime="text/csv")

            return df

        with st.expander("**Clustering Jerárquico automatizado**"):
            st.markdown("En esta sección se aplica clustering jerárquico aglomerativo a la base de datos. Puede seleccionar las variables para crear los subclusters y también su número. Después de seleccionarlos, verá desplegado el mapa de dispersión de los subclusters, los diagramas de caja donde se comparan en sus variables numéricas, la prueba DS para cada uno y, por último, el mapa con las galaxias ya separadas por subcluster (puede ocultar aquellos que no pasan la prueba DS simplemente dando click en su etiqueta). ")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            # Define tus columnas preferidas:
            default_cols = [col for col in ["Vel", "Delta", "Cl_d", "RA", "Dec"] if col in numeric_cols]

            selected_cols = st.multiselect(
                "Variables numéricas:",
                numeric_cols,
                default=default_cols
            )
            
            
            num_clusters = st.slider("Número de clusters:", 2, 10, 4)
            df = full_pipeline(df, selected_cols, num_clusters)
            # Y muestra el mapa final interactivo:
            plot_final_map_subclusters(df)


        
    
        import numpy as np    
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial import KDTree
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st


        def run_subclustering_iterative(df, parent_col, selected_cols, num_clusters, level):
            child_col = f'Subcluster_{level}'

            all_labels = []

            parents = sorted(df[parent_col].dropna().unique())
            scaler = StandardScaler()

            for parent in parents:
                df_parent = df[df[parent_col] == parent].copy()
                if df_parent.shape[0] < num_clusters:
                    continue  # no tiene suficientes galaxias para subdividir

                data = df_parent[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if data.empty:
                    continue
        
                scaled_data = scaler.fit_transform(data)
                Z = linkage(scaled_data, method='ward')
                labels = fcluster(Z, t=num_clusters, criterion='maxclust')

                # etiqueta compuesta: padre_hijo
                combined_labels = [f"{int(parent)}_{label}" for label in labels]

                df.loc[data.index, child_col] = combined_labels
                all_labels.extend(combined_labels)

            st.info(f"✅ Subclusters creados a nivel {level}: {len(set(all_labels))}")
            return df



        
        # ================================
        # 📌 Dressler–Shectman (DS) test
        # ================================
        def run_ds_iterative(df, cluster_col, level, n_permutations=500, alpha=0.05):
            delta_col = f'Delta_{level}'
            cat_col = f'Delta_cat_{level}'
            pass_col = f'DS_Pass_{level}'

            df[delta_col] = np.nan
            df[cat_col] = np.nan
            df[pass_col] = np.nan

            passed = []
            clusters = df[cluster_col].dropna().unique()

            for cluster in clusters:
                df_sub = df[df[cluster_col] == cluster].copy()
                if df_sub.shape[0] < 5:
                    continue

                coords = df_sub[['RA', 'Dec']].values
                velocities = df_sub['Vel'].values

                N = int(np.sqrt(len(coords)))
                tree = KDTree(coords)
                neighbors_idx = [tree.query(coords[i], k=N+1)[1][1:] for i in range(len(coords))]

                Vg, sg = np.mean(velocities), np.std(velocities)
                delta = []
                for i, n in enumerate(neighbors_idx):
                    lv, ls = np.mean(velocities[n]), np.std(velocities[n])
                    d_i = ((N+1)/sg**2) * ((lv-Vg)**2 + (ls-sg)**2)
                    delta.append(np.sqrt(d_i))
                df_sub[delta_col] = delta
                DS_real = np.sum(delta)

                DS_permuted = []
                for _ in range(n_permutations):
                    v_perm = np.random.permutation(velocities)
                    dp = []
                    for i, n in enumerate(neighbors_idx):
                        lv, ls = np.mean(v_perm[n]), np.std(v_perm[n])
                        d_i = ((N+1)/sg**2) * ((lv-Vg)**2 + (ls-sg)**2)
                        dp.append(np.sqrt(d_i))
                    DS_permuted.append(np.sum(dp))

                p_val = np.sum(np.array(DS_permuted) >= DS_real) / n_permutations
                df_sub[cat_col] = pd.cut(df_sub[delta_col], [0,1,2,3,5], labels=['Bajo','Medio','Alto','Muy Alto'])
                df.loc[df[cluster_col] == cluster, delta_col] = df_sub[delta_col]
                df.loc[df[cluster_col] == cluster, cat_col] = df_sub[cat_col]
                df.loc[df[cluster_col] == cluster, pass_col] = int(p_val < alpha)

                if p_val < alpha:
                    passed.append(cluster)

            return df, passed

        # ================================
        # 📌 Panel PCA+t-SNE + Boxplots
        # ================================

        def plot_tsne_and_boxplots(df, parent_col, selected_cols, level):
            tsne1 = f'TSNE1_{level}'
            tsne2 = f'TSNE2_{level}'

            # Si faltan, genera PCA + TSNE a nivel actua    l        
            if tsne1 not in df.columns or tsne2 not in df.columns:
                parents = sorted(df[parent_col].dropna().unique())
                scaler = StandardScaler()
                tsne1_vals = []
                tsne2_vals = []

                for parent in parents:
                    d = df[df[parent_col] == parent].copy()
                    data = d[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
                    if data.shape[0] < 2:
                        tsne1_vals.extend([np.nan] * len(d))
                        tsne2_vals.extend([np.nan] * len(d))
                        continue

                    scaled = scaler.fit_transform(data)
                    pca = PCA(n_components=min(20, scaled.shape[1])).fit_transform(scaled)
                    #perplexity = min(30, max(5, scaled.shape[0] - 1))
                    # Perplexity dinámico
                    n_points = pca.shape[0]
                    perplexity = max(5, min(30, n_points - 1))

                    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                    result = tsne.fit_transform(pca)

                    # Asigna resultados respetando los índices
                    df.loc[data.index, tsne1] = result[:, 0]
                    df.loc[data.index, tsne2] = result[:, 1]

            # Ahora sí: genera plotly        
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            unique_clusters = sorted(df[parent_col].dropna().unique())
            colors = px.colors.qualitative.Set2

            fig = go.Figure()
            for i, c in enumerate(unique_clusters):
                d = df[df[parent_col] == c]
                fig.add_trace(go.Scatter(
                    x=d[tsne1], y=d[tsne2],
                    mode='markers',
                    name=f'{parent_col} {c}',
                    marker=dict(size=8, color=colors[i % len(colors)]),
                    hovertext=d.apply(lambda row: "<br>".join([f"{col}: {row[col]}" for col in df.columns]), axis=1),
                    hoverinfo='text'
                ))

            fig.update_layout(
                title=f"PCA + t-SNE nivel {level}",
                xaxis_title=tsne1, yaxis_title=tsne2,
                template='plotly_white',
                height=600        
            )        
            st.plotly_chart(fig)

            # Boxplots para cada variable
            for var in selected_cols:
                fig_box = px.box(
                    df[df[parent_col].notna()],
                    x=parent_col, y=var, color=parent_col,
                    points='all', notched=True,
                    color_discrete_sequence=colors,
                    title=f"{var} por {parent_col}"
                )
                st.plotly_chart(fig_box)


        # ================================
        # 📌 Mapa validado final
        # ================================
 #       def plot_validated_map(df, level):
 #           cluster_col = f'Subcluster_{level}'
 #           pass_col = f'DS_Pass_{level}'
 #           df_pass = df[df[pass_col] == 1].copy()

 #           fig = go.Figure()
 #           fig.add_trace(go.Scatter(
 #               x=df['RA'], y=df['Dec'],
 #               mode='markers',
 #               marker=dict(size=4, color='lightgrey', opacity=0.3),
 #               name="Fondo"
 #           ))

 #           colors = px.colors.qualitative.Set2
 #           for i, cluster in enumerate(df_pass[cluster_col].unique()):
 #               d = df_pass[df_pass[cluster_col] == cluster]
 #               hover = d.apply(lambda row: "<br>".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
 #               fig.add_trace(go.Scatter(
 #                   x=d['RA'], y=d['Dec'],
 #                   mode='markers', name=f'{cluster_col} {cluster}',
 #                   marker=dict(size=8, color=colors[i % len(colors)], line=dict(width=0.5)),
 #                   text=hover, hoverinfo='text'
 #               ))
 #               fig.add_trace(go.Histogram2dContour(
 #                   x=d['RA'], y=d['Dec'],
 #                   colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[i % len(colors)]]],
 #                   showscale=False, opacity=0.3, ncontours=10
 #               ))

 #           fig.update_layout(title=f"Mapa nivel {level}: estructuras validadas DS",
 #                             template='plotly_white', xaxis=dict(autorange='reversed'))
 #           st.plotly_chart(fig, use_container_width=True)

 #       def plot_validated_map(df, current_level):
        import plotly.graph_objects as go
        import plotly.express as px

        def plot_validated_map(df, level):
            cluster_col = f'Subcluster_{level}'
            pass_col = f'DS_Pass_{level}'

            if cluster_col not in df.columns or pass_col not in df.columns:
                st.warning(f"❌ Faltan columnas '{cluster_col}' o '{pass_col}'. Ejecuta la prueba DS antes.")
                return

            # ✅ Filtra validados
            passed_list = df[df[pass_col] == 1][cluster_col].unique()
            df_pass = df[df[pass_col] == 1].copy()
            df_fail = df[df[pass_col] != 1].copy()

            st.subheader(f"Subclusters validados (nivel {level}): {len(passed_list)}")
            show_all = st.checkbox("Mostrar galaxias que NO pasan (símbolo X)")

            fig = go.Figure()

            # 🔹 Fondo: NO validadas
            if show_all and not df_fail.empty:
                hover_fail = df_fail.apply(
                    lambda row: "<br>".join([f"{col}: {row[col]}" for col in df.columns if col in row]),
                    axis=1
                )
                fig.add_trace(go.Scatter(
                    x=df_fail['RA'],
                    y=df_fail['Dec'],
                    mode='markers',
                    marker=dict(symbol='x', size=6, color='grey', opacity=0.5),
                    name="No validadas",
                    text=hover_fail,
                    hoverinfo='text'
                ))

            # 🔹 Validadas + contornos
            colors = px.colors.qualitative.Set2

            for i, cluster in enumerate(passed_list):
                data_sub = df_pass[df_pass[cluster_col] == cluster].copy()        
                hover_pass = data_sub.apply(
                    lambda row: "<br>".join([f"{col}: {row[col]}" for col in df.columns if col in row]),
                    axis=1
                )        
                fig.add_trace(go.Scatter(
                    x=data_sub['RA'],
                    y=data_sub['Dec'],
                    mode='markers',
                    marker=dict(size=8, color=colors[i % len(colors)],
                                line=dict(width=0.5, color='DarkSlateGrey')),
                    name=f'Cluster {cluster} (nivel {level})',
                    text=hover_pass,
                    hoverinfo='text'
                ))
    
                fig.add_trace(go.Histogram2dContour(
                    x=data_sub['RA'],
                    y=data_sub['Dec'],
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors[i % len(colors)]]],
                    showscale=False,
                    opacity=0.3,
                    ncontours=10,        
                    line=dict(width=1),
                    hoverinfo='skip',
                    name=f'Contorno {cluster}'
                ))

            fig.update_layout(
                title=f"🗺️ Mapa Abell 85 — Subclusters validados (nivel {level})",
                xaxis_title="Ascensión Recta (RA, grados)",
                yaxis_title="Declinación (Dec, grados)",
                xaxis=dict(autorange='reversed'),
                template='plotly_white',
                height=800,
                width=1000,
                legend_title="Estructuras"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ✅ Botón exportar solo validadas
            if not df_pass.empty:
                st.download_button(
                    "💾 Descargar galaxias validadas",
                    df_pass.to_csv(index=False).encode('utf-8'),
                    file_name=f"Galaxias_DS_Validadas_Nivel_{level}.csv",
                    mime="text/csv"
                )


        with st.expander("**Sub-sub-clustering**"):
            parent_col = 'Subcluster'   # Nivel inicial
            current_level = 1

            while True:
                st.subheader(f"🔄 Clustering nivel {current_level}")

                # Número de clusters para este nivel
                num_clusters = st.slider(f"Clusters para nivel {current_level}", 2, 10, 2)

                # Ejecuta clustering sobre parent_col y guarda nuevas columnas niveladas
                df = run_subclustering_iterative(
                    df,
                    parent_col=parent_col,
                    selected_cols=selected_cols,
                    num_clusters=num_clusters,
                    level=current_level
                )

                # Ejecuta DS sobre la columna recién creada
                df, passed = run_ds_iterative(
                    df,
                    cluster_col=f'Subcluster_{current_level}',
                    level=current_level
                )

                plot_tsne_and_boxplots(
                    df,
                    parent_col=f'Subcluster_{current_level}',
                    selected_cols=selected_cols,
                    level=current_level
                )

            
                # Visualiza mapa con galaxias validadas en este nivel
                plot_validated_map(
                    df,
                    current_level
                )

                # ¿Quieres otro nivel?
                if not st.checkbox(f"➡️ Clustering otro nivel basado en nivel {current_level}?", value=False):
                    break

                # Actualiza parent_col para que el próximo nivel divida estos clusters
                parent_col = f'Subcluster_{current_level}'
                current_level += 1

        import plotly.express as px

        def plot_dispersion_by_subcluster(df, level, variables=None, show_only_validated=True):
            """
            📊 Visualiza dispersión de variables numéricas o distribución de categóricas por subcluster validado.
            """
            cluster_col = f'Subcluster_{level}'
            pass_col = f'DS_Pass_{level}'

            if show_only_validated:
                df_filtered = df[df[pass_col] == 1].copy()
            else:
                df_filtered = df.copy()

            if df_filtered.empty:
                st.info("⚠️ No hay datos para mostrar dispersión.")
                return

            if variables is None:
                variables = df_filtered.columns.tolist()

            unique_clusters = sorted(df_filtered[cluster_col].dropna().unique())
            st.write(f"📊 Mostrando dispersión para {len(unique_clusters)} subclusters validados (Nivel {level})")

            for var in variables:
                st.subheader(f"🔍 {var} por {cluster_col}")

                if pd.api.types.is_numeric_dtype(df_filtered[var]):
                    # Numérica: histograma + boxplot
                    fig_hist = px.histogram(
                        df_filtered,
                        x=var,
                        color=cluster_col,
                        nbins=30,
                        barmode='overlay',
                        title=f"Histograma de {var}",
                        labels={var: var, cluster_col: "Subcluster"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    fig_box = px.box(        
                        df_filtered,
                        x=cluster_col,
                        y=var,
                        color=cluster_col,
                        points='all',
                        notched=True,
                        title=f"Boxplot de {var}",
                        labels={var: var, cluster_col: "Subcluster"}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    # Categórica: barras de frecuencia
                    fig_bar = px.histogram(
                        df_filtered,
                        x=var,
                        color=cluster_col,
                        barmode='group',        
                        title=f"Frecuencias de {var}",
                        labels={var: var, cluster_col: "Subcluster"}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        def plot_global_dispersion(df, variables):
            st.subheader("📊 Dispersión Global de Variables")
            for var in variables:
                if pd.api.types.is_numeric_dtype(df[var]):
                    fig = px.box(df, y=var, points='all', notched=True, title=f'Dispersión de {var}')
                else:
                    fig = px.histogram(df, x=var, color=var, text_auto=True, title=f'Distribución de {var}')
                st.plotly_chart(fig, use_container_width=True)



        def plot_dispersion(df, level, variables, 
                    mode='Global', 
                    show_only_validated=True):
            """
            Muestra la dispersión de variables numéricas o categóricas:
            🔹 Global (todas las galaxias)
            🔹 Por subcluster
            🔹 Por subcluster solo los que pasaron DS
            """
            st.subheader(f"📊 Dispersión de Variables — Modo: {mode}")

            if mode == 'Global':
                d = df.copy()
            elif mode == 'Subclusters':
                subcluster_col = f'Subcluster_{level}'
                d = df[df[subcluster_col].notna()].copy()
            elif mode == 'Validados':
                subcluster_col = f'Subcluster_{level}'
                pass_col = f'DS_Pass_{level}'
                d = df[(df[subcluster_col].notna()) & (df[pass_col] == 1)].copy()
            else:
                st.warning("Modo no válido.")
                return

            for var in variables:
                if var not in d.columns:
                    continue
                #if pd.api.types.is_numeric_dtype(d[var]):
                #    fig = px.box(
                #        d, 
                #        x=subcluster_col if mode != 'Global' else None,
                #        y=var, 
                #        points='all', 
                #        notched=True,
                #        title=f'Dispersión de {var}'
                #    )

                if pd.api.types.is_numeric_dtype(d[var]):
                    #fig = px.box(
                    #    d, 
                    #    x=subcluster_col if mode != 'Global' else None,
                    #    y=var, 
                    #    points='all', 
                    #    notched=True,
                    #    color=subcluster_col if mode != 'Global' else None,
                    #    color_discrete_map=my_color_map if mode != 'Global' else None,
                    #    title=f'Dispersión de {var}'
                    #)
                    fig = px.box(
                        d, 
                        x=subcluster_col if mode != 'Global' else None,
                        y=var, 
                        points='all', 
                        notched=True,
                        color=subcluster_col if mode != 'Global' else None,
                        title=f'Dispersión de {var}'
                    )                
                else:
                    fig = px.histogram(
                        d,
                        x=var,
                        color=subcluster_col if mode != 'Global' else var,
                        text_auto=True,
                        title=f'Distribución de {var}'
                    )
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("**Dispersión de Variables**"):
            mode = st.radio(            
                "¿Qué quieres visualizar?",
                options=['Global', 'Subclusters', 'Validados'],
                index=0
            )

            variables_to_plot = st.multiselect(        
                "Selecciona variables:",
                options=df.select_dtypes(include=['number', 'object', 'category']).columns.tolist(),
                default=['Vel']
            )

            plot_dispersion(
                df,
                level=current_level,   # Nivel actual de clustering
                variables=variables_to_plot,
                mode=mode,
                show_only_validated=(mode == 'Validados')
            )

        
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import streamlit as st

        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go


        def plot_realistic_galaxy_map(
            df,
            ra_col='RA',        
            dec_col='Dec',
            morph_col='M(ave)',
            mag_col='M(IPn)',
            color_index_col='(g-r)',
            subcluster_col='Subcluster',
            vel_col='Vel',
            add_kde=True
        ):
            st.subheader("Mapa con las galaxias separadas de acuerdo a Morfología")
            st.markdown("Aquí puede ver las galaxias separas por morfología (**Nota: de momento solo se divide entre galaxias elípticas y espirales**)")
            fig = go.Figure()

            # 🔹 Estrellas de fondo
            np.random.seed(42)
            fig.add_trace(go.Scatter(
                x=np.random.uniform(df[ra_col].min(), df[ra_col].max(), 500),
                y=np.random.uniform(df[dec_col].min(), df[dec_col].max(), 500),
                mode='markers',
                marker=dict(size=1, color='white', opacity=0.02),
                hoverinfo='skip',
                showlegend=False
            ))

            # 🔹 Símbolos y colores refinados por morfología
            morph_map = {
                'E': {'symbol': 'circle'},
                'S': {'symbol': 'star'},
                'I': {'symbol': 'diamond'},
                'UNK': {'symbol': 'x'}
            }

            # Agrupa por letra principal
            df['Morph_Group'] = df[morph_col].str[0].fillna('UNK')

            # 🔹 Rango de color (g-r) continuo
            color_range = df[color_index_col].quantile([0.05, 0.95]).values.tolist()

            for morph_type, style in morph_map.items():
                df_m = df[df['Morph_Group'] == morph_type]
                if df_m.empty:
                    continue

                sizes = 12 - df_m[mag_col].fillna(df_m[mag_col].max())
                sizes = sizes.clip(lower=2, upper=20)

                fig.add_trace(go.Scatter(
                    x=df_m[ra_col],
                    y=df_m[dec_col],
                    mode='markers',
                    name=f"{morph_type} ({len(df_m)})",
                    marker=dict(
                        size=sizes,
                        symbol=style['symbol'],
                        color=df_m[color_index_col],
                        colorscale='RdBu_r',
                        cmin=color_range[0],
                        cmax=color_range[1],
                        colorbar=dict(title='(g-r)', len=0.3),
                        opacity=0.85,
                        line=dict(width=0.3, color='white')
                    ),
                    text=df_m.apply(
                        lambda r: f"ID: {r['ID']}<br>"
                                  f"RA: {r[ra_col]:.3f}°<br>"
                                  f"Dec: {r[dec_col]:.3f}°<br>"
                                  f"Vel: {r[vel_col]:.1f} km/s<br>"
                                  f"Morph: {r[morph_col]}<br>"
                                  f"Mag: {r[mag_col]:.2f}<br>"
                                  f"(g-r): {r[color_index_col]:.2f}<br>"
                                  f"Subcluster: {r[subcluster_col]}",
                        axis=1
                    ),
                    hoverinfo='text'
                ))

            # 🔹 KDE opcional por subcluster
            if add_kde and subcluster_col in df.columns:
                for subc in sorted(df[subcluster_col].dropna().unique()):
                    subc_data = df[df[subcluster_col] == subc]
                    fig.add_trace(go.Histogram2dContour(
                        x=subc_data[ra_col],
                        y=subc_data[dec_col],
#                        colorscale='Greys',
                        colorscale = [[0, 'white'], [1, 'blue']],
                        showscale=False,
                        opacity=0.15,
                        name=f'KDE Subcluster {subc}',
                        hoverinfo='skip',
                        contours=dict(coloring='lines')  # 👉 Modo solo líneas

                    ))

            fig.update_layout(
                paper_bgcolor='black',
                plot_bgcolor='black',
                xaxis=dict(
                    title='Ascensión Recta (RA, grados)',
                    autorange='reversed',
                    showgrid=False,
                    color='white'
                ),
                yaxis=dict(
                    title='Declinación (Dec, grados)',
                    showgrid=False,
                    color='white'
                ),
                legend=dict(
                    font=dict(color='white')
                ),
                font=dict(color='white'),
                title=dict(text="Mapa Realista: Campo Profundo + (g-r) + Morfología", font=dict(color='white'))
            )

            fig.update_layout(
                width=800,      # Aumenta ancho
                height=1000    # Más alto = aspecto más vert
            )

            
            st.plotly_chart(fig, use_container_width=True)



#        with st.expander("🌌 Mapa Realista Campo Profundo"):
#            plot_realistic_map_streamlit(df)

        with st.expander("**Mapa de acuerdo a Morfología**"):
            plot_realistic_galaxy_map(
                df,
                add_kde=True  # True o False
            )


  #      import plotly.graph_objects as go
  #      import numpy as np
  #      import streamlit as st

  #      def plot_realistic_plotly(df, ra_col='RA', dec_col='Dec',
  #                                morph_col='M(C)', mag_col='M(IPn)', color_index_col='(g-r)',
  #                                subcluster_col='Subcluster'):
  #          st.subheader("🌌 Mapa Abell 85 — Estilo Plotly")

  #          fig = go.Figure()

   #         # 🔹 Fondo de estrellas
   #         np.random.seed(42)
   #         fig.add_trace(go.Scatter(
   #             x=np.random.uniform(df[ra_col].min(), df[ra_col].max(), 500),
   #             y=np.random.uniform(df[dec_col].min(), df[dec_col].max(), 500),
   #             mode='markers',
   #             marker=dict(size=1, color='white', opacity=0.02),
   #             hoverinfo='skip',
   #             showlegend=False
   #         ))

   #         morph_map = {
   #             'E': {'color': 'gold', 'symbol': 'circle'},
   #             'S': {'color': 'deepskyblue', 'symbol': 'star'},
#            'I': {'color': 'red', 'symbol': 'diamond'},
#                'UNK': {'color': 'grey', 'symbol': 'x'}
#            }

#            df['Morph_Group'] = df[morph_col].str[0].fillna('UNK')

#            for morph_type, style in morph_map.items():
#                df_m = df[df['Morph_Group'] == morph_type]
#                if df_m.empty:
#                    continue

#                sizes = 12 - df_m[mag_col].fillna(df_m[mag_col].max())
#                sizes = sizes.clip(lower=2, upper=18)

#                fig.add_trace(go.Scatter(
#                    x=df_m[ra_col],
 #                   y=df_m[dec_col],
 #                   mode='markers',
 #                   name=f"{morph_type} ({len(df_m)})",
 #                   marker=dict(
 #                       size=sizes,
 #                       color=style['color'],
 #                       symbol=style['symbol'],
 #                       opacity=0.85,
 #                       line=dict(width=0.5, color='white')
 #                   ),
 #                   text=df_m.apply(
 #                       lambda r: f"ID: {r['ID']}<br>"
 #                                 f"RA: {r[ra_col]:.3f}°<br>"
 #                                 f"Dec: {r[dec_col]:.3f}°<br>"
 #                                 f"Vel: {r['Vel']:.1f} km/s<br>"
 #                                 f"Morph: {r[morph_col]}<br>"
 #                                 f"Mag: {r[mag_col]:.2f}<br>"
 #                                 f"(g-r): {r[color_index_col]:.2f}<br>"
 #                                 f"Subcluster: {r[subcluster_col]}",
 #                       axis=1
 #                   ),
 #                   hoverinfo='text'
 #               ))

 #           fig.update_layout(
 #               paper_bgcolor='black',
 #               plot_bgcolor='black',
 #               xaxis=dict(title='RA', autorange='reversed', showgrid=False, color='white'),
 #               yaxis=dict(title='Dec', showgrid=False, color='white'),
 #               legend=dict(font=dict(color='white')),
 #               font=dict(color='white'),
 #               height=800, width=800,
 #               title=dict(text="Mapa Realista: Plotly", font=dict(color='white'))
 #           )
#
#            st.plotly_chart(fig, use_container_width=True)

#        with st.expander("🌌 Mapa Plotly Realista"):
#            plot_realistic_plotly(df)


#        import streamlit as st
#    
#        import streamlit as st

#        def generate_simple_svg():
#            svg_code = """
#            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" style="background:black;">
#              <defs>
#                <radialGradient id="grad1" cx="50%" cy="50%" r="50%">
#                  <stop offset="0%" style="stop-color:rgba(0,255,200,0.5);" />
#                  <stop offset="100%" style="stop-color:rgba(0,0,0,0);" />
#                </radialGradient>
#                <filter id="blur">
#                  <feGaussianBlur stdDeviation="2"/>
#                </filter>
#              </defs>

#              <circle cx="100" cy="100" r="30" fill="url(#grad1)" filter="url(#blur)" opacity="0.8" />
      
#              <g>
#                <ellipse cx="150" cy="50" rx="20" ry="8" fill="url(#grad1)" filter="url(#blur)" opacity="0.8"/>
#                <animateTransform attributeName="transform" attributeType="XML"
#                  type="rotate" from="0 150 50" to="360 150 50" dur="20s" repeatCount="indefinite"/>
#              </g>
#            </svg>
#            """
#            st.markdown(svg_code, unsafe_allow_html=True)

#        with st.expander("SVG Test"):
#            generate_simple_svg()



#        import streamlit as st
#        import numpy as np

#        import streamlit as st
#
#        def generate_realistic_svg(df,
#                                   ra_col='RA', dec_col='Dec',
#                                   morph_col='M(ave)', mag_col='M(IPn)',
#                                   vel_col='Vel'):
#            """
#            🌌 Mapa Deep Field SVG realista adaptado.
#            Incluye:
#            - Gradientes difusos        
#            - Filtros blur
#            - Formas realistas por morfología
#            """

#            svg_parts = [
#                '<svg viewBox="0 0 1000 1000" style="background:black;" xmlns="http://www.w3.org/2000/svg">',
#                '<defs>',
#                '<filter id="blur"><feGaussianBlur stdDeviation="4"/></filter>',
#            ]

#            # === Gradientes ===
#            gradients = [
#                '<radialGradient id="grad0" cx="50%" cy="50%" r="50%"><stop offset="0%" style="stop-color:rgba(200,255,255,0.2);"/><stop offset="100%" style="stop-color:rgba(0,0,0,0);"/></radialGradient>',
#                '<radialGradient id="grad1" cx="50%" cy="50%" r="50%"><stop offset="0%" style="stop-color:rgba(0,150,255,0.2);"/><stop offset="100%" style="stop-color:rgba(0,0,0,0);"/></radialGradient>',
#                '<radialGradient id="grad2" cx="50%" cy="50%" r="50%"><stop offset="0%" style="stop-color:rgba(255,255,200,0.15);"/><stop offset="100%" style="stop-color:rgba(0,0,0,0);"/></radialGradient>',
#                '<radialGradient id="grad3" cx="50%" cy="50%" r="50%"><stop offset="0%" style="stop-color:rgba(255,100,200,0.15);"/><stop offset="100%" style="stop-color:rgba(0,0,0,0);"/></radialGradient>',
#            ]
#            svg_parts.extend(gradients)
#            svg_parts.append('</defs>')

#            # === Escala RA/Dec ===
#            ra_min, ra_max = df[ra_col].min(), df[ra_col].max()
#            dec_min, dec_max = df[dec_col].min(), df[dec_col].max()

#            def scale_ra(ra):
#                return int(1000 * (ra - ra_min) / (ra_max - ra_min))

#            def scale_dec(dec):
#                return int(1000 * (1 - (dec - dec_min) / (dec_max - dec_min)))

#            # === Galaxias ===
#            for idx, row in df.iterrows():
#                x = scale_ra(row[ra_col])
#                y = scale_dec(row[dec_col])
#                morph = str(row[morph_col]) if pd.notna(row[morph_col]) else 'UNK'

#                # Tamaño aproximado inverso a magnitud
#                size = max(4, 12 - float(row[mag_col])) if mag_col in row else 8

#                # Asigna gradiente por tipo
#                if "E" in morph:
#                    grad_id = "grad0"
#                    shape = f'<circle cx="{x}" cy="{y}" r="{size}" fill="url(#{grad_id})" filter="url(#blur)" opacity="0.9"><title>ID: {row.get("ID","")} | RA: {row[ra_col]:.3f} | Dec: {row[dec_col]:.3f} | Vel: {row.get(vel_col,"")} | Morph: {morph}</title></circle>'
#                elif "Sb" in morph or "Sc" in morph:
#                    grad_id = "grad1"
#                    shape = (
#                        f'<g>'
#                        f'<ellipse cx="{x}" cy="{y}" rx="{size}" ry="{size//2}" '
#                        f'fill="url(#{grad_id})" filter="url(#blur)" opacity="0.9">'
#                        f'<title>ID: {row.get("ID","")} | RA: {row[ra_col]:.3f} | Dec: {row[dec_col]:.3f} | Vel: {row.get(vel_col,"")} | Morph: {morph}</title>'
#                        f'</ellipse>'
#                        f'<animateTransform attributeName="transform" attributeType="XML" '
#                        f'type="rotate" from="0 {x} {y}" to="360 {x} {y}" dur="40s" repeatCount="indefinite"/>'
#                        f'</g>'
#                   )
#                elif "S0" in morph:
#                    grad_id = "grad2"
#                    shape = f'<ellipse cx="{x}" cy="{y}" rx="{size}" ry="{size//3}" fill="url(#{grad_id})" filter="url(#blur)" opacity="0.7"><title>ID: {row.get("ID","")} | RA: {row[ra_col]:.3f} | Dec: {row[dec_col]:.3f} | Vel: {row.get(vel_col,"")} | Morph: {morph}</title></ellipse>'
#                else:
#                    grad_id = "grad3"
#                    shape = f'<circle cx="{x}" cy="{y}" r="{size//2}" fill="url(#{grad_id})" filter="url(#blur)" opacity="0.5"><title>ID: {row.get("ID","")} | RA: {row[ra_col]:.3f} | Dec: {row[dec_col]:.3f} | Vel: {row.get(vel_col,"")} | Morph: {morph}</title></circle>'

#                svg_parts.append(shape)

#            svg_parts.append('</svg>')

#            svg_code = "\n".join(svg_parts)
#            st.subheader("🌌 Vista Realista Deep Field SVG")
#            st.markdown(f'<div style="background:black;">{svg_code}</div>', unsafe_allow_html=True)
#        with st.expander("🌌 Mapa Realista Deep Field SVG"):
#            generate_realistic_svg(df)



 #       import streamlit as st
 #       import numpy as np
 #       import pandas as pd

#        def generate_realistic_kde_svg(
#            df,
#            ra_col='RA',
#            dec_col='Dec',
#            morph_col='M(ave)',
#            subcluster_col='Subcluster',
#            rf_col='Rf',
#            vel_col='Vel'
#        ):
#            st.subheader("🌌 Campo Profundo Abell 85 — Mapa Realista SVG (Gradientes + KDE)")

#            blur_strength = st.slider("🔆 Blur KDE", 1, 20, 4)
#            brightness_factor = st.slider("💡 Brillo máximo KDE", 0.05, 1.0, 0.2)

#            active_morphs = st.multiselect(
#                "Morfologías visibles",
#                options=sorted(df[morph_col].dropna().unique()),
#                default=sorted(df[morph_col].dropna().unique())
#            )

#            active_subs = st.multiselect(
#                "Subclusters visibles",
#                options=sorted(df[subcluster_col].dropna().unique()),
#                default=sorted(df[subcluster_col].dropna().unique())
#            )
            
#            # --- Escalado de RA invertido ---
#            ra_min, ra_max = df[ra_col].min(), df[ra_col].max()
#            dec_min, dec_max = df[dec_col].min(), df[dec_col].max()


#            svg_parts = [
#                '<svg viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg" style="background:black;">',
#                '<defs>',
#                f'<filter id="blur"><feGaussianBlur stdDeviation="{blur_strength}"/></filter>',
#                '<radialGradient id="gradHalo" cx="50%" cy="50%" r="50%">'
#                '<stop offset="0%" style="stop-color:rgba(0,255,200,0.5);" />'
#                '<stop offset="100%" style="stop-color:rgba(0,0,0,0);" />'
#                '</radialGradient>',
#                '</defs>'
#            ]
            
#            x_min, x_max = df[ra_col].min(), df[ra_col].max()
#            y_min, y_max = df[dec_col].min(), df[dec_col].max()

#            def scale_ra(ra):
#                return int(1000 * (ra - x_min) / (x_max - x_min))

#            def scale_dec(dec):
#                return int(1000 * (1 - (dec - y_min) / (y_max - y_min)))

#            # === KDE Halos ===
#            for _, row in df.iterrows():
#                if row[subcluster_col] not in active_subs:
#                    continue
#                x = scale_ra(row[ra_col])
#                y = scale_dec(row[dec_col])

#                rf_value = row.get(rf_col, -2)
#                if pd.isna(rf_value):
#                    rf_value = -2
#                radius = int(10 + np.clip(abs(rf_value) * 40, 10, 150))
#                opacity = np.clip(abs(rf_value) * brightness_factor, 0.05, 0.3)

#                svg_parts.append(
#                    f'<circle cx="{x}" cy="{y}" r="{radius}" fill="url(#gradHalo)" filter="url(#blur)" opacity="{opacity:.2f}"/>'
#                )

#            morph_colors = {
#                'E': '#FFDAB9',
#                'S': '#00FFFF',
#                'I': '#FF69B4',
#                'UNK': '#CCCCCC'
#            }

#            for _, row in df.iterrows():
#                morph = str(row.get(morph_col, 'UNK'))
#                sub = row.get(subcluster_col, None)
#                if morph not in active_morphs or sub not in active_subs:
#                    continue

#                x = scale_ra(row[ra_col])
#                y = scale_dec(row[dec_col])
#                base_color = morph_colors.get(morph[0], '#FFFFFF')
#                size = 6

#                title = (
#                    f"ID: {row.get('ID','')} | RA: {row[ra_col]:.3f} | Dec: {row[dec_col]:.3f} | "
#                    f"Vel: {row.get(vel_col,'')} | Rf: {row.get(rf_col,''):.2f} | "
#                    f"Morph: {morph} | Sub: {sub}"
#                )

#                if "E" in morph:
#                    shape = f'<circle cx="{x}" cy="{y}" r="{size}" fill="{base_color}" opacity="0.9"><title>{title}</title></circle>'
#                elif "Sa" in morph or "Sb" in morph or "Sc" in morph:
#                    shape = f'<ellipse cx="{x}" cy="{y}" rx="{size*1.5}" ry="{size}" fill="{base_color}" opacity="0.9"><title>{title}</title></ellipse>'
#                else:
#                    shape = f'<circle cx="{x}" cy="{y}" r="{size//2}" fill="{base_color}" opacity="0.5"><title>{title}</title></circle>'

#                svg_parts.append(shape)

#            np.random.seed(42)
#            for _ in range(400):
#                x = np.random.randint(0, 1000)
#                y = np.random.randint(0, 1000)
#                star_size = np.random.uniform(0.2, 1.0)
#                opacity = np.random.uniform(0.05, 0.2)
#                svg_parts.append(
#                    f'<circle cx="{x}" cy="{y}" r="{star_size:.2f}" fill="white" opacity="{opacity:.2f}"/>'
#                )

#            svg_parts.append('</svg>')

#            svg_code = "\n".join(svg_parts)
#            st.markdown(f'<div style="background:black;">{svg_code}</div>', unsafe_allow_html=True)


#        with st.expander("🌌 Mapa Realista Deep Field SVG"):
#            generate_realistic_kde_svg(df)


        import streamlit as st
        import numpy as np
        from PIL import Image, ImageDraw
        import math
        import random




        import streamlit as st
        from plot_galaxy_map import plot_galaxy_map
        #show_stars = st.sidebar.checkbox("Mostrar estrellas de campo", value=True)

        # Suponiendo que ya cargaste df
        img=plot_galaxy_map(df)
         

        from PIL import ImageDraw
        import streamlit as st

        from PIL import ImageDraw, ImageFont
        import streamlit as st
        import numpy as np
        
        import streamlit as st
        from PIL import ImageDraw, ImageFont
        import numpy as np


        import pandas as pd
        import streamlit as st
        from PIL import ImageDraw, ImageFont

        import pandas as pd
        import streamlit as st
        from PIL import ImageDraw, ImageFont

        def highlight_ranked_galaxies_with_selector(img, df_filtered,
                                                     ra_col='RA', dec_col='Dec', id_col='ID',
                                                     rf_col='Rf', delta_col='Delta', vel_col='Vel', cld_col='Cl_d',
                                                     width=1024, height=1024):
            RA_min, RA_max = df_filtered[ra_col].min(), df_filtered[ra_col].max()
            Dec_min, Dec_max = df_filtered[dec_col].min(), df_filtered[dec_col].max()
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            variables = {
                rf_col: "Más Brillantes",
                delta_col: "Delta",
                vel_col: "Vel",
                cld_col: "Cl_d"
            }

            quartile_colors = {        
                4: "deepskyblue",
                3: "#cd7f32",
                2: "silver",  # bronce
                1: "gold"
            }

            var_options = list(variables.keys())
            selected_var = st.sidebar.selectbox("Variable para ranking:", var_options, format_func=lambda x: variables[x])
            top_n = st.sidebar.slider("Número de galaxias a resaltar:", min_value=1, max_value=100, value=5)

            sorted_df = df_filtered.copy()
            if selected_var == rf_col:
                sorted_df = sorted_df.sort_values(selected_var)
            else:
                sorted_df = sorted_df.sort_values(selected_var, ascending=False)

            df_extreme = sorted_df.head(top_n).copy()

            try:
                #df_extreme['cuartil'] = pd.qcut(
                #    df_extreme[selected_var],
                #    q=4,
                #    labels=[4, 3, 2, 1]
                #).astype(int)

                # 1. Calcula el rango global
                min_val = df_extreme[selected_var].min()
                max_val = df_extreme[selected_var].max()

                # 2. Crea cortes de rango igual
                df_extreme['cuartil'] = pd.cut(
                    df_extreme[selected_var],
                    bins=4,
                    labels=[4, 3, 2, 1],
                    include_lowest=True
                ).astype(int)
                                                         
            except ValueError:
                df_extreme['cuartil'] = 4  # Fallback: todos igual

            for rank, (_, row) in enumerate(df_extreme.iterrows(), 1):
                color = quartile_colors.get(row['cuartil'], 'white')
                RA, Dec = row[ra_col], row[dec_col]
                x = int((RA - RA_min) / (RA_max - RA_min) * width)
                y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height)
                    # 🔄 Ajuste para ROTATE_180: invierte ambos ejes
                x = width - x
                y = height - y

                r = 8
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
                draw.text((x + r + 2, y), f"{rank}", fill=color, font=font)

            return img

        # Asegúrate de pasar el mismo DataFrame filtrado o usa df.copy() si es todo
        img = highlight_ranked_galaxies_with_selector(
            img,
            df,
            ra_col='RA',
            dec_col='Dec',
            id_col='ID',
            rf_col='Rf',
            delta_col='Delta',
            vel_col='Vel',
            cld_col='Cl_d',
            width=1024,
            height=1024
        )

        with st.expander("**Mapa Final con Resaltado**"):
            st.markdown("Seleccione la variable a mostrar y el número de galaxias a desplegarse (puede seleccionar hasta 100 con los valores mas altos en la variable).")
            st.image(img)  # ✅ Esto mostrará el resultado actualizado

        with st.expander("**Mapa con tamaños aparentes**"):
            # 2. Calcular distancia comóvil


            from astropy.cosmology import FlatLambdaCDM

            def calculate_comoving_distance(df, H0=70.0, Om0=0.3, z_cluster=0.0555):
                """
                Calcula la distancia comóvil a partir del redshift estimado con velocidad.
                Agrega columnas z_gal y Dist al DataFrame.
                """
                import numpy as np
                df = df.copy()
                c = 3e5  # km/s
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                df['z_gal'] = z_cluster + (df['Vel'] / c) * (1 + z_cluster)
                df['Dist'] = cosmo.comoving_distance(df['z_gal']).value  # en Mpc
                return df

            #df_with_dist = calculate_comoving_distance(df)

            from plot_galaxy_map_c import plot_galaxy_map_with_distances
            img = plot_galaxy_map_with_distances(df)
            img.save("mapa_galaxias_corregido.png")
            img.show()

        

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


    
        with st.expander("**Panel condicional Delta × Vel con KDE**"):
            plot_conditional_panel(df)

    
        #else:
        #    st.warning("Selecciona al menos una columna numérica.")
    else:
        st.info("Por favor, sube un archivo CSV.")

elif opcion == "Equipo de trabajo":
    st.subheader("Equipo de Trabajo")

       # Información del equipo
    equipo = [{
               "nombre": "D. C. (Astrofísica). Santiago Arceo Díaz",
               "foto": "ArceoS.jpg",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima. Miembro del Sistema Nacional de Investigadoras e Investigadores (Nivel 1).",
               "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "santiagoarceodiaz@gmail.com"},
              {
               "nombre": "M. C. (Astrofísica) Carlos Arturo Flores Hernández",
               "foto": "FloresC.jpeg",
               "reseña": "Licenciado en Física, Maestro en Astrofísica.",
               "CV": "https://www.linkedin.com/in/carlos-arturo-flores-hernandez-824019117/?originalSubdomain=mx", "contacto": " "},
              {
               "nombre": "D. C. (Astrofísica) Juan Manuel Islas Islas",
               "foto": "IslasJ.jpeg",
               "reseña": "Licenciado en Física, Maestro en Astrofísica y Doctor en Ciencias (Astrofísica).",
               "CV": "https://scholar.google.com/citations?hl=es&user=dvyLfnUAAAAJ", "contacto": " "},
              {
               "nombre": "D. C. (Astrofísica) René Alberto Ortega Minakata",
               "foto": "MinakataR.jpg",
               "reseña": "Licenciado en Física, Maestro en Astrofísica y Doctor en Ciencias (Astrofísica).",
               "CV": "https://www.irya.unam.mx/web/es/gente/tecnicos-academicos/r-ortega", "contacto": "r.ortega@irya.unam.mx"},
              {
               "nombre": "D. C. (Astrofísica) Josué de Jesús Trejo Alonso",
               "foto": "TrejoJ.png",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica).",
               "CV": "https://www.researchgate.net/profile/Josue-Trejo-Alonso", "contacto": " "}
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
