import streamlit as st
from PIL import ImageDraw, ImageFont
import numpy as np

def plot_galaxy_map(img, df_filtered, ra_col='RA', dec_col='Dec', id_col='ID', rf_col='Rf', delta_col='Delta', vel_col='Vel', cld_col='Cl_d', width=1024, height=1024):
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

    var_options = list(variables.keys())
    selected_var = st.sidebar.selectbox("Variable para ranking:", var_options, format_func=lambda x: variables[x])
    top_n = st.sidebar.slider("Número de galaxias a resaltar:", min_value=1, max_value=20, value=5)

    sorted_df = df_filtered.copy()
    if selected_var == rf_col:
        sorted_df = sorted_df.sort_values(selected_var)
    else:
        sorted_df = sorted_df.sort_values(selected_var, ascending=False)

    values = sorted_df[selected_var].values
    q75, q50, q25 = np.percentile(values, [75, 50, 25])

    for rank, (_, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
        val = row[selected_var]
        if val >= q75:
            color = "gold"
        elif val >= q50:
            color = "orange"
        elif val >= q25:
            color = "deepskyblue"
        else:
            color = "silver"

        RA, Dec = row[ra_col], row[dec_col]
        x = int((RA - RA_min) / (RA_max - RA_min) * width)
        y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height)

        # Dibujar flecha
        offset = 40
        end_x = x + offset
        end_y = y - offset
        draw.line([(end_x, end_y), (x, y)], fill=color, width=3)

        # Poner numerito al inicio de la flecha
        draw.text((end_x + 4, end_y - 4), f"{rank}", fill=color, font=font)

    return img
