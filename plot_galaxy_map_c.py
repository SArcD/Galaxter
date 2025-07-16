from PIL import Image, ImageDraw
import numpy as np
import random
from astropy.cosmology import FlatLambdaCDM
import streamlit as st

def plot_galaxy_map_with_distances(df, H0=70.0, Om0=0.3, z_cluster=0.0555,
                                    ra_col='RA', dec_col='Dec', vel_col='Vel',
                                    morph_col='M(ave)', rf_col='Rf', base_size=6):
    """
    Genera una imagen tipo mapa estelar, escalando las galaxias según su distancia comóvil.
    """

    # Función interna para clasificar morfología
    def classify_morphology(morph_str):
        try:
            morph_str = morph_str.lower()
            if 'e' in morph_str:
                return 'eliptica'
            elif 's' in morph_str:
                return 'espiral'
            else:
                return 'irregular'
        except:
            return 'desconocida'

    # Función interna para calcular distancias comóviles
    def calculate_comoving_distance(df_local, H0, Om0, z_cluster):
        df_local = df_local.copy()
        c = 3e5  # km/s
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        df_local['z_gal'] = z_cluster + (df_local[vel_col] / c) * (1 + z_cluster)
        df_local['Dist'] = cosmo.comoving_distance(df_local['z_gal']).value  # en Mpc
        return df_local

    # Parámetros interactivos
    widget_prefix = "galaxy_map"
    show_stars = st.sidebar.checkbox("Mostrar estrellas de campo", value=True, key=f"{widget_prefix}_stars")
    morph_filter = st.sidebar.multiselect("Filtrar por morfología", options=df[morph_col].dropna().unique().tolist(),
                                          default=df[morph_col].dropna().unique().tolist(), key=f"{widget_prefix}_filter")

    df = calculate_comoving_distance(df, H0, Om0, z_cluster)
    df_filtered = df[df[morph_col].isin(morph_filter)].dropna(subset=[ra_col, dec_col, morph_col, rf_col])

    # Imagen base
    img_size = (800, 800)
    img = Image.new("RGB", img_size, "black")
    draw = ImageDraw.Draw(img)

    ra = df_filtered[ra_col]
    dec = df_filtered[dec_col]
    rf = df_filtered[rf_col]
    morph = df_filtered[morph_col].apply(classify_morphology)

    # Normalización
    ra_scaled = (ra - ra.min()) / (ra.max() - ra.min()) * img_size[0]
    dec_scaled = img_size[1] - (dec - dec.min()) / (dec.max() - dec.min()) * img_size[1]

    # Escala por distancia
    distances = df_filtered['Dist'].values
    scale_factor = np.mean(distances) / distances
    scale_factor = np.clip(scale_factor, 0.5, 2.0)

    colors = {
        'eliptica': (255, 128, 0),
        'espiral': (0, 255, 255),
        'irregular': (255, 0, 255),
        'desconocida': (200, 200, 200)
    }

    for (x, y), morph_type, sf in zip(zip(ra_scaled, dec_scaled), morph, scale_factor):
        size = int(base_size * sf * random.uniform(0.9, 1.2))
        bbox = [x - size, y - size, x + size, y + size]
        draw.ellipse(bbox, fill=colors.get(morph_type, (255, 255, 255)))

    if show_stars:
        for _ in range(300):
            x = random.randint(0, img_size[0] - 1)
            y = random.randint(0, img_size[1] - 1)
            img.putpixel((x, y), (255, 255, 255))

    return img
