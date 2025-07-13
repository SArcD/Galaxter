import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random

def plot_galaxy_map(
    df,
    ra_col='RA',
    dec_col='Dec',
    morph_col='M(ave)',
    subcluster_col='Subcluster',
    rf_col='Rf',
    width=1024,
    height=1024
):
    st.header("Mapa de Cúmulo con RA/Dec desde DataFrame")

    all_morphs = sorted(df[morph_col].dropna().unique())
    all_subclusters = sorted(df[subcluster_col].dropna().unique())

    morph_filter = st.sidebar.multiselect("Filtrar por morfología", all_morphs, default=all_morphs)
    subcluster_filter = st.sidebar.multiselect("Filtrar por Subcluster", all_subclusters, default=all_subclusters)

    df_filtered = df[
        df[morph_col].isin(morph_filter) &
        df[subcluster_col].isin(subcluster_filter)
    ].dropna(subset=[ra_col, dec_col, rf_col])

    if df_filtered.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return

    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()

    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # Agregar halo central
    halo_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_halo = ImageDraw.Draw(halo_layer)
    draw_halo.ellipse(
        [width // 2 - 400, height // 2 - 400, width // 2 + 400, height // 2 + 400],
        fill=(0, 180, 150, 40)
    )
    halo_blurred = halo_layer.filter(ImageFilter.GaussianBlur(100))
    img.alpha_composite(halo_blurred)

    # Agregar halos locales por subcluster
    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()
    for _, row in subcluster_positions.iterrows():
        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)
        local_halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_local = ImageDraw.Draw(local_halo)
        draw_local.ellipse(
            [cx - 200, cy - 200, cx + 200, cy + 200],
            fill=(0, 200, 180, 25)
        )
        local_blurred = local_halo.filter(ImageFilter.GaussianBlur(50))
        img.alpha_composite(local_blurred)

    for _, row in df_filtered.iterrows():
        RA = row[ra_col]
        Dec = row[dec_col]
        morph = row[morph_col]
        try:
            mag_rf = float(row[rf_col])
        except:
            mag_rf = -15.0

        size = max(8, int(30 - abs(mag_rf)))
        brightness = 200

        if morph.lower() == 'spiral':
            galaxy = draw_spiral(size, brightness)
        elif morph.lower() == 'elliptical':
            galaxy = draw_elliptical(size, brightness)
        else:
            galaxy = draw_irregular(size, brightness)

        angle = random.randint(0, 360)
        galaxy = galaxy.rotate(angle, expand=True)

        x = int((RA - RA_min) / (RA_max - RA_min) * width) - galaxy.width // 2
        y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height) - galaxy.height // 2

        img.alpha_composite(galaxy, (x, y))

    for _ in range(2000):
        sx, sy = random.randint(0, width - 1), random.randint(0, height - 1)
        b = random.randint(120, 220)
        draw.point((sx, sy), fill=(b, b, b, 255))

    with st.expander("Mapa generado"):
        st.image(img)
        st.dataframe(df_filtered)

def draw_spiral(size, brightness):
    g = Image.new('RGBA', (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(100):
        theta = i * 0.15
        radius = size * (i / 100)
        noise = np.random.normal(0, 0.5)
        x = int(cx + (radius + noise) * math.cos(theta))
        y = int(cy + (radius + noise) * math.sin(theta))
        if 0 <= x < g.width and 0 <= y < g.height:
            draw.point((x, y), fill=(100, 200, 255, brightness))
    draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=(220, 220, 255, 255))
    return g

def draw_elliptical(size, brightness):
    g = Image.new('RGBA', (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(3, 0, -1):
        rx = int(size * (i / 3))
        ry = int(size * (i / 3) * 0.6)
        alpha = int(brightness * (i / 3) * 0.5)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=(150, 220, 255, alpha))
    return g

def draw_irregular(size, brightness):
    g = Image.new('RGBA', (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for _ in range(size * 4):
        dx = random.randint(-size // 2, size // 2)
        dy = random.randint(-size // 2, size // 2)
        draw.point((cx + dx, cy + dy), fill=(120, 200, 255, brightness))
    return g
