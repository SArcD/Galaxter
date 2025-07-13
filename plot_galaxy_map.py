import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
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

    # Sidebar: filtros
    all_morphs = sorted(df[morph_col].unique())
    all_subclusters = sorted(df[subcluster_col].unique())

    morph_filter = st.sidebar.multiselect(
        "Filtrar por morfología",
        all_morphs,
        default=all_morphs
    )

    subcluster_filter = st.sidebar.multiselect(
        "Filtrar por Subcluster",
        all_subclusters,
        default=all_subclusters
    )

    df_filtered = df[
        df[morph_col].isin(morph_filter) & 
        df[subcluster_col].isin(subcluster_filter)
    ]

    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()

    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    for _, row in df_filtered.iterrows():
        RA = row[ra_col]
        Dec = row[dec_col]
        morph = row[morph_col]
        mag_rf = row[rf_col]

        size = max(4, int(20 - abs(mag_rf)))
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
        sx, sy = random.randint(0, width-1), random.randint(0, height-1)
        b = random.randint(120, 220)
        draw.point((sx, sy), fill=(b, b, b, 255))

    with st.expander("Mapa generado"):
        st.image(img)
        st.dataframe(df_filtered)

def draw_spiral(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(100):
        theta = i * 0.15
        radius = size * (i / 100)
        noise = np.random.normal(0, 0.5)
        x = int(cx + (radius + noise) * math.cos(theta))
        y = int(cy + (radius + noise) * math.sin(theta))
        if 0 <= x < g.width and 0 <= y < g.height:
            draw.point((x,y), fill=(100,200,255, brightness))
    draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(220,220,255,255))
    return g

def draw_elliptical(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(3,0,-1):
        rx = int(size * (i/3))
        ry = int(size * (i/3) * 0.6)
        alpha = int(brightness * (i/3) * 0.5)
        bbox = [cx-rx, cy-ry, cx+rx, cy+ry]
        draw.ellipse(bbox, fill=(150,220,255,alpha))
    return g

def draw_irregular(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for _ in range(size*4):
        dx = random.randint(-size//2, size//2)
        dy = random.randint(-size//2, size//2)
        draw.point((cx+dx, cy+dy), fill=(120,200,255, brightness))
    return g
