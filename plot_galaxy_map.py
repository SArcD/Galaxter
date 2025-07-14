import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
from noise import pnoise2

# --- Generador de halos con Perlin ---
def generate_perlin_halo(width, height, scale=0.02, octaves=2, alpha=60):
    halo = Image.new('RGBA', (width, height), (0,0,0,0))
    for x in range(width):
        for y in range(height):
            noise_val = pnoise2(x * scale, y * scale, octaves=octaves)
            val = int(100 * (noise_val + 0.5))
            halo.putpixel((x,y), (0, 180, 150, min(max(val,0), alpha)))
    return halo.filter(ImageFilter.GaussianBlur(60))

# --- Función principal ---
def plot_galaxy_map(df, ra_col='RA', dec_col='Dec', morph_col='M(ave)', subcluster_col='Subcluster', rf_col='Rf', width=1024, height=1024):
    st.header("Mapa de Cúmulo con Morfología + Halo Perlin")
    all_morphs = sorted(df[morph_col].dropna().unique())
    all_subclusters = sorted(df[subcluster_col].dropna().unique())
    morph_filter = st.sidebar.multiselect("Filtrar por morfología", all_morphs, default=all_morphs)
    subcluster_filter = st.sidebar.multiselect("Filtrar por Subcluster", all_subclusters, default=all_subclusters)
    df_filtered = df[df[morph_col].isin(morph_filter) & df[subcluster_col].isin(subcluster_filter)].dropna(subset=[ra_col, dec_col, rf_col])
    if df_filtered.empty:
        st.warning("No hay datos.")
        return
    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()
    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    img.alpha_composite(generate_perlin_halo(width, height))

    for _, row in df_filtered.iterrows():
        RA, Dec, morph = row[ra_col], row[dec_col], row[morph_col]
        try: mag_rf = float(row[rf_col])
        except: mag_rf = -15.0
        size = max(30, int(60 - abs(mag_rf)))
        brightness = 255
        if morph.lower() == 'spiral': galaxy = draw_spiral(size, brightness)
        elif morph.lower() == 'elliptical': galaxy = draw_elliptical(size, brightness)
        elif morph.lower() == 'lenticular': galaxy = draw_lenticular(size, brightness)
        else: galaxy = draw_irregular(size, brightness)
        galaxy = galaxy.rotate(random.randint(0, 360), expand=True)
        x = int((RA - RA_min) / (RA_max - RA_min) * width) - galaxy.width // 2
        y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height) - galaxy.height // 2
        img.alpha_composite(galaxy, (x, y))

    for _ in range(2000):
        sx, sy = random.randint(0, width - 1), random.randint(0, height - 1)
        b = random.randint(120, 220)
        img.putpixel((sx, sy), (b, b, b, 255))
    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    with st.expander("Mapa"):
        st.image(img)
        st.dataframe(df_filtered)

# --- Formas mejoradas ---
def draw_spiral(size, brightness):
    g = Image.new('RGBA', (size*3, size*3), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for arm in range(2):
        points = []
        for i in range(150):
            t = i / 150 * (3 * 2 * math.pi)
            r = size * (i/150)
            x = int(cx + r * math.cos(t + arm * math.pi))
            y = int(cy + r * math.sin(t + arm * math.pi))
            points.append((x, y))
        draw.line(points, fill=(100,200,255,brightness), width=1)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(220,220,255,255))
    return g.filter(ImageFilter.GaussianBlur(0.7))

def draw_lenticular(size, brightness):
    g = Image.new('RGBA', (size*3, size*3), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(2, 0, -1):
        rx = int(size * (i/2))
        ry = int(size * (i/2) * 0.3)
        alpha = int(brightness * (i/2) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(180,220,255,alpha))
    draw.line([cx-size//2, cy, cx+size//2, cy], fill=(200,220,255,180), width=2)
    return g.filter(ImageFilter.GaussianBlur(0.5))

def draw_elliptical(size, brightness):
    g = Image.new('RGBA', (size*3, size*3), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(4,0,-1):
        rx = int(size * (i/4))
        ry = int(size * (i/4) * 0.6)
        alpha = int(brightness * (i/4) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(150,220,255,alpha))
    return g.filter(ImageFilter.GaussianBlur(0.7))

def draw_irregular(size, brightness):
    g = Image.new('RGBA', (size*3, size*3), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for _ in range(size*6):
        dx = random.randint(-size, size)
        dy = random.randint(-size, size)
        draw.point((cx+dx, cy+dy), fill=(120,200,255,brightness))
    return g.filter(ImageFilter.GaussianBlur(0.7))
