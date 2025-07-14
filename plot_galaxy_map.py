import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
from noise import pnoise2

# --- Generador de halo Perlin ---
def generate_perlin_halo(width, height, scale=0.02, octaves=2, alpha=120):
    halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for x in range(width):
        for y in range(height):
            n = pnoise2(x * scale, y * scale, octaves=octaves)
            val = int(200 * (n + 0.5))
            a = int(min(max(val, 0), int(alpha * 2)))
            a = min(a, 255)
            halo.putpixel((x, y), (0, 220, 180, a))
    return halo.filter(ImageFilter.GaussianBlur(90))

# --- Clasificador de morfología ---
def classify_morphology(morph_str):
    morph_str = morph_str.lower()
    if morph_str.startswith('e'):
        return 'elliptical'
    elif 's0' in morph_str:
        return 'lenticular'
    elif any(s in morph_str for s in ['sa', 'sb', 'sc', 'sdm']):
        return 'spiral'
    else:
        return 'irregular'

# --- Formas de galaxias realistas ---
def draw_spiral(size, brightness):
    g = Image.new('RGBA', (size*8, size*8), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    arms = 4
    for arm in range(arms):
        theta_offset = (2 * math.pi / arms) * arm
        points = []
        for i in range(300):
            t = i / 300 * (4 * 2 * math.pi)
            r = size * (i / 300)
            x = cx + r * math.cos(t + theta_offset)
            y = cy + r * math.sin(t + theta_offset)
            points.append((x, y))
        draw.line(points, fill=(200, 220, 255, 255), width=2)
    draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_elliptical(size, brightness):
    g = Image.new('RGBA', (size*8, size*8), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(5, 0, -1):
        rx = int(size * (i / 5))
        ry = int(size * (i / 5) * 0.6)
        alpha = int(brightness * (i / 5) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(150, 220, 255, alpha))
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_lenticular(size, brightness):
    g = Image.new('RGBA', (size*8, size*8), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(2, 0, -1):
        rx = int(size * (i/2))
        ry = int(size * (i/2) * 0.3)
        alpha = int(brightness * (i/2) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(180, 220, 255, alpha))
    draw.line([cx-size//2, cy, cx+size//2, cy], fill=(200,220,255,180), width=2)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_irregular(size, brightness):
    g = Image.new('RGBA', (size*8, size*8), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for _ in range(size*6):
        dx = random.randint(-size, size)
        dy = random.randint(-size, size)
        draw.point((cx+dx, cy+dy), fill=(120, 200, 255, brightness))
    draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

# --- Función principal ---
def plot_galaxy_map(df, ra_col='RA', dec_col='Dec', morph_col='M(ave)', subcluster_col='Subcluster', rf_col='Rf', width=1024, height=1024):
    st.header("Mapa de Cúmulo con Halo Perlin + Subclusters")
    all_morphs = sorted(df[morph_col].dropna().unique())
    all_subclusters = sorted(df[subcluster_col].dropna().unique())
    morph_filter = st.sidebar.multiselect("Filtrar morfología", all_morphs, default=all_morphs)
    subcluster_filter = st.sidebar.multiselect("Filtrar Subclusters", all_subclusters, default=all_subclusters)
    df_filtered = df[df[morph_col].isin(morph_filter) & df[subcluster_col].isin(subcluster_filter)].dropna(subset=[ra_col, dec_col, rf_col])
    if df_filtered.empty:
        st.warning("No hay datos.")
        return

    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()

    img = Image.new('RGBA', (width, height), (0,0,0,255))
    img.alpha_composite(generate_perlin_halo(width, height))

    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()
    for _, row in subcluster_positions.iterrows():
        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)
        local_halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_local = ImageDraw.Draw(local_halo)
        draw_local.ellipse([cx-200, cy-200, cx+200, cy+200], fill=(0, 200, 180, 40))
        local_blurred = local_halo.filter(ImageFilter.GaussianBlur(50))
        img.alpha_composite(local_blurred)

    for _, row in df_filtered.iterrows():
        RA, Dec, morph_raw = row[ra_col], row[dec_col], row[morph_col]
        morph = classify_morphology(morph_raw)
        try: mag_rf = float(row[rf_col])
        except: mag_rf = -15.0
        size = max(80, int(120 - abs(mag_rf)))
        brightness = 255
        if morph == 'spiral':
            galaxy = draw_spiral(size, brightness)
        elif morph == 'elliptical':
            galaxy = draw_elliptical(size, brightness)
        elif morph == 'lenticular':
            galaxy = draw_lenticular(size, brightness)
        else:
            galaxy = draw_irregular(size, brightness)
        galaxy = galaxy.rotate(random.randint(0, 360), expand=True)
        x = int((RA - RA_min) / (RA_max - RA_min) * width) - galaxy.width // 2
        y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height) - galaxy.height // 2
        img.alpha_composite(galaxy, (x, y))

    for _ in range(2000):
        sx, sy = random.randint(0, width - 1), random.randint(0, height - 1)
        b = random.randint(120, 220)
        img.putpixel((sx, sy), (b, b, b, 255))

    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    with st.expander("Mapa Generado"):
        st.image(img)
        st.dataframe(df_filtered)
    return img
