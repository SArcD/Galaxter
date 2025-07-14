import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
from noise import pnoise2

# ✅ Generador de halo Perlin global

#def generate_perlin_halo(width, height, scale=0.02, octaves=1, alpha=100):
#    halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
#    for x in range(width):
#        for y in range(height):
#            n = pnoise2(x * scale, y * scale, octaves=octaves)
#            val = int(200 * (n + 0.5))
 #           a = min(max(val, 0), alpha)
 #           halo.putpixel((x, y), (0, 180, 150, int(a)))
#    return halo.filter(ImageFilter.GaussianBlur(40))

def generate_perlin_halo(width, height, scale=0.02, octaves=1, alpha=150):
    halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for x in range(width):
        for y in range(height):
            n = pnoise2(x * scale, y * scale, octaves=octaves)
            val = max(n + 0.5, 0)  # Normaliza
            val = val ** 2.5        # Ajusta contraste: más exponente = menos brillo base
            a = int(alpha * val)
            halo.putpixel((x, y), (0, 180, 150, a))
    return halo.filter(ImageFilter.GaussianBlur(40))


# Clasificador de morfología

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

# Funciones draw_*

def draw_spiral(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    arms = 2
    for arm in range(arms):
        theta_offset = (2 * math.pi / arms) * arm
        points = []
        for i in range(150):
            t = i / 150 * (2 * math.pi)
            r = size * (i / 150)
            x = cx + r * math.cos(t + theta_offset)
            y = cy + r * math.sin(t + theta_offset)
            points.append((x, y))
        draw.line(points, fill=(200, 220, 255, 255), width=1)
    draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_elliptical(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(5, 0, -1):
        rx = int(size * (i / 5))
        ry = int(size * (i / 5) * 0.6)
        alpha = int(brightness * (i / 5) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(150, 220, 255, alpha))
    draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_lenticular(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for i in range(2, 0, -1):
        rx = int(size * (i/2))
        ry = int(size * (i/2) * 0.3)
        alpha = int(brightness * (i/2) * 0.5)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(180, 220, 255, alpha))
    draw.line([cx-size//2, cy, cx+size//2, cy], fill=(200,220,255,180), width=1)
    draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

def draw_irregular(size, brightness):
    g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
    draw = ImageDraw.Draw(g)
    cx, cy = size, size
    for _ in range(size*4):
        dx = random.randint(-size, size)
        dy = random.randint(-size, size)
        draw.point((cx+dx, cy+dy), fill=(120, 200, 255, brightness))
    draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(255, 255, 255, 255))
    return g.filter(ImageFilter.GaussianBlur(1))

# Función principal

def plot_galaxy_map(df, ra_col='RA', dec_col='Dec', morph_col='M(ave)', subcluster_col='Subcluster', rf_col='Rf', width=1024, height=1024):
    st.header("Mapa de Cúmulo con Halo Perlin + Subclusters")
    all_morphs = sorted(df[morph_col].dropna().unique())
    all_subclusters = sorted(df[subcluster_col].dropna().unique())
    morph_filter = st.sidebar.multiselect("Filtrar morfología", all_morphs, default=all_morphs)
    subcluster_filter = st.sidebar.multiselect("Filtrar Subclusters", all_subclusters, default=all_subclusters)
    df_filtered = df[df[morph_col].isin(morph_filter) & df[subcluster_col].isin(subcluster_filter)].dropna()
    if df_filtered.empty:
        st.warning("No hay datos.")
        return

    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()

    img = Image.new('RGBA', (width, height), (0,0,0,255))
    img.alpha_composite(generate_perlin_halo(width, height))

    for _, row in df_filtered.iterrows():
        RA, Dec, morph_raw = row[ra_col], row[dec_col], row[morph_col]
        morph = classify_morphology(morph_raw)
        try: mag_rf = float(row[rf_col])
        except: mag_rf = -15.0
        size = max(2, int(20 - abs(mag_rf)/2))
        brightness = 255
        if morph == 'spiral': galaxy = draw_spiral(size, brightness)
        elif morph == 'elliptical': galaxy = draw_elliptical(size, brightness)
        elif morph == 'lenticular': galaxy = draw_lenticular(size, brightness)
        else: galaxy = draw_irregular(size, brightness)
        galaxy = galaxy.rotate(random.randint(0, 360), expand=True)
        x = int((RA - RA_min) / (RA_max - RA_min) * width) - galaxy.width // 2
        y = int((Dec - Dec_min) / (Dec_max - Dec_min) * height) - galaxy.height // 2
        img.alpha_composite(galaxy, (x, y))

#    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()
#    for _, row in subcluster_positions.iterrows():
#        num_galaxias = len(df_filtered[df_filtered[subcluster_col]==row[subcluster_col]])
#        if num_galaxias == 0: continue
#        halo_alpha = min(200, max(10, num_galaxias))
#        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
#        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)
#        local_halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
#        draw_local = ImageDraw.Draw(local_halo)
#        draw_local.ellipse([cx-150, cy-150, cx+150, cy+150], fill=(255, 160, 50, halo_alpha))
#        local_blurred = local_halo.filter(ImageFilter.GaussianBlur(60))
#        img.alpha_composite(local_blurred)

    
#    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()

#    for _, row in subcluster_positions.iterrows():
        # 🔍 Filtra galaxias de este subcluster
#        galaxies_in_subcluster = df_filtered[df_filtered[subcluster_col] == row[subcluster_col]]
#        num_galaxias = len(galaxies_in_subcluster)
#        if num_galaxias == 0:
#            continue

#        # ✅ 1. Calcula orientación (covarianza + eigenvectors)
#        coords = galaxies_in_subcluster[[ra_col, dec_col]].values
 #       coords -= coords.mean(axis=0)  # centra
#        cov = np.cov(coords, rowvar=False)
#        eigvals, eigvecs = np.linalg.eigh(cov)
#        angle_rad = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
#        angle_deg = np.degrees(angle_rad)

#        # ✅ 2. Define forma elíptica
#        halo_alpha = min(200, max(10, num_galaxias))
#        rx = 200  # Eje mayor
#        ry = 100  # Eje menor

#        # ✅ 3. Dibuja halo elíptico pequeño
#        single_halo = Image.new('RGBA', (rx * 2, ry * 2), (0, 0, 0, 0))
#        draw_local = ImageDraw.Draw(single_halo)
#        draw_local.ellipse([0, 0, rx * 2, ry * 2], fill=(255, 160, 50, halo_alpha))
#        blurred = single_halo.filter(ImageFilter.GaussianBlur(60))

#        # ✅ 4. Rota para alinearlo
#        rotated = blurred.rotate(-angle_deg, expand=True)

#        # ✅ 5. Calcula posición global
#        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
#        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)

#        # ✅ 6. Pega halo rotado centrado
#        offset_x = cx - rotated.width // 2
#        offset_y = cy - rotated.height // 2
#        img.alpha_composite(rotated, (offset_x, offset_y))

#        import numpy as np

#    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()

#    for _, row in subcluster_positions.iterrows():
#        # 🔍 Filtra galaxias de este subcluster
#        galaxies_in_subcluster = df_filtered[df_filtered[subcluster_col] == row[subcluster_col]]
#        num_galaxias = len(galaxies_in_subcluster)
#        if num_galaxias == 0:
#            continue

#        # ✅ 1. Calcula orientación (covarianza + eigenvectors)
#        coords = galaxies_in_subcluster[[ra_col, dec_col]].values
#        coords -= coords.mean(axis=0)  # centra
#        cov = np.cov(coords, rowvar=False)
#        eigvals, eigvecs = np.linalg.eigh(cov)
#        angle_rad = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
#        angle_deg = np.degrees(angle_rad)

#        # ✅ 2. Define forma elíptica
#        halo_alpha = min(200, max(10, num_galaxias))
#        rx = 200  # Eje mayor
#        ry = 100  # Eje menor

#        # ✅ 3. Dibuja halo elíptico pequeño
#        single_halo = Image.new('RGBA', (rx * 2, ry * 2), (0, 0, 0, 0))
#        draw_local = ImageDraw.Draw(single_halo)
#        draw_local.ellipse([0, 0, rx * 2, ry * 2], fill=(255, 160, 50, halo_alpha))
#        blurred = single_halo.filter(ImageFilter.GaussianBlur(60))

#        # ✅ 4. Rota para alinearlo
#        rotated = blurred.rotate(-angle_deg, expand=True)

#        # ✅ 5. Calcula posición global
#        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
#        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)

        # ✅ 6. Pega halo rotado centrado
#        offset_x = cx - rotated.width // 2
#        offset_y = cy - rotated.height // 2
#        img.alpha_composite(rotated, (offset_x, offset_y))


    import numpy as np
    from PIL import ImageOps, ImageChops, Image

    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()

    for _, row in subcluster_positions.iterrows():
        galaxies_in_subcluster = df_filtered[df_filtered[subcluster_col] == row[subcluster_col]]
        num_galaxias = len(galaxies_in_subcluster)
        if num_galaxias == 0:
            continue

        # 1️⃣ Calcula orientación
        coords = galaxies_in_subcluster[[ra_col, dec_col]].values
        coords -= coords.mean(axis=0)
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle_rad = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
        angle_deg = np.degrees(angle_rad)

        # 2️⃣ Tamaño y opacidad
        halo_alpha = min(200, max(10, num_galaxias))
        rx = 200
        ry = 100

        # 3️⃣ Dibuja halo elíptic  o
        single_halo = Image.new('RGBA', (rx * 2, ry * 2), (0, 0, 0, 0))
        draw_local = ImageDraw.Draw(single_halo)
        draw_local.ellipse([0, 0, rx * 2, ry * 2], fill=(255, 160, 50, halo_alpha))

        # 4️⃣ Aplica blur
        blurred = single_halo.filter(ImageFilter.GaussianBlur(60))

        # 5️⃣ Crea máscara circular suave 
        mask = Image.new('L', blurred.size, 0)
        draw_mask = ImageDraw.Draw(mask)  
        draw_mask.ellipse([0, 0, rx * 2, ry * 2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(60))

        # 6️⃣ Aplica la máscara para recortar bordes
        blurred.putalpha(mask)

        # 7️⃣ Rota
        rotated = blurred.rotate(-angle_deg, expand=True)

        # 8️⃣ Encuentra posición global
        cx = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width)
        cy = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height)

        # 9️⃣ Pega centrado
        offset_x = cx - rotated.width // 2
        offset_y = cy - rotated.height // 2
        img.alpha_composite(rotated, (offset_x, offset_y))

 

 
    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    #img = img.transpose(Image.ROTATE_180)
    st.image(img)
    st.dataframe(df_filtered)
    return img
