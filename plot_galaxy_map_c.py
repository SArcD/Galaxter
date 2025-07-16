def plot_galaxy_map_with_distances(df, ra_col='RA', dec_col='Dec', morph_col='M(ave)', rf_col='Rf',
                                   subcluster_col='Subcluster', subsubcluster_col="Subcluster_1",
                                   distance_col='Distance_Mpc', width=1024, height=1024):
    import streamlit as st
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    import random
    import math
    from scipy.stats import gaussian_kde

    # -------------------------------------------------------
    # Funciones auxiliares (recorta si ya las tienes)
    # -------------------------------------------------------
    def classify_morphology(morph_str):
        morph_str = morph_str.lower()
        if morph_str.startswith('e'):
            return 'elliptical'
        elif 's0' in morph_str:
            return 'lenticular'
        elif 'sa' in morph_str:
            return 'spiral_sa'
        elif 'sb' in morph_str:
            return 'spiral_sb'
        elif 'sc' in morph_str:
            return 'spiral_sc'
        elif 'sd' in morph_str:
            return 'spiral_sd'
        else:
            return 'irregular'

    def draw_spiral_type(size, spiral_type='sa'):
        g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
        draw = ImageDraw.Draw(g)
        cx, cy = size, size
        tightness = {'sa': 0.4, 'sb': 0.7, 'sc': 1.0, 'sd': 1.4}.get(spiral_type, 0.8)
        tint = random.choice([(220, 240, 255), (200, 220, 255), (240, 240, 255)])
        for arm in range(2):
            theta_offset = (2 * math.pi / 2) * arm
            points = []
            for i in range(150):
                t = i / 150 * (2 * math.pi)
                r = size * (i / 150) * tightness
                if spiral_type == 'sd' and random.random() < 0.05:
                    continue
                x = cx + r * math.cos(t + theta_offset)
                y = cy + r * math.sin(t + theta_offset)
                points.append((x, y))
            draw.line(points, fill=(tint[0], tint[1], tint[2], 255), width=1)
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 255, 255, 255))
        return g.filter(ImageFilter.GaussianBlur(1))

    def draw_elliptical(size, brightness):
        g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
        draw = ImageDraw.Draw(g)
        cx, cy = size, size
        tint = random.choice([(180, 220, 255), (200, 240, 255), (220, 220, 255)])
        for i in range(5, 0, -1):
            rx = int(size * (i / 5))
            ry = int(size * (i / 5) * 0.6)
            alpha = int(brightness * (i / 5) * 0.5)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(tint[0], tint[1], tint[2], alpha))
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255,255,255,255))
        return g.filter(ImageFilter.GaussianBlur(1))

    def draw_lenticular(size, brightness):
        g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
        draw = ImageDraw.Draw(g)
        cx, cy = size, size
        tint = random.choice([(200, 220, 255), (255, 230, 200), (220, 220, 240)])
        for i in range(2, 0, -1):
            rx = int(size * (i/2))
            ry = int(size * (i/2) * 0.3)
            alpha = int(brightness * (i/2) * 0.5)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(tint[0], tint[1], tint[2], alpha))
        draw.line([cx-size//2, cy, cx+size//2, cy], fill=(200,220,255,180), width=1)
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255,255,255,255))
        return g.filter(ImageFilter.GaussianBlur(1))

    def draw_irregular(size, brightness):
        g = Image.new('RGBA', (size*2, size*2), (0,0,0,0))
        draw = ImageDraw.Draw(g)
        cx, cy = size, size
        tint = random.choice([(120, 200, 255), (180, 240, 200), (140, 220, 250)])
        for _ in range(size*4):
            dx = random.randint(-size, size)
            dy = random.randint(-size, size)
            draw.point((cx+dx, cy+dy), fill=(tint[0], tint[1], tint[2], brightness))
        draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(255,255,255,255))
        return g.filter(ImageFilter.GaussianBlur(1))

    def generate_perlin_halo(width, height, scale=0.02, octaves=1, alpha=150):
        from noise import pnoise2
        halo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        for x in range(width):
            for y in range(height):
                n = pnoise2(x * scale, y * scale, octaves=octaves)
                val = max(n + 0.5, 0)
                val = val ** 2.5
                a = int(alpha * val)
                halo.putpixel((x, y), (0, 180, 150, a))
        return halo.filter(ImageFilter.GaussianBlur(40))

    # -------------------------------------------------------
    # Ajustes dinámicos
    # -------------------------------------------------------
    st.header("Simulación visual del cúmulo (con distancia)")
    #show_stars = st.sidebar.checkbox("Mostrar estrellas de campo", value=True)
    show_stars = st.sidebar.checkbox("Mostrar estrellas de campo", value=True, key=f"0_stars")
    morph_col = "M(ave)"
    ra_col = "RA"
    dec_col = "Dec"
    rf_col = "Rf"

                                     
    morph_filter = st.sidebar.multiselect("Filtrar morfología", sorted(df[morph_col].dropna().unique()),
                                          default=sorted(df[morph_col].dropna().unique()),key=f"dsd")
    df_filtered = df[df[morph_col].isin(morph_filter)].dropna(subset=[ra_col, dec_col, morph_col, rf_col])

    if df_filtered.empty:
        st.warning("No hay datos disponibles con los filtros actuales.")
        return

    # Distancia: normaliza con log para evitar extremos
    if distance_col in df.columns:
        df_filtered = df_filtered[df_filtered[distance_col] > 0]
        distances = df_filtered[distance_col]
        dist_log_norm = np.log10(distances)
        min_log, max_log = dist_log_norm.min(), dist_log_norm.max()
        scale_factor = (max_log - dist_log_norm) / (max_log - min_log)  # más cerca → más grande
    else:
        scale_factor = np.ones(len(df_filtered))

    # Imagen base
    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()
    img = Image.new('RGBA', (width, height), (0,0,0,255))
    img.alpha_composite(generate_perlin_halo(width, height))

    # Estrellas
    if show_stars:
        field = Image.new('RGBA', (width, height), (0,0,0,0))
        draw_field = ImageDraw.Draw(field)
        for _ in range(1500):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.uniform(0.5, 1.8)
            brightness = random.randint(150, 255)
            tint = random.choice([(255,255,255), (200,220,255), (255,240,200)])
            draw_field.ellipse([x - r, y - r, x + r, y + r], fill=(tint[0], tint[1], tint[2], brightness))
        img.alpha_composite(field.filter(ImageFilter.GaussianBlur(0.5)))

    # Dibujo de galaxias
    for idx, row in df_filtered.iterrows():
        morph = classify_morphology(row[morph_col])
        try: mag_rf = float(row[rf_col])
        except: mag_rf = -15.0

        base_size = max(2, int(20 - abs(mag_rf)/2))
        size = int(base_size * scale_factor[idx] * random.uniform(0.9, 1.2))
        brightness = 255

        if morph.startswith('spiral'):
            galaxy = draw_spiral_type(size, spiral_type=morph.split('_')[-1])
        elif morph == 'elliptical':
            galaxy = draw_elliptical(size, brightness)
        elif morph == 'lenticular':
            galaxy = draw_lenticular(size, brightness)
        else:
            galaxy = draw_irregular(size, brightness)

        galaxy = galaxy.rotate(random.randint(0, 360), expand=True)
        x = int((row[ra_col] - RA_min) / (RA_max - RA_min) * width) - galaxy.width // 2
        y = int((row[dec_col] - Dec_min) / (Dec_max - Dec_min) * height) - galaxy.height // 2
        img.alpha_composite(galaxy, (x, y))

    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    st.image(img)
    st.dataframe(df_filtered)
    return img
