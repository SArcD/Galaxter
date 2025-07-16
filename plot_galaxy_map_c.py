from PIL import Image, ImageDraw
import numpy as np
import random
from astropy.cosmology import FlatLambdaCDM
import streamlit as st

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.stats import gaussian_kde
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import streamlit as st
import random
import math

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


def plot_galaxy_map_cosmology(df, ra_col='RA', dec_col='Dec', morph_col='M(ave)', rf_col='Rf',
                               subcluster_col='Subcluster', subsubcluster_col="Subcluster_1",
                               width=1024, height=1024):
    
    # 🌌 Cosmología plana
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DEFAULT_REDSHIFT = 0.0555  # Redshift medio de Abell 85

    st.header("🌌 Simulación visual del Cúmulo (Tamaños corregidos por redshift z=0.0555)")
    st.markdown("Esta visualización incluye: halo de Perlin, estrellas de campo, galaxias según morfología y halos locales por subcluster.")

    # ---------------------- Filtros en sidebar ----------------------
    show_stars = st.sidebar.checkbox("Mostrar estrellas de campo", value=True, key=f"dd")
    morphs = sorted(df[morph_col].dropna().unique())
    morph_filter = st.sidebar.multiselect("Filtrar morfología", morphs, default=morphs, key=f"ddd")

    all_subclusters = sorted(set(df[subcluster_col].dropna()))
    st.sidebar.markdown("### Subclusters visibles")
    subcluster_visibility = {str(sub): st.sidebar.checkbox(f"Subcluster {sub}", value=True, key=f"sub_{sub}") for sub in all_subclusters}

    all_subsubclusters = sorted(set(df[subsubcluster_col].dropna()))
    st.sidebar.markdown("### Sub-Subclusters visibles")
    subsubcluster_visibility = {str(sub): st.sidebar.checkbox(f"SubSubcluster {sub}", value=True, key=f"subsub_{sub}") for sub in all_subsubclusters}

    # ---------------------- Filtro de DataFrame ----------------------
    df_filtered = df[
        (df[morph_col].isin(morph_filter)) &
        (df[subcluster_col].astype(str).isin([k for k, v in subcluster_visibility.items() if v])) &
        (df[subsubcluster_col].astype(str).isin([k for k, v in subsubcluster_visibility.items() if v]))
    ].dropna()

    if df_filtered.empty:
        st.warning("No hay datos con los filtros actuales.")
        return

    RA_min, RA_max = df[ra_col].min(), df[ra_col].max()
    Dec_min, Dec_max = df[dec_col].min(), df[dec_col].max()

    # ---------------------- Fondo base ----------------------
    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    img.alpha_composite(generate_perlin_halo(width, height))

    # ---------------------- Estrellas de campo ----------------------
    if show_stars:
        field = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw_field = ImageDraw.Draw(field)
        for _ in range(1500):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.uniform(0.5, 1.8)
            brightness = random.randint(150, 255)
            tint = random.choice([(255,255,255), (200,220,255), (255,240,200)])
            draw_field.ellipse([x - r, y - r, x + r, y + r], fill=(tint[0], tint[1], tint[2], brightness))
        field_blur = field.filter(ImageFilter.GaussianBlur(0.5))
        img.alpha_composite(field_blur)

    # ---------------------- Dibujo de galaxias ----------------------
    for _, row in df_filtered.iterrows():
        morph = classify_morphology(row[morph_col])
        try:
            DA = cosmo.angular_diameter_distance(DEFAULT_REDSHIFT).to(u.Mpc).value
            angular_size_kpc = 5
            theta_rad = angular_size_kpc / (DA * 1000)
            theta_arcsec = theta_rad * (180 / math.pi) * 3600
            size = max(2, min(int(theta_arcsec / 2), 50))
        except:
            size = 5

        size = int(size * random.uniform(0.8, 1.2))
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

    # ---------------------- Halos KDE por subcluster ----------------------
    subcluster_positions = df_filtered.groupby(subcluster_col)[[ra_col, dec_col]].mean().reset_index()
    for _, row in subcluster_positions.iterrows():
        galaxies = df_filtered[df_filtered[subcluster_col] == row[subcluster_col]]
        if galaxies.empty:
            continue

        coords = galaxies[[ra_col, dec_col]].values.T
        kde = gaussian_kde(coords, bw_method='scott')

        grid_size = 300
        xgrid = np.linspace(RA_min, RA_max, grid_size)
        ygrid = np.linspace(Dec_min, Dec_max, grid_size)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        threshold = np.percentile(Z, 80)
        mask_array = (Z > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_array).convert("L")
        mask_img = mask_img.resize((grid_size, grid_size), resample=Image.BILINEAR)
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(40))

        halo_rgba = Image.new('RGBA', mask_blurred.size, (0, 180, 150, 0))
        alpha = mask_blurred.point(lambda p: int(p * 0.25))
        halo_rgba.putalpha(alpha)
        halo_resized = halo_rgba.resize((width, height), resample=Image.BILINEAR)
        img.alpha_composite(halo_resized)

    # ---------------------- Final: espejo y despliegue ----------------------
    img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    st.image(img, caption="🌠 Tamaños aparentes corregidos por cosmología (z=0.0555)")
    st.dataframe(df_filtered)

    return img

