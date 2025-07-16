import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from PIL import Image, ImageDraw
import random


def plot_galaxy_map_c(df, image_size=(2048, 2048), background_color=(0, 0, 0)):
    """
    Dibuja la imagen simulada de galaxias con tamaños aparentes corregidos por distancia comóvil.
    """
    img = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(img)

    ra_min, ra_max = df['RA'].min(), df['RA'].max()
    dec_min, dec_max = df['Dec'].min(), df['Dec'].max()

    # Normalizar tamaño por distancia
    df['size_factor'] = 1 / df['Dist']
    df['size_factor'] /= df['size_factor'].median()

    for _, row in df.iterrows():
        ra, dec = row['RA'], row['Dec']
        color_val = row.get('(r-i)', 0.4)
        mag_rf = row.get('Rf', -22)
        dist = row.get('Dist', 100)

        # Coordenadas a píxeles (RA invertido para orientación astronómica)
        x = int(image_size[0] * (1 - (ra - ra_min) / (ra_max - ra_min)))
        y = int(image_size[1] * (1 - (dec - dec_min) / (dec_max - dec_min)))

        # Color por índice (r-i)
        r = int(min(255, max(0, 255 * (1.5 - color_val))))
        b = int(min(255, max(0, 255 * color_val)))
        g = int((r + b) / 4)
        color = (r, g, b)

        # Tamaño base y corrección
        size = max(2, int(20 - abs(mag_rf) / 2))
        size = int(size * row['size_factor'] * random.uniform(0.9, 1.1))

        # Dibujar
        bbox = [x - size, y - size, x + size, y + size]
        draw.ellipse(bbox, fill=color)

    return img
