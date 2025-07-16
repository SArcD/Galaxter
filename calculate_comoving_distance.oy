def calculate_comoving_distance(df, H0=70.0, Om0=0.3, z_cluster=0.0555):
    """
    Calcula la distancia comóvil para cada galaxia en el DataFrame a partir de su velocidad radial.
    """
    c = 3e5  # km/s
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    df = df.copy()

    # Redshift efectivo (considerando movimiento peculiar)
    df['z_gal'] = z_cluster + (df['Vel'] / c) * (1 + z_cluster)
    df['Dist'] = cosmo.comoving_distance(df['z_gal']).value  # Mpc

    return df

