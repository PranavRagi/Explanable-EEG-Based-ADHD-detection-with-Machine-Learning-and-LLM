def dct_feats(df):
    # ======================================================
    #  DCT FEATURES (same as notebook)
    # ======================================================
    n_coef = config["dct_coefficients"]  # e.g. 40
    dct_feats = []

    for ep in epochs:
        coeffs = dct(ep, axis=0, norm="ortho")[:n_coef, :]
        dct_feats.append(coeffs.flatten())

    dct_feats = np.array(dct_feats)