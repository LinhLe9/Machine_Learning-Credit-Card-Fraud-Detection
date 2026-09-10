import numpy as np

def sin_transformer(X, max_val=24):
    return np.sin(2 * np.pi * X / max_val)

def cos_transformer(X, max_val=24):
    return np.cos(2 * np.pi * X / max_val)