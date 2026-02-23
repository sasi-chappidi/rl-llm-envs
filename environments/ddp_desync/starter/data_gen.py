import numpy as np

def make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32)
    y_logit = X @ w + 0.25 * (X[:, 0] * X[:, 1]) - 0.1 * (X[:, 2] ** 2)
    y = (y_logit > np.median(y_logit)).astype(np.int64)
    return X, y