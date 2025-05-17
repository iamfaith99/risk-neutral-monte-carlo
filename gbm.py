import numpy as np


def generate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False,
) -> np.ndarray:
    """Generate asset price paths using Geometric Brownian Motion."""
    dt = T / n_steps
    nudt = (r - delta - 0.5 * sigma**2) * dt
    sigsdt = sigma * np.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    if antithetic:
        half_paths = n_paths // 2
        Z = np.random.normal(0, 1, (half_paths, n_steps))
        increments = np.exp(nudt + sigsdt * Z)
        paths[:half_paths, 1:] = S0 * np.cumprod(increments, axis=1)
        anti_increments = np.exp(nudt + sigsdt * (-Z))
        paths[half_paths:, 1:] = S0 * np.cumprod(anti_increments, axis=1)
    else:
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        increments = np.exp(nudt + sigsdt * Z)
        paths[:, 1:] = S0 * np.cumprod(increments, axis=1)

    return paths
