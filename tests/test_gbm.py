import numpy as np
import gbm


def test_generate_gbm_paths_shape_and_initial_price():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    delta = 0.0
    n_steps = 5
    n_paths = 3

    paths = gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths)

    assert paths.shape == (n_paths, n_steps + 1)
    assert np.all(paths[:, 0] == S0)
