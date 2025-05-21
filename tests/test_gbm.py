import numpy as np
import gbm # Assuming gbm.py is in the same directory or PYTHONPATH

def test_generate_gbm_paths_shape_and_initial_price():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    delta = 0.0
    n_steps = 5
    n_paths = 3

    paths = gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths)

    assert paths.shape == (n_paths, n_steps + 1), "Shape mismatch for standard paths"
    assert np.all(paths[:, 0] == S0), "Initial prices are not S0 for standard paths"

def test_generate_gbm_paths_antithetic():
    """
    Tests the antithetic sampling property of the generate_gbm_paths function.
    """
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    delta = 0.0
    n_steps = 10
    n_paths = 4  # Must be even for antithetic

    # 1. Verify shape and initial prices
    paths_anti = gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths, antithetic=True)
    assert paths_anti.shape == (n_paths, n_steps + 1), "Shape mismatch for antithetic paths"
    assert np.all(paths_anti[:, 0] == S0), "Initial prices are not S0 for antithetic paths"

    # 2. Verify antithetic property using random seed
    seed = 42
    half_n_paths = n_paths // 2

    # Generate standard paths for the first half using the seed
    np.random.seed(seed)
    paths_half_std = gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, half_n_paths, antithetic=False)

    # Generate full antithetic paths using the same seed
    np.random.seed(seed)
    paths_anti_full = gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths, antithetic=True)

    # The first half of antithetic paths should be very close to standard paths
    assert np.allclose(paths_anti_full[:half_n_paths, :], paths_half_std), \
        "First half of antithetic paths does not match standard generated paths with the same seed"

    # Check the log return relationship for antithetic pairs
    dt = T / n_steps
    nudt = (r - delta - 0.5 * sigma**2) * dt

    for k in range(half_n_paths):
        for t_idx in range(n_steps): # t_idx is the index for Z, paths are indexed from 0 to n_steps
            # log(S_t+1 / S_t) = nudt + sigsdt * Z
            # (log(S_t+1 / S_t) - nudt) / sigsdt = Z
            log_return_k = np.log(paths_anti_full[k, t_idx + 1] / paths_anti_full[k, t_idx])
            log_return_k_anti = np.log(paths_anti_full[k + half_n_paths, t_idx + 1] / paths_anti_full[k + half_n_paths, t_idx])
            
            # (log_return_k - nudt) should be approximately -(log_return_k_anti - nudt)
            # This means (log_return_k - nudt) + (log_return_k_anti - nudt) should be close to 0
            # Or log_return_k + log_return_k_anti should be close to 2 * nudt
            assert np.isclose(log_return_k - nudt, -(log_return_k_anti - nudt), atol=1e-9), \
                f"Antithetic relationship for Z not held for path pair {k} and {k+half_n_paths} at step {t_idx+1}"
            assert np.isclose(log_return_k + log_return_k_anti, 2 * nudt, atol=1e-9), \
                f"Sum of log returns for antithetic pair {k} and {k+half_n_paths} at step {t_idx+1} not close to 2*nudt"

    # 3. Test ValueError for odd n_paths with antithetic=True
    try:
        gbm.generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths + 1, antithetic=True)
        assert False, "ValueError not raised for odd n_paths with antithetic=True"
    except ValueError as e:
        assert str(e) == "n_paths must be even when antithetic sampling is enabled"
    except Exception as e:
        assert False, f"Unexpected exception {type(e)} raised instead of ValueError for odd n_paths"

def test_generate_gbm_paths_statistical_properties():
    """
    Tests basic statistical properties of the generate_gbm_paths function.
    """
    # Scenario 1: r=0, sigma=0.1. Expected S_T should be S0.
    S0_1 = 100.0
    r_1 = 0.0
    sigma_1 = 0.1
    T_1 = 1.0
    delta_1 = 0.0
    n_steps_1 = 20
    n_paths_1 = 2000  # Increased for better statistical significance

    paths_1 = gbm.generate_gbm_paths(S0_1, r_1, sigma_1, T_1, delta_1, n_steps_1, n_paths_1, antithetic=False)
    
    expected_S_T_1 = S0_1 * np.exp((r_1 - delta_1) * T_1)
    mean_terminal_price_1 = np.mean(paths_1[:, -1])
    
    # Check if the mean terminal price is close to the expected value.
    # Tolerance is relative to S0; for r=0, it's just S0.
    # With 2000 paths, we expect a tighter bound than 5-10%.
    # Std Error of mean ~ sigma*S0/sqrt(n_paths*n_steps_for_T_variance) approx 0.1*100/sqrt(2000) ~ 10/44 ~ 0.22
    # So, within 3*SE should be ~0.66. Let's use 2% of S0 as tolerance for the mean.
    assert np.isclose(mean_terminal_price_1, expected_S_T_1, rtol=0.03), \
        f"Mean terminal price {mean_terminal_price_1} is not close to expected {expected_S_T_1} for r=0, sigma=0.1"

    # Scenario 2: r=0.05, sigma=0.0. All paths should be deterministic.
    S0_2 = 100.0
    r_2 = 0.05
    sigma_2 = 0.0
    T_2 = 1.0
    delta_2 = 0.0
    n_steps_2 = 10
    n_paths_2 = 2 # Only need a few paths as they should be identical

    paths_2 = gbm.generate_gbm_paths(S0_2, r_2, sigma_2, T_2, delta_2, n_steps_2, n_paths_2, antithetic=False)
    
    expected_S_T_2 = S0_2 * np.exp((r_2 - delta_2) * T_2)
    
    # All paths should be identical and deterministic
    for k in range(n_paths_2):
        assert np.isclose(paths_2[k, -1], expected_S_T_2), \
            f"Path {k} terminal price {paths_2[k, -1]} is not equal to deterministic expected {expected_S_T_2} for sigma=0"
        # Also check intermediate steps for one path
        if k == 0:
            for t_idx in range(n_steps_2 + 1):
                time_t = (T_2 / n_steps_2) * t_idx
                expected_S_t = S0_2 * np.exp((r_2 - delta_2) * time_t)
                assert np.isclose(paths_2[k, t_idx], expected_S_t), \
                    f"Path {k} price {paths_2[k, t_idx]} at step {t_idx} is not equal to deterministic expected {expected_S_t} for sigma=0"
