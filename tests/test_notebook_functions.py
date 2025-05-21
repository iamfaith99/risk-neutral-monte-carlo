import numpy as np
import pytest
from scipy.stats import norm
from tests.notebook_helpers import (
    geometric_asian_call_price,
    generate_sv_paths,
    continuous_lookback_call_price,
    arithmetic_asian_call_mc,
    lookback_call_mc
)

# Common parameters for tests
S0_default = 100.0
K_default = 100.0
r_default = 0.05
sigma_default = 0.2
T_default = 1.0
delta_default = 0.02
n_steps_default = 10 # For Asian
V0_default = sigma_default**2
alpha_default = 5.0
V_bar_default = sigma_default**2
xi_default = 0.1 # Vol of vol
rho_default = -0.5
n_sv_steps_default = 52 # For SV/Lookback
n_paths_mc_default = 100 # Small for MC tests to run fast, use 100 for more stability than 4 or 10
n_paths_mc_antithetic_default = 100 # Must be even


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)

# --- Tests for geometric_asian_call_price ---

def test_geometric_asian_call_price_deep_itm():
    """ Test geometric Asian call price for a deep in-the-money option. """
    S0 = 200.0
    K = 100.0
    # Expected price approx. exp(-rT) * (S0*exp(a) - K)
    # nu = r - delta - 0.5 * sigma^2
    # a = (nu * T + 0.5 * sigma^2 * T * (n_steps + 1) / (2 * n_steps)) / n_steps
    nu = r_default - delta_default - 0.5 * sigma_default**2
    a = (nu * T_default + 0.5 * sigma_default**2 * T_default * (n_steps_default + 1) / (2 * n_steps_default)) / n_steps_default
    expected_approx_price = np.exp(-r_default * T_default) * (S0 * np.exp(a) - K)
    
    price = geometric_asian_call_price(S0, K, r_default, sigma_default, T_default, delta_default, n_steps_default)
    assert price > 0
    # This is a rough approximation, so use a wide tolerance or just check it's reasonably close
    assert np.isclose(price, expected_approx_price, rtol=0.2), \
        f"Price {price} not close to approximation {expected_approx_price} for deep ITM."

def test_geometric_asian_call_price_low_sigma():
    """ Test geometric Asian call price with very low sigma. """
    sigma_low = 1e-4
    # With low sigma, geometric average should be close to S0 * exp((r-delta-0.5*sigma^2)*T_avg_time)
    # where T_avg_time is a bit complex.
    # Easier: for low sigma, geometric_avg is close to S0 * exp( (r-delta) * T_adj ),
    # where T_adj is related to average time of observation.
    # The 'a' term captures this: S0 * exp(a) is the expected geometric mean at T
    nu = r_default - delta_default - 0.5 * sigma_low**2
    a = (nu * T_default + 0.5 * sigma_low**2 * T_default * (n_steps_default + 1) / (2 * n_steps_default)) / n_steps_default
    expected_future_geo_mean = S0_default * np.exp(a)
    expected_price = np.exp(-r_default * T_default) * max(0, expected_future_geo_mean - K_default)
    
    price = geometric_asian_call_price(S0_default, K_default, r_default, sigma_low, T_default, delta_default, n_steps_default)
    assert np.isclose(price, expected_price, atol=1e-3), \
        f"Price {price} not close to deterministic {expected_price} for low sigma."

def test_geometric_asian_call_price_K_zero():
    """ Test geometric Asian call price with K=0. """
    # Price should be S0 * exp(a) * exp(-rT)
    # which is S0 * exp(a - rT)
    # This is E[G_T] * exp(-rT)
    nu = r_default - delta_default - 0.5 * sigma_default**2
    a = (nu * T_default + 0.5 * sigma_default**2 * T_default * (n_steps_default + 1) / (2 * n_steps_default)) / n_steps_default
    expected_price = S0_default * np.exp(a) * np.exp(-r_default * T_default)
    
    price = geometric_asian_call_price(S0_default, 0, r_default, sigma_default, T_default, delta_default, n_steps_default)
    assert np.isclose(price, expected_price, rtol=1e-5)

# --- Tests for continuous_lookback_call_price ---

def test_continuous_lookback_call_price_T_zero():
    """ Test continuous lookback call price with T=0. """
    M = 110.0
    price = continuous_lookback_call_price(S0_default, K_default, r_default, sigma_default, 0, delta_default, M=M)
    assert np.isclose(price, max(0, M - K_default))

    price_no_M = continuous_lookback_call_price(S0_default, K_default, r_default, sigma_default, 0, delta_default)
    assert np.isclose(price_no_M, max(0, S0_default - K_default))


def test_continuous_lookback_call_price_atm_low_sigma():
    """ Test continuous lookback call with S0=K, M=S0, low sigma. Price should be small. """
    S0 = 100.0
    K = 100.0
    M = 100.0
    sigma_low = 1e-3
    price = continuous_lookback_call_price(S0, K, r_default, sigma_low, T_default, delta_default, M=M)
    # For very low sigma, if S0=M, price should be close to a regular call, and if S0=K that should be small
    # but not zero due to drift (r-delta)
    # A simple Black-Scholes call for reference (not exactly the same but gives magnitude)
    d1 = (np.log(S0/K) + (r_default - delta_default + 0.5*sigma_low**2)*T_default) / (sigma_low*np.sqrt(T_default))
    d2 = d1 - sigma_low*np.sqrt(T_default)
    bs_call_approx = S0 * np.exp(-delta_default*T_default)*norm.cdf(d1) - K*np.exp(-r_default*T_default)*norm.cdf(d2)
    assert price >= 0
    assert price < bs_call_approx * 2 + 0.1 # Should be small, use BS as a rough upper bound check
    # A more direct check: if sigma is tiny and S0=M=K, only drift can make it ITM.
    # If r-delta > 0, price will be >0. If r-delta <0, it might be very close to 0.
    expected_drift_effect = S0 * (np.exp((r_default-delta_default)*T_default) -1) * np.exp(-r_default*T_default) # very rough
    if (r_default - delta_default) > 0:
        assert price > 0 # Expect some value from positive drift
    else:
        assert np.isclose(price, 0, atol=0.1)


def test_continuous_lookback_call_price_high_sigma():
    """ Test continuous lookback call price with high sigma. Price should be higher. """
    sigma_high = 0.50
    price_low_sigma = continuous_lookback_call_price(S0_default, K_default, r_default, sigma_default, T_default, delta_default, M=S0_default)
    price_high_sigma = continuous_lookback_call_price(S0_default, K_default, r_default, sigma_high, T_default, delta_default, M=S0_default)
    assert price_high_sigma > price_low_sigma

# --- Tests for generate_sv_paths ---

def test_generate_sv_paths_shapes_initial_values():
    V0 = sigma_default**2
    S_paths, V_paths, S_max_paths = generate_sv_paths(
        S0_default, V0, r_default, delta_default, alpha_default, V_bar_default, xi_default,
        T_default, n_sv_steps_default, n_paths_mc_default
    )
    assert S_paths.shape == (n_paths_mc_default, n_sv_steps_default + 1)
    assert V_paths.shape == (n_paths_mc_default, n_sv_steps_default + 1)
    assert S_max_paths.shape == (n_paths_mc_default, n_sv_steps_default + 1)

    assert np.all(S_paths[:, 0] == S0_default)
    assert np.all(V_paths[:, 0] == V0)
    assert np.all(S_max_paths[:, 0] == S0_default)

def test_generate_sv_paths_variance_positivity():
    V0 = sigma_default**2
    _, V_paths, _ = generate_sv_paths(
        S0_default, V0, r_default, delta_default, alpha_default, V_bar_default, xi_default,
        T_default, n_sv_steps_default, n_paths_mc_default
    )
    assert np.all(V_paths >= 1e-7) # Due to max(..., 1e-6) and potential small float inaccuracies

def test_generate_sv_paths_s_max_logic():
    V0 = sigma_default**2
    S_paths, _, S_max_paths = generate_sv_paths(
        S0_default, V0, r_default, delta_default, alpha_default, V_bar_default, xi_default,
        T_default, n_sv_steps_default, n_paths_mc_default
    )
    # S_max_paths should be the cumulative maximum of S_paths along axis 1
    expected_S_max = np.maximum.accumulate(S_paths, axis=1)
    assert np.allclose(S_max_paths, expected_S_max)
    assert np.all(S_max_paths >= S_paths - 1e-9) # Check with tolerance for float issues


# --- Basic Integration Tests for MC Pricers ---
# These are stochastic, so we check for non-errors, basic properties (price > 0 for ITM)

@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("control_variate", [True, False])
def test_arithmetic_asian_call_mc_runs(antithetic, control_variate):
    n_paths_test = n_paths_mc_antithetic_default if antithetic else n_paths_mc_default
    if n_paths_test % 2 != 0 and antithetic: # Ensure even paths for antithetic
        n_paths_test +=1

    price, se, time = arithmetic_asian_call_mc(
        S0_default, K_default, r_default, sigma_default, T_default, delta_default,
        n_steps_default, n_paths_test, antithetic, control_variate
    )
    assert price >= 0
    assert se >= 0
    assert time > 0

    # Test ITM case
    S0_itm = 120.0
    price_itm, _, _ = arithmetic_asian_call_mc(
        S0_itm, K_default, r_default, sigma_default, T_default, delta_default,
        n_steps_default, n_paths_test, antithetic, control_variate
    )
    assert price_itm > 0

    # Test OTM case
    S0_otm = 80.0
    price_otm, _, _ = arithmetic_asian_call_mc(
        S0_otm, K_default, r_default, sigma_default, T_default, delta_default,
        n_steps_default, n_paths_test, antithetic, control_variate
    )
    # Price can be non-zero due to volatility, but should be small for far OTM.
    # For a simple test, just ensure it runs and is non-negative.
    # A tighter check might be np.isclose(price_otm, 0, atol= S0_default*0.05) for very far OTM,
    # but with small n_paths, this can be noisy.
    assert price_otm >= 0


@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("control_variate", [True, False])
@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_lookback_call_mc_runs(antithetic, control_variate, scheme):
    n_paths_test = n_paths_mc_antithetic_default if antithetic else n_paths_mc_default
    if n_paths_test % 2 != 0 and antithetic:
         n_paths_test +=1
        
    price, se, time = lookback_call_mc(
        S0_default, K_default, r_default, delta_default, sigma_default, alpha_default,
        V_bar_default, xi_default, T_default, n_sv_steps_default, n_paths_test,
        antithetic, control_variate, rho_default, scheme
    )
    assert price >= 0
    assert se >= 0
    assert time > 0

    # Test ITM case (S0 > K, and M will likely be > K)
    S0_itm = 120.0
    price_itm, _, _ = lookback_call_mc(
        S0_itm, K_default, r_default, delta_default, sigma_default, alpha_default,
        V_bar_default, xi_default, T_default, n_sv_steps_default, n_paths_test,
        antithetic, control_variate, rho_default, scheme
    )
    assert price_itm > 0

    # Test OTM case (S0 < K, but max price can still exceed K)
    S0_otm = 80.0
    price_otm, _, _ = lookback_call_mc(
        S0_otm, K_default, r_default, delta_default, sigma_default, alpha_default,
        V_bar_default, xi_default, T_default, n_sv_steps_default, n_paths_test,
        antithetic, control_variate, rho_default, scheme
    )
    assert price_otm >= 0 # Price for lookback can be substantial even if S0 < K
                         # due to the max operator.

# Example of a more specific value check for geometric_asian_call_price
# This requires known external validated values.
# For now, we use limiting cases and property checks.
# def test_geometric_asian_call_price_known_value():
#     # Parameters from a known source or previous validated run
#     S0, K, r, sigma, T, delta, n_steps = 100, 100, 0.05, 0.2, 1, 0.02, 10
#     expected_price = 4.77 # Replace with actual known value
#     price = geometric_asian_call_price(S0, K, r, sigma, T, delta, n_steps)
#     assert np.isclose(price, expected_price, rtol=1e-2) # Adjust tolerance as needed
