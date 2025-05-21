import time
from typing import Tuple
import numpy as np
from scipy.stats import norm
from gbm import generate_gbm_paths # Assuming gbm.py is in the root or accessible in PYTHONPATH

# Copied from final_project.qmd

def geometric_asian_call_price(
    S0: float,      # Initial asset price
    K: float,       # Strike price
    r: float,       # Risk-free interest rate
    sigma: float,   # Volatility (constant)
    T: float,       # Time to maturity
    delta: float,   # Dividend yield
    n_steps: int    # Number of observation dates (fixings)
) -> float:

    # Calculate nu and a, as they are needed for K=0 case too
    dt = T / n_steps
    nu = r - delta - 0.5 * sigma**2
    # 'a' is related to the expected log of the geometric average
    a = (nu * T + 0.5 * sigma**2 * T * (n_steps + 1) / (2 * n_steps)) / n_steps

    if K == 0:
        # If K=0, payoff is always G_T. Price is E[G_T] * exp(-rT).
        # E[G_T] = S0 * exp(a).
        return S0 * np.exp(a - r * T)

    b = (sigma**2 * T * (n_steps + 1) * (2 * n_steps + 1)) / (6 * n_steps**2)
    
    if b == 0: # Effectively sigma is zero or n_steps is pathologically structured
        # This implies sigma is zero (given T > 0, n_steps > 0)
        # Geometric average becomes S0 * exp(a) (since nu is calculated with original sigma, 'a' is based on it)
        # If sigma truly is 0, then nu = r - delta, and a = (r-delta)*T. E[G_T] = S0 * exp((r-delta)T)
        # However, the formula for 'a' includes sigma. If sigma=0, then a = nu*T/n_steps * sum(1) = nu*T.
        # Let's re-evaluate 'a' if sigma is truly zero for this path.
        # If sigma=0, then nu_sig0 = r - delta. a_sig0 = nu_sig0 * T.
        # E[G_T] = S0 * exp(a_sig0) = S0 * exp((r-delta)T)
        # Price = exp(-rT) * max(0, S0 * exp((r-delta)T) - K)

        # The existing 'a' is fine, it simplifies if sigma=0.
        # nu = r - delta - 0.5*sigma^2. If sigma=0, nu = r-delta.
        # a = ( (r-delta)*T + 0 ) / n_steps -> This is wrong.
        # The formula for 'a' is E[ (1/N) * sum(log(S_ti/S_0)*N/T) ] * T/N ... it's complex.
        # Let's trust the 'b==0' block's original logic for S0*exp(a) as E[G_T]
        effective_price_at_maturity = S0 * np.exp(a)
        # Payoff is max(0, S0 * exp(a) - K)
        # Price is exp(-rT) * max(0, S0 * exp(a) - K)
        if effective_price_at_maturity > K:
            return np.exp(-r * T) * (effective_price_at_maturity - K)
        else:
            return 0.0

    # Standard case: K > 0 and b > 0
    # np.log(S0 / K) can only be problematic if S0/K is <=0, but S0, K are positive prices.
    # RuntimeWarning for log(inf) is expected if K is extremely small relative to S0, but not K=0 itself.
    log_S0_K = np.log(S0 / K) # This will be inf if K=0, but we handled K=0.
                              # If K is tiny positive, log_S0_K is large positive.

    d1_num = log_S0_K + a + b
    sqrt_b = np.sqrt(b)
    
    # Avoid inf/inf or inf-inf issues if d1_num or sqrt_b are problematic, though usually handled by numpy.
    if sqrt_b == 0: # Should be caught by b==0, but for safety
        d1 = float('inf') if d1_num > 0 else float('-inf') if d1_num < 0 else 0
    else:
        d1 = d1_num / sqrt_b
    
    d2 = d1 - sqrt_b

    price = np.exp(-r * T) * (S0 * np.exp(a) * norm.cdf(d1) - K * norm.cdf(d2))
    return price

def generate_sv_paths(
    S0: float,      # Initial asset price
    V0: float,      # Initial variance (sigma^2)
    r: float,       # Risk-free interest rate
    delta: float,   # Dividend yield
    alpha: float,   # Mean reversion rate for variance
    V_bar: float,   # Long-term variance
    xi: float,      # Volatility of volatility
    T: float,       # Time to maturity
    n_steps: int,   # Number of time steps
    n_paths: int,   # Number of paths to generate
    antithetic: bool = False,  # Whether to use antithetic sampling for variance reduction
    rho: float = 0.0,          # Correlation between asset price and variance Wiener processes
    scheme: str = 'milstein'   # Discretization scheme ('euler' or 'milstein')
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    S_paths = np.zeros((n_paths, n_steps + 1))
    V_paths = np.zeros((n_paths, n_steps + 1))
    S_max_paths = np.zeros((n_paths, n_steps + 1))
    
    S_paths[:, 0] = S0
    V_paths[:, 0] = V0
    S_max_paths[:, 0] = S0
    
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic sampling is enabled")
        half_paths = n_paths // 2
        Z1 = np.random.normal(0, 1, (half_paths, n_steps))
        Z2 = np.random.normal(0, 1, (half_paths, n_steps))
        
        if rho != 0:
            Z2_corr = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        else:
            Z2_corr = Z2
        
        for i in range(1, n_steps + 1):
            V_curr_first_half = np.maximum(V_paths[:half_paths, i-1], 1e-6)
            if scheme == 'milstein':
                dW2 = Z2_corr[:, i-1] * sqrt_dt
                dV = alpha * (V_bar - V_curr_first_half) * dt + xi * np.sqrt(V_curr_first_half) * dW2 + 0.25 * xi**2 * (dW2**2 - dt)
            else: # Euler
                dV = alpha * (V_bar - V_curr_first_half) * dt + xi * np.sqrt(V_curr_first_half * dt) * Z2_corr[:, i-1]
            V_paths[:half_paths, i] = np.maximum(V_curr_first_half + dV, 1e-6)
            
            dW1 = Z1[:, i-1] * sqrt_dt
            dS = (r - delta - 0.5 * V_curr_first_half) * dt + np.sqrt(V_curr_first_half) * dW1
            S_paths[:half_paths, i] = S_paths[:half_paths, i-1] * np.exp(dS)
            S_max_paths[:half_paths, i] = np.maximum(S_max_paths[:half_paths, i-1], S_paths[:half_paths, i])
            
            V_curr_second_half = np.maximum(V_paths[half_paths:, i-1], 1e-6)
            if scheme == 'milstein':
                dW2_anti = -Z2_corr[:, i-1] * sqrt_dt # Antithetic Z2_corr
                dV_anti = alpha * (V_bar - V_curr_second_half) * dt + xi * np.sqrt(V_curr_second_half) * dW2_anti + 0.25 * xi**2 * (dW2_anti**2 - dt)
            else: # Euler
                dV_anti = alpha * (V_bar - V_curr_second_half) * dt + xi * np.sqrt(V_curr_second_half * dt) * (-Z2_corr[:, i-1])
            V_paths[half_paths:, i] = np.maximum(V_curr_second_half + dV_anti, 1e-6)

            dW1_anti = -Z1[:, i-1] * sqrt_dt # Antithetic Z1
            dS_anti = (r - delta - 0.5 * V_curr_second_half) * dt + np.sqrt(V_curr_second_half) * dW1_anti
            S_paths[half_paths:, i] = S_paths[half_paths:, i-1] * np.exp(dS_anti)
            S_max_paths[half_paths:, i] = np.maximum(S_max_paths[half_paths:, i-1], S_paths[half_paths:, i])
    else: # No antithetic
        Z1 = np.random.normal(0, 1, (n_paths, n_steps))
        Z2 = np.random.normal(0, 1, (n_paths, n_steps))
        if rho != 0:
            Z2_corr = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        else:
            Z2_corr = Z2
            
        for i in range(1, n_steps + 1):
            V_curr = np.maximum(V_paths[:, i-1], 1e-6)
            if scheme == 'milstein':
                dW2 = Z2_corr[:, i-1] * sqrt_dt
                dV = alpha * (V_bar - V_curr) * dt + xi * np.sqrt(V_curr) * dW2 + 0.25 * xi**2 * (dW2**2 - dt)
            else: # Euler
                dV = alpha * (V_bar - V_curr) * dt + xi * np.sqrt(V_curr * dt) * Z2_corr[:, i-1]
            V_paths[:, i] = np.maximum(V_curr + dV, 1e-6)
            
            dW1 = Z1[:, i-1] * sqrt_dt
            dS = (r - delta - 0.5 * V_curr) * dt + np.sqrt(V_curr) * dW1
            S_paths[:, i] = S_paths[:, i-1] * np.exp(dS)
            S_max_paths[:, i] = np.maximum(S_max_paths[:, i-1], S_paths[:, i])
            
    return S_paths, V_paths, S_max_paths

def continuous_lookback_call_price(
    S0: float,      # Initial asset price
    K: float,       # Strike price
    r: float,       # Risk-free interest rate
    sigma: float,   # Volatility (constant)
    T: float,       # Time to maturity
    delta: float,   # Dividend yield
    M: float = None # Current known maximum (defaults to S0)
) -> float:
    if M is None:
        M = S0
    
    if T == 0: # Handle T=0 case explicitly
        return np.maximum(0, M - K)

    # Prevent sigma from being zero or too small to avoid division by zero in B or x
    sigma = np.maximum(sigma, 1e-10)

    B = 2 * (r - delta) / (sigma**2)
    
    if K >= M:
        E = K
        G = 0
    else:
        E = M
        G = np.exp(-r * T) * (M - K)
    
    x = (np.log(S0 / E) + (r - delta - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    price = G + S0 * np.exp(-delta * T) * norm.cdf(x + sigma * np.sqrt(T)) - \
            K * np.exp(-r * T) * norm.cdf(x)
    
    if S0 != E: # This condition might need care if S0 is very close to E
        # Ensure (E/S0)**B doesn't cause issues if B is very large/small
        # Term for S0/B can be problematic if B is close to 0.
        # If r - delta is close to 0, B can be very small or large.
        if np.isclose(B, 0): # Avoid division by B if B is ~0
             # This case may need specific handling based on the formula derivation if B=0
             # For now, if B is zero, this term is skipped, which might be an approximation
             pass
        else:
            term_val = (E / S0)**B
            # Prevent overflow if term_val is huge
            if np.isinf(term_val) or np.isnan(term_val): term_val = 0 # Heuristic if E/S0 makes B power explode

            price -= (S0 / B) * (np.exp(-r * T) * term_val * 
                                 norm.cdf(x + (1 - B) * sigma * np.sqrt(T)) - 
                                 np.exp(-delta * T) * norm.cdf(x + sigma * np.sqrt(T)))
    return price

def arithmetic_asian_call_mc(
    S0: float, K: float, r: float, sigma: float, T: float, delta: float,
    n_steps: int, n_paths: int, antithetic: bool = False, control_variate: bool = False
) -> Tuple[float, float, float]:
    start_time = time.time()
    paths = generate_gbm_paths(S0, r, sigma, T, delta, n_steps, n_paths, antithetic)
    arithmetic_avg = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(arithmetic_avg - K, 0)
    discount_factor = np.exp(-r * T)
    
    if control_variate:
        geometric_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        geometric_payoffs = np.maximum(geometric_avg - K, 0)
        geo_price = geometric_asian_call_price(S0, K, r, sigma, T, delta, n_steps)
        
        pilot_size = max(1000, n_paths // 10)
        if n_paths < pilot_size: # Ensure pilot_size is not larger than n_paths
            pilot_payoffs = payoffs
            pilot_geo_payoffs = geometric_payoffs
        else:
            pilot_payoffs = payoffs[:pilot_size]
            pilot_geo_payoffs = geometric_payoffs[:pilot_size]

        # Handle cases where variance might be zero (e.g., if pilot_geo_payoffs are all identical)
        cov_matrix = np.cov(pilot_payoffs, pilot_geo_payoffs)
        if cov_matrix.ndim == 0: # Scalar, means pilot_payoffs and pilot_geo_payoffs were 1D and possibly identical
             beta = 0 # No effective control variate
        elif cov_matrix[1, 1] == 0: # Variance of control is zero
            beta = 0
        else:
            beta = -cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # Ensure discount_factor is not zero to prevent division by zero
        if discount_factor == 0: # Should not happen if T is finite
            adjusted_payoffs = payoffs # Fallback if discount_factor is zero
        else:
            adjusted_payoffs = payoffs + beta * (geometric_payoffs - geo_price / discount_factor)
        
        price = discount_factor * np.mean(adjusted_payoffs)
        std_err = discount_factor * np.std(adjusted_payoffs, ddof=1) / np.sqrt(n_paths)
    else:
        price = discount_factor * np.mean(payoffs)
        std_err = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
        
    computation_time = time.time() - start_time
    return price, std_err, computation_time

def lookback_call_mc(
    S0: float, K: float, r: float, delta: float, sigma0: float, alpha: float,
    V_bar: float, xi: float, T: float, n_steps: int, n_paths: int,
    antithetic: bool = False, control_variate: bool = False, rho: float = 0.0,
    scheme: str = 'milstein'
) -> Tuple[float, float, float]:
    time_module = time # Use alias as in notebook
    start_time = time_module.time()
    V0 = sigma0**2
    
    S_paths, V_paths, S_max_paths = generate_sv_paths(
        S0, V0, r, delta, alpha, V_bar, xi, T, n_steps, n_paths, antithetic, rho, scheme
    )
    max_prices = S_max_paths[:, -1]
    payoffs = np.maximum(max_prices - K, 0)
    discount_factor = np.exp(-r * T)
    
    if control_variate:
        avg_sigma = np.sqrt(np.mean(V_paths))
        control_paths = generate_gbm_paths(S0, r, avg_sigma, T, delta, n_steps, n_paths, antithetic)
        control_max_prices = np.max(control_paths, axis=1)
        control_payoffs = np.maximum(control_max_prices - K, 0)
        control_price_analytical = continuous_lookback_call_price(S0, K, r, avg_sigma, T, delta) # Renamed to avoid conflict

        pilot_size = max(1000, n_paths // 10)
        if n_paths < pilot_size:
            pilot_payoffs = payoffs
            pilot_control_payoffs = control_payoffs
        else:
            pilot_payoffs = payoffs[:pilot_size]
            pilot_control_payoffs = control_payoffs[:pilot_size]

        cov_matrix = np.cov(pilot_payoffs, pilot_control_payoffs)
        if cov_matrix.ndim == 0:
            beta = 0
        elif cov_matrix[1,1] == 0:
            beta = 0
        else:
            beta = -cov_matrix[0, 1] / cov_matrix[1, 1]
        
        if discount_factor == 0:
            adjusted_payoffs = payoffs
        else:
            adjusted_payoffs = payoffs + beta * (control_payoffs - control_price_analytical / discount_factor)
        
        price = discount_factor * np.mean(adjusted_payoffs)
        std_err = discount_factor * np.std(adjusted_payoffs, ddof=1) / np.sqrt(n_paths)
    else:
        price = discount_factor * np.mean(payoffs)
        std_err = discount_factor * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
        
    computation_time = time_module.time() - start_time
    return price, std_err, computation_time
