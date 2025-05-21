import numpy as np

def generate_gbm_paths(
    S0: float,      # Initial asset price
    r: float,       # Risk-free interest rate
    sigma: float,   # Volatility (constant)
    T: float,       # Time to maturity
    delta: float,   # Dividend yield
    n_steps: int,   # Number of time steps
    n_paths: int,   # Number of paths to generate
    antithetic: bool = False  # Whether to use antithetic sampling for variance reduction
) -> np.ndarray:

    # Calculate time step size for discretization
    dt = T / n_steps  # Uniform time step
    
    # Pre-compute the drift term for the log-normal process
    # The term (r - delta - 0.5 * sigma^2) is the risk-neutral drift adjusted for Itô's correction
    nudt = (r - delta - 0.5 * sigma**2) * dt  # Drift term × dt
    
    # Pre-compute the volatility scaling factor
    sigsdt = sigma * np.sqrt(dt)  # Volatility × sqrt(dt) for Brownian motion scaling
    
    # Initialize array to store all price paths
    # Shape: (n_paths, n_steps+1) to include initial values at t=0
    paths = np.zeros((n_paths, n_steps + 1))
    
    # Set initial price for all paths at t=0
    paths[:, 0] = S0  # All paths start at S0
    
    # Branch based on whether antithetic sampling is used for variance reduction
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic sampling is enabled")
        # With antithetic sampling, we generate half the paths and use negated random numbers for the other half
        # This creates negatively correlated paths that reduce the overall variance of the estimator
        half_paths = n_paths // 2  # Integer division to get half the number of paths
        
        # Generate all random numbers at once for the first half of paths (vectorized approach)
        # These are standard normal random variables for the Brownian motion increments
        Z = np.random.normal(0, 1, (half_paths, n_steps))  # Shape: (half_paths, n_steps)
        
        # Calculate price increments for the first half of paths using vectorized operations
        # The log-normal price evolution follows: S(t+dt) = S(t) * exp((r-delta-0.5σ²)dt + σ√dt*Z)
        increments = np.exp(nudt + sigsdt * Z)  # Shape: (half_paths, n_steps)
        
        # Calculate all future prices by multiplying the initial price by the cumulative product of increments
        # This is more numerically stable than adding log-returns and then exponentiating
        paths[:half_paths, 1:] = S0 * np.cumprod(increments, axis=1)  # Cumulative product along time axis
        
        # For the second half of paths, use antithetic sampling (negated random numbers)
        # This creates paths that are negatively correlated with the first half
        anti_increments = np.exp(nudt + sigsdt * (-Z))  # Negate the random numbers
        
        # Calculate all future prices for the antithetic paths
        paths[half_paths:, 1:] = S0 * np.cumprod(anti_increments, axis=1)
    else:
        # Standard Monte Carlo without variance reduction
        # Generate all random numbers at once for all paths (vectorized approach)
        Z = np.random.normal(0, 1, (n_paths, n_steps))  # Shape: (n_paths, n_steps)
        
        # Calculate price increments for all paths using vectorized operations
        # Each increment represents the multiplicative factor for the price change in one time step
        increments = np.exp(nudt + sigsdt * Z)  # Shape: (n_paths, n_steps)
        
        # Calculate all future prices by multiplying the initial price by the cumulative product of increments
        # This efficiently generates the entire price paths in a single vectorized operation
        paths[:, 1:] = S0 * np.cumprod(increments, axis=1)  # Shape: (n_paths, n_steps)
    
    # Return the generated paths (immutable array following functional programming principles)
    # Shape: (n_paths, n_steps+1) where paths[:,0] = S0 and paths[:,1:] are the simulated prices
    return paths
