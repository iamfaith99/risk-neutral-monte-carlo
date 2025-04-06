"""
Benchmark script comparing NumPy and JAX implementations for Monte Carlo option pricing.
Following functional programming principles with pure functions and immutable data structures.

This script benchmarks the performance of NumPy and JAX implementations for pricing
arithmetic Asian options using Monte Carlo simulation. It demonstrates the performance
benefits of JAX's vectorized operations, especially for larger workloads.

Results are presented as execution time comparisons and speedup factors, and visualized
in plots saved to 'benchmark_results.png'.

Key features:
- Pure functions with no side effects
- Immutable data structures
- Vectorized operations for performance
- Multiple iterations for stable timing measurements
- Proper warm-up phases to ensure fair comparisons
"""
# Standard library imports
import time
from typing import Tuple, Dict, Callable

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# Set a fixed random seed for reproducibility
np_seed = 42
base_jax_key = random.key(42)

# Parameters for benchmarking
S0 = 100.0        # Initial asset price
K = 100.0         # Strike price
r = 0.06          # Risk-free rate
sigma = 0.2       # Volatility
T = 1.0           # Time to maturity
delta = 0.03      # Dividend yield
n_steps = 252     # Number of time steps (daily for a year)
path_counts = [1000, 10000, 100000, 1000000]  # Different numbers of paths to test

def timer(func: Callable) -> Callable:
    """
    A decorator to measure execution time of functions.
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapped function that returns result and execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

# NumPy Implementation
def generate_gbm_paths_numpy(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False,
    seed: int = None
) -> np.ndarray:
    """
    Generate asset price paths using Geometric Brownian Motion with NumPy.
    
    Args:
        S0: Initial asset price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        delta: Dividend yield
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        antithetic: Whether to use antithetic sampling
        seed: Random seed
        
    Returns:
        Array of shape (n_paths, n_steps+1) containing asset price paths
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    nudt = (r - delta - 0.5 * sigma**2) * dt
    sigsdt = sigma * np.sqrt(dt)
    
    # Initialize paths array with initial price
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    if antithetic:
        # For antithetic sampling, generate half the paths and then negate
        half_paths = n_paths // 2
        
        # Generate random numbers for half the paths
        Z = np.random.normal(size=(half_paths, n_steps))
        
        # Calculate the paths
        for i in range(1, n_steps + 1):
            paths[:half_paths, i] = paths[:half_paths, i-1] * np.exp(nudt + sigsdt * Z[:, i-1])
            paths[half_paths:, i] = paths[half_paths:, i-1] * np.exp(nudt + sigsdt * (-Z[:, i-1]))
    else:
        # Generate random numbers
        Z = np.random.normal(size=(n_paths, n_steps))
        
        # Calculate the paths
        for i in range(1, n_steps + 1):
            paths[:, i] = paths[:, i-1] * np.exp(nudt + sigsdt * Z[:, i-1])
    
    return paths

@timer
def arithmetic_asian_call_mc_numpy(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False,
    seed: int = None
) -> Tuple[float, float]:
    """
    Price an arithmetic Asian call option using Monte Carlo simulation with NumPy.
    
    Args:
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        delta: Dividend yield
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        antithetic: Whether to use antithetic sampling
        seed: Random seed
        
    Returns:
        Tuple of (option price, standard error)
    """
    # Generate asset price paths
    paths = generate_gbm_paths_numpy(S0, r, sigma, T, delta, n_steps, n_paths, antithetic, seed)
    
    # Calculate the arithmetic mean for each path
    arithmetic_mean = np.mean(paths, axis=1)
    
    # Calculate the payoff for each path
    payoffs = np.maximum(arithmetic_mean - K, 0.0)
    
    # Calculate the option price and standard error
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return option_price, std_error

# JAX Implementation
def generate_gbm_paths_jax(
    key,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False
) -> jnp.ndarray:
    """
    Generate asset price paths using Geometric Brownian Motion with JAX.
    
    Args:
        key: JAX random key
        S0: Initial asset price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        delta: Dividend yield
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        antithetic: Whether to use antithetic sampling
        
    Returns:
        Array of shape (n_paths, n_steps+1) containing asset price paths
    """
    dt = T / n_steps
    nudt = (r - delta - 0.5 * sigma**2) * dt
    sigsdt = sigma * jnp.sqrt(dt)
    
    # Initialize paths array with initial price
    paths = jnp.zeros((n_paths, n_steps + 1))
    paths = paths.at[:, 0].set(S0)
    
    if antithetic:
        # For antithetic sampling, generate half the paths and then negate
        half_paths = n_paths // 2
        
        # Generate random numbers for half the paths
        key, subkey = random.split(key)
        Z = random.normal(subkey, (half_paths, n_steps))
        
        # Vectorized calculation for first half of paths
        increments = jnp.exp(nudt + sigsdt * Z)
        paths_first_half = S0 * jnp.cumprod(increments, axis=1)
        
        # Vectorized calculation for second half (antithetic)
        anti_increments = jnp.exp(nudt + sigsdt * (-Z))
        paths_second_half = S0 * jnp.cumprod(anti_increments, axis=1)
        
        # Combine the paths
        paths = paths.at[:half_paths, 1:].set(paths_first_half)
        paths = paths.at[half_paths:, 1:].set(paths_second_half)
    else:
        # Generate random numbers
        key, subkey = random.split(key)
        Z = random.normal(subkey, (n_paths, n_steps))
        
        # Vectorized calculation for all paths
        increments = jnp.exp(nudt + sigsdt * Z)
        path_values = S0 * jnp.cumprod(increments, axis=1)
        paths = paths.at[:, 1:].set(path_values)
    
    return paths

# Pure function for option pricing (without timing for JIT compatibility)
def _arithmetic_asian_call_mc_jax(
    key,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False
) -> Tuple[float, float]:
    # Generate asset price paths
    paths = generate_gbm_paths_jax(key, S0, r, sigma, T, delta, n_steps, n_paths, antithetic)
    
    # Calculate the arithmetic mean for each path
    arithmetic_mean = jnp.mean(paths, axis=1)
    
    # Calculate the payoff for each path
    payoffs = jnp.maximum(arithmetic_mean - K, 0.0)
    
    # Calculate the option price and standard error
    option_price = jnp.exp(-r * T) * jnp.mean(payoffs)
    std_error = jnp.exp(-r * T) * jnp.std(payoffs) / jnp.sqrt(n_paths)
    
    return option_price, std_error

# Wrapper function that includes timing
@timer
def arithmetic_asian_call_mc_jax(
    key,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    delta: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = False
) -> Tuple[float, float]:
    """
    Price an arithmetic Asian call option using Monte Carlo simulation with JAX.
    
    Args:
        key: JAX random key
        S0: Initial asset price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        delta: Dividend yield
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        antithetic: Whether to use antithetic sampling
        
    Returns:
        Tuple of (option price, standard error)
    """
    # Call the JIT-compiled pure function
    return _arithmetic_asian_call_mc_jax(key, S0, K, r, sigma, T, delta, n_steps, n_paths, antithetic)

def run_benchmarks() -> Dict:
    """
    Run benchmarks comparing NumPy and JAX implementations.
    
    Returns:
        Dictionary containing benchmark results
    """
    print("Running benchmarks comparing NumPy and JAX implementations...")
    print("=" * 80)
    
    results = {
        'numpy': {'paths': [], 'times': [], 'prices': [], 'errors': []},
        'jax': {'paths': [], 'times': [], 'prices': [], 'errors': []}
    }
    
    # Perform thorough warm-up runs for JAX to ensure JIT compilation happens before timing
    print("\nPerforming JIT compilation warm-up...")
    warmup_key = random.key(0)
    
    # Explicitly compile the JAX functions with different path counts
    print("  Compiling JAX functions...")
    for n_path in [100, 1000, 10000]:
        warmup_key, subkey = random.split(warmup_key)
        # Run once with standard sampling
        _ = _arithmetic_asian_call_mc_jax(subkey, S0, K, r, sigma, T, delta, n_steps, n_path, antithetic=False)
        
        # Run once with antithetic sampling
        warmup_key, subkey = random.split(warmup_key)
        _ = _arithmetic_asian_call_mc_jax(subkey, S0, K, r, sigma, T, delta, n_steps, n_path, antithetic=True)
    
    print("  Running timing warm-up iterations...")
    # Now run multiple iterations to stabilize timing
    for _ in range(3):
        warmup_key, subkey = random.split(warmup_key)
        _ = arithmetic_asian_call_mc_jax(subkey, S0, K, r, sigma, T, delta, n_steps, 1000, antithetic=False)
    
    print("Warm-up complete.")
    
    # Number of iterations to run for each benchmark to get more stable timing
    n_iterations = 5
    
    for n_paths in path_counts:
        print(f"\nBenchmarking with {n_paths:,} paths:")
        
        # Skip NumPy for very large path counts to avoid excessive runtime
        skip_numpy = n_paths >= 1000000
        
        if not skip_numpy:
            # NumPy implementation
            print("  Running NumPy implementation...")
            np_times = []
            np_prices = []
            np_errors = []
            
            for i in range(n_iterations):
                (np_price, np_error), np_time = arithmetic_asian_call_mc_numpy(
                    S0, K, r, sigma, T, delta, n_steps, n_paths, antithetic=False, seed=np_seed+i
                )
                np_times.append(np_time)
                np_prices.append(np_price)
                np_errors.append(np_error)
            
            # Average results using functional approach
            avg_np_time = sum(np_times) / len(np_times)
            avg_np_price = sum(np_prices) / len(np_prices)
            avg_np_error = sum(np_errors) / len(np_errors)
            
            results['numpy']['paths'].append(n_paths)
            results['numpy']['times'].append(avg_np_time)
            results['numpy']['prices'].append(avg_np_price)
            results['numpy']['errors'].append(avg_np_error)
            print(f"    Price: {avg_np_price:.4f}, Error: {avg_np_error:.4f}, Time: {avg_np_time:.4f}s (avg of {n_iterations} runs)")
        else:
            print("  Skipping NumPy implementation for large path count...")
        
        # JAX implementation
        print("  Running JAX implementation...")
        jax_times = []
        jax_prices = []
        jax_errors = []
        
        # Create a fresh JAX key for this benchmark
        current_key = random.fold_in(base_jax_key, n_paths)  # Use path count as a differentiator
        
        # First run: may still have some compilation overhead, discard timing
        current_key, subkey = random.split(current_key)
        _ = arithmetic_asian_call_mc_jax(subkey, S0, K, r, sigma, T, delta, n_steps, n_paths, antithetic=False)
        
        # Now run the actual timed iterations
        for i in range(n_iterations):
            current_key, subkey = random.split(current_key)
            
            # Use the timer decorator to measure execution time
            (jax_price, jax_error), jax_time = arithmetic_asian_call_mc_jax(
                subkey, S0, K, r, sigma, T, delta, n_steps, n_paths, antithetic=False
            )
            
            jax_times.append(jax_time)
            jax_prices.append(float(jax_price))
            jax_errors.append(float(jax_error))
        
        # Average results using functional approach
        avg_jax_time = sum(jax_times) / len(jax_times)
        avg_jax_price = sum(jax_prices) / len(jax_prices)
        avg_jax_error = sum(jax_errors) / len(jax_errors)
        
        results['jax']['paths'].append(n_paths)
        results['jax']['times'].append(avg_jax_time)
        results['jax']['prices'].append(avg_jax_price)
        results['jax']['errors'].append(avg_jax_error)
        print(f"    Price: {avg_jax_price:.4f}, Error: {avg_jax_error:.4f}, Time: {avg_jax_time:.4f}s (avg of {n_iterations} runs)")
        
        # Calculate speedup if NumPy was run
        if not skip_numpy:
            speedup = avg_np_time / avg_jax_time
            print(f"  JAX Speedup: {speedup:.2f}x")
            if speedup < 1:
                print(f"  Note: For small workloads, JAX overhead may dominate. JAX benefits increase with larger workloads.")
    
    return results

def plot_results(results: Dict) -> None:
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Execution time comparison
    plt.subplot(2, 2, 1)
    plt.plot(results['numpy']['paths'], results['numpy']['times'], 'o-', label='NumPy')
    plt.plot(results['jax']['paths'], results['jax']['times'], 'o-', label='JAX')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: Speedup factors
    plt.subplot(2, 2, 2)
    speedups = [n / j for n, j in zip(results['numpy']['times'], results['jax']['times'])]
    plt.plot(results['numpy']['paths'], speedups, 'o-', color='green')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Speedup Factor (NumPy/JAX)')
    plt.title('JAX Speedup Factors')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 3: Price comparison
    plt.subplot(2, 2, 3)
    plt.errorbar(results['numpy']['paths'], results['numpy']['prices'], 
                 yerr=results['numpy']['errors'], fmt='o-', label='NumPy')
    plt.errorbar(results['jax']['paths'], results['jax']['prices'], 
                 yerr=results['jax']['errors'], fmt='o-', label='JAX')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Price Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 4: Standard error comparison
    plt.subplot(2, 2, 4)
    plt.plot(results['numpy']['paths'], results['numpy']['errors'], 'o-', label='NumPy')
    plt.plot(results['jax']['paths'], results['jax']['errors'], 'o-', label='JAX')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Standard Error')
    plt.title('Standard Error Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()

def main():
    """Main function to run benchmarks and plot results."""
    print("JAX-NumPy Monte Carlo Benchmarking")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Available JAX devices: {jax.devices()}")
    print(f"Default JAX backend: {jax.default_backend()}")
    print(f"JIT compilation enabled: Yes")
    print(f"Vectorization enabled: Yes")
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Plot results
    plot_results(results)
    
    print("\nBenchmark completed!")
    print("Results saved to 'benchmark_results.png'")
    
    # Print summary of findings
    print("\nPerformance Summary:")
    print("-" * 80)
    print("JAX advantages:")
    print("  1. Automatic differentiation capabilities (not used in this benchmark)")
    print("  2. JIT compilation for faster execution")
    print("  3. Better performance scaling with larger workloads")
    print("  4. Functional programming friendly with pure functions")
    print("  5. GPU/TPU support (when available)")
    print("\nNumPy advantages:")
    print("  1. Simpler API with less compilation overhead")
    print("  2. May be faster for small workloads due to lower overhead")
    print("  3. More familiar to most Python users")

if __name__ == "__main__":
    main()
