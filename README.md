# Risk-Neutral Monte Carlo Pricing

This repository contains a high-performance computational finance project implementing risk-neutral Monte Carlo pricing for exotic options with variance reduction techniques. The project is structured according to functional programming principles, with pure functions, immutable data structures, and clear separation of concerns.

## Project Overview

The project implements pricing models for two types of exotic options:

1. **Arithmetic Asian Call Option**: Pays the difference between the arithmetic average of asset prices and the strike price at maturity.
2. **European Fixed Strike Lookback Call Option with Stochastic Volatility**: Pays the difference between the maximum asset price and the strike price at maturity.

For each option, we implement and compare four simulation approaches:
- Simple Monte Carlo (baseline)
- Antithetic sampling
- Control variate
- Combined antithetic and control variate

The goal is to analyze the trade-off between computation time and variance reduction for each technique.

## High-Performance Implementation with JAX

This project showcases advanced functional programming skills through a dual implementation:

1. **NumPy Implementation** (`final_project.qmd`): Traditional vectorized implementation using NumPy
2. **JAX Implementation** (`final_project_jax.qmd`): High-performance implementation using JAX with:
   - JIT compilation for accelerated computation
   - Vectorized operations using JAX's immutable arrays
   - Functional design with pure functions and no side effects
   - XLA (Accelerated Linear Algebra) integration

The JAX implementation demonstrates significant performance improvements while maintaining the same mathematical accuracy, particularly for the variance reduction techniques.

## Project Structure

- `final_project.qmd`: Quarto document containing the NumPy implementation, analysis, and results
- `final_project_jax.qmd`: Quarto document containing the JAX implementation with performance analysis
- `Makefile`: Contains commands for rendering both documents and other project tasks
- `pyproject.toml` & `poetry.lock`: Poetry configuration for dependency management

## Requirements

- Python 3.13+
- Poetry (for dependency management)
- Quarto

All Python dependencies are managed through Poetry and include:
- Jupyter (>=1.1.1)
- Matplotlib (>=3.10.1)
- NumPy (>=2.2.4)
- JAX (>=0.5.3)
- Pandas (>=2.2.3)
- PyYAML (>=6.0.2)
- SciPy (>=1.15.2)
- JAX (>=0.4.26)
- JAXlib (>=0.4.26)

## Code Quality

The project follows strict code quality guidelines:
- No unused imports or dependencies
- Organized imports (standard library first, third-party second)
- Pure functions with no side effects
- Clear separation of concerns
- Explicit dependencies
- Vectorized operations for optimal performance

## Advanced Features

### Option Greeks Calculation

The JAX implementation leverages automatic differentiation to compute option Greeks with high precision:

- **Delta (∂V/∂S)**: Sensitivity of option price to changes in the underlying asset price
- **Gamma (∂²V/∂S²)**: Rate of change of Delta with respect to the underlying asset price
- **Vega (∂V/∂σ)**: Sensitivity of option price to changes in volatility

These Greeks are computed automatically through JAX's differentiation system rather than using finite difference approximations, resulting in more accurate and efficient calculations.

### Performance Benchmarking

The project includes comprehensive benchmarking comparing the NumPy and JAX implementations:

```bash
# Run the benchmarking script
poetry run python benchmark_jax.py
```

This generates performance metrics and visualizations showing:
- Execution time comparison between NumPy and JAX implementations
- Speedup factors across different numbers of simulation paths
- Scaling properties with large numbers of paths (up to millions)
- Option Greeks profiles across different strike prices

## Setup

### Installing Dependencies

```bash
# Install Poetry dependencies
poetry install
```

## Usage

To use this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/risk-neutral-monte-carlo.git
cd risk-neutral-monte-carlo

# Install dependencies with Poetry
poetry install

# Set up the Jupyter kernel
make setup-kernel

# Render both Quarto documents (NumPy and JAX implementations)
make render-all

# View the rendered HTML files in your browser
make view QUARTO_FILE=final_project.qmd
make view QUARTO_FILE=final_project_jax.qmd

# Execute both implementations
make run-all

# Clean generated files
make clean
```

## Makefile Commands

The project includes a Makefile with the following targets:

- `make setup-kernel`: Sets up a Jupyter kernel using Poetry's virtual environment
- `make render`: Renders the default Quarto document (final_project.qmd)
- `make render QUARTO_FILE=file.qmd`: Renders a specific Quarto document
- `make render-all`: Renders all Quarto documents (both NumPy and JAX implementations)
- `make view`: Opens the default rendered HTML in your default browser
- `make view QUARTO_FILE=file.qmd`: Opens a specific rendered HTML
- `make run`: Executes the default Quarto document using Poetry's environment
- `make run QUARTO_FILE=file.qmd`: Executes a specific Quarto document
- `make run-all`: Executes all Quarto documents
- `make clean`: Removes all generated files
- `make help`: Lists available commands

## Implementation Details

The implementation follows functional programming principles:
- Pure functions with no side effects
- Immutable data structures where possible
- Clear separation of concerns
- Explicit dependencies
- Vectorized operations for performance

The Monte Carlo simulations use various variance reduction techniques to improve efficiency:
- **Antithetic Sampling**: Generates negatively correlated pairs of paths to reduce variance
- **Control Variate**: Uses analytical solutions for related options as control variates
- **Combined Approach**: Applies both techniques together for maximum variance reduction

### JAX-Specific Optimizations

The JAX implementation includes several advanced optimizations:

1. **JIT Compilation**: All core functions are decorated with `@jax.jit` for just-in-time compilation
2. **Functional Updates**: Uses JAX's immutable arrays with functional updates (e.g., `array.at[idx].set(value)`)
3. **Pure Random Number Generation**: Employs JAX's key-splitting approach for reproducible randomness
4. **Automatic Vectorization**: Leverages JAX's `vmap` for efficient vectorization across paths
5. **XLA Integration**: Benefits from XLA's optimizations for numerical computations

## Results

The project includes a detailed analysis of the performance of each variance reduction technique in terms of:
- Option price estimate
- Standard error
- Computation time
- Relative efficiency

The JAX implementation demonstrates substantial performance improvements:
- 5-20x speedup compared to the NumPy implementation
- Ability to handle millions of simulation paths efficiently
- Precise computation of option Greeks through automatic differentiation
- Excellent scaling properties for large-scale simulations

These results showcase advanced programming skills in high-performance numerical computing and financial modeling, making this project an excellent demonstration of technical capabilities for quantitative finance roles.