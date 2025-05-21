# Risk-Neutral Monte Carlo Pricing

This repository contains a computational finance project implementing risk-neutral Monte Carlo pricing for exotic options with variance reduction techniques. The project is structured according to functional programming principles, with pure functions, immutable data structures, and clear separation of concerns.

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

## Implementation Features

This project showcases advanced functional programming skills through:

- Vectorized operations using NumPy for high performance
- Functional design with pure functions and no side effects
- Immutable data structures where possible
- Clear separation of concerns
- Comprehensive documentation with detailed explanations

## Project Structure

- `final_project.qmd`: Quarto document containing the implementation, analysis, and results
- `gbm.py`: Module for generating Geometric Brownian Motion paths.
- `Makefile`: Contains commands for rendering the document and other project tasks
- `pyproject.toml` & `poetry.lock`: Poetry configuration for dependency management
- `.gitignore`: Configuration for excluding generated files from version control
- `tests/`: Directory containing test files.
    - `tests/test_gbm.py`: Contains unit tests for the `gbm.py` module.
    - `tests/notebook_helpers.py`: Contains helper functions extracted from `final_project.qmd` for testability.
    - `tests/test_notebook_functions.py`: Contains tests for core financial functions defined in `final_project.qmd` and extracted into `notebook_helpers.py`.

## Requirements

- Python 3.10+
- Poetry (for dependency management)
- Quarto

All Python dependencies are managed through Poetry and include:
- Jupyter (>=1.1.1)
- Matplotlib (>=3.10.1)
- NumPy (>=2.2.4)
- Pandas (>=2.2.3)
- PyYAML (>=6.0.2)
- SciPy (>=1.15.2)

## Advanced Features

### Variance Reduction Techniques

The project implements several variance reduction techniques to improve the efficiency of Monte Carlo simulations:

- **Antithetic Sampling**: Generates negatively correlated pairs of paths to reduce variance. This is particularly effective for monotonic payoff functions like those in option pricing.

- **Control Variate**: Uses analytical solutions for related options (geometric Asian option and continuous lookback option) as control variates. This significantly reduces variance when the control variate is highly correlated with the target.

- **Combined Approach**: Applies both techniques together for maximum variance reduction, demonstrating how complementary techniques can be combined for optimal results.

### Stochastic Volatility Model

The project implements a sophisticated stochastic volatility model for the lookback option, featuring:

- Heston-style mean-reverting variance process
- Milstein discretization scheme for improved accuracy
- Correlation between asset price and variance processes (leverage effect)
- Efficient tracking of running maximum prices for lookback options

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

# Render the Quarto document
make render

# View the rendered HTML file in your browser
make view

# Execute the implementation
make run

# Clean generated files
make clean
```

## Makefile Commands

The project includes a Makefile with the following targets:

- `make setup-kernel`: Sets up a Jupyter kernel using Poetry's virtual environment
- `make render`: Renders the Quarto document
- `make view`: Opens the rendered HTML in your default browser
- `make run`: Executes the Quarto document using Poetry's environment
- `make test`: Runs the pytest test suite (New).
- `make clean`: Removes all generated files
- `make help`: Lists available commands

## Testing

The project now includes expanded test coverage for both the GBM path generation (`gbm.py`) and the core financial functions from the notebook, utilizing pytest. Tests are located in the `tests/` directory and can be run using `poetry run pytest` or `make test`.

A correction was made to the control variate calculation within the Monte Carlo pricing functions to ensure more accurate variance reduction.

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

### NumPy Optimizations

The implementation includes several optimizations for performance:

1. **Vectorized Operations**: Uses NumPy's vectorized operations to avoid explicit loops
2. **Pre-computed Constants**: Calculates constants outside of loops for efficiency
3. **Efficient Random Number Generation**: Generates all random numbers at once
4. **Numerical Stability**: Uses logarithmic transformations for better numerical stability
5. **Pilot Sampling**: Uses a subset of paths to estimate optimal control variate parameters

## Results

The project includes a detailed analysis of the performance of each variance reduction technique in terms of:
- Option price estimate
- Standard error with reduction percentages
- Computation time with relative factors
- Efficiency gain metrics

The enhanced results presentation features:
- Visual indicators (↑/↓) to highlight improvements and deteriorations
- Color-coding (green for improvements, red for deteriorations)
- Normalized metrics for easy comparison
- Consistent decimal precision across all metrics
- Clear section headers and structured organization

The analysis demonstrates:
- Significant variance reduction from control variate techniques (up to 97.4% for Asian options)
- Modest computational overhead for variance reduction methods
- Substantial efficiency improvements when combining techniques (up to 2134.5× for Asian options)
- Different effectiveness by option type (control variates highly effective for Asian options but less so for lookback options with stochastic volatility)
- Trade-offs between computation time and estimation accuracy
