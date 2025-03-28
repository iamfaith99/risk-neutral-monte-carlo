# Risk-Neutral Monte Carlo Pricing

This repository contains a computational finance project implementing risk-neutral Monte Carlo pricing for exotic options with variance reduction techniques. The project is structured according to functional programming principles, with pure functions, explicit dependencies, and clear separation of concerns.

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

## Project Structure

- `final_project.qmd`: Main Quarto document containing the implementation, analysis, and results
- `Makefile`: Contains commands for rendering the document and other project tasks
- `pyproject.toml` & `poetry.lock`: Poetry configuration for dependency management

## Requirements

- Python 3.8+
- Poetry (for dependency management)
- Quarto

All Python dependencies are managed through Poetry and include:
- NumPy
- Pandas
- Matplotlib
- SciPy
- Jupyter

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

# Render the Quarto document to HTML
make render

# View the rendered HTML file in your browser
make view

# Execute the Python code in the Quarto document
make run

# Clean generated files
make clean
```

## Makefile Commands

The project includes a Makefile with the following targets:

- `make render`: Renders the Quarto document to HTML using Poetry
- `make view`: Opens the rendered HTML file in the default browser
- `make run`: Executes the Python code in the Quarto document using Poetry's environment
- `make clean`: Removes generated files
- `make help`: Lists available commands

## Implementation Details

The implementation follows functional programming principles:
- Pure functions with no side effects
- Immutable data structures where possible
- Clear separation of concerns
- Explicit dependencies

The Monte Carlo simulations use various variance reduction techniques to improve efficiency:
- **Antithetic Sampling**: Generates negatively correlated pairs of paths to reduce variance
- **Control Variate**: Uses analytical solutions for related options as control variates
- **Combined Approach**: Applies both techniques together for maximum variance reduction

## Results

The project includes a detailed analysis of the performance of each variance reduction technique in terms of:
- Option price estimate
- Standard error
- Computation time
- Efficiency (variance reduction per unit of computation time)