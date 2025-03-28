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

### Rendering the Document

To render the Quarto document to HTML:

```bash
make render
```

### Viewing the Document

To open the rendered HTML file in your default browser:

```bash
make view
```

### Running the Project

To run the project as a Python module:

```bash
make run
```

### Cleaning Generated Files

To clean up generated files:

```bash
make clean
```

### Getting Help

For a list of available commands:

```bash
make help
```

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

## License

MIT