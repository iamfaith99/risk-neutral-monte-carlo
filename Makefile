# Makefile for Risk-Neutral Monte Carlo Pricing Project
# Following functional programming principles with clean, elegant code

# Variables
DEFAULT_QUARTO_FILE = final_project.qmd
JAX_QUARTO_FILE = final_project_jax.qmd
POETRY = poetry
VENV_PATH = .venv
KERNEL_NAME = risk-neutral-monte-carlo

# Default target
.PHONY: all
all: render-all

# Ensure Jupyter kernel is set up to use Poetry's virtual environment
.PHONY: setup-kernel
setup-kernel:
	@echo "Setting up Jupyter kernel to use Poetry's virtual environment..."
	$(POETRY) run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Python ($(KERNEL_NAME))"
	@echo "Kernel setup complete."

# Render a specific Quarto document to HTML
.PHONY: render
render:
	@if [ -z "$(QUARTO_FILE)" ]; then \
		echo "Rendering $(DEFAULT_QUARTO_FILE) to HTML..."; \
		$(POETRY) run quarto render $(DEFAULT_QUARTO_FILE); \
		echo "Done! Output saved to $(DEFAULT_QUARTO_FILE:.qmd=.html)"; \
	else \
		echo "Rendering $(QUARTO_FILE) to HTML..."; \
		$(POETRY) run quarto render $(QUARTO_FILE); \
		echo "Done! Output saved to $(QUARTO_FILE:.qmd=.html)"; \
	fi

# Render all Quarto documents
.PHONY: render-all
render-all:
	@echo "Rendering all Quarto documents..."
	$(POETRY) run quarto render $(DEFAULT_QUARTO_FILE)
	$(POETRY) run quarto render $(JAX_QUARTO_FILE)
	@echo "Done! All documents rendered successfully."

# View a specific rendered HTML file in the default browser
.PHONY: view
view:
	@if [ -z "$(QUARTO_FILE)" ]; then \
		echo "Opening $(DEFAULT_QUARTO_FILE:.qmd=.html) in browser..."; \
		open $(DEFAULT_QUARTO_FILE:.qmd=.html); \
	else \
		echo "Opening $(QUARTO_FILE:.qmd=.html) in browser..."; \
		open $(QUARTO_FILE:.qmd=.html); \
	fi

# Convert and execute a specific Quarto document
.PHONY: run
run: setup-kernel
	@if [ -z "$(QUARTO_FILE)" ]; then \
		echo "Converting $(DEFAULT_QUARTO_FILE) to notebook and executing..."; \
		$(POETRY) run quarto convert $(DEFAULT_QUARTO_FILE) -o $(DEFAULT_QUARTO_FILE:.qmd=.ipynb); \
		$(POETRY) run jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $(DEFAULT_QUARTO_FILE:.qmd=.ipynb); \
		echo "Done! Code execution complete."; \
	else \
		echo "Converting $(QUARTO_FILE) to notebook and executing..."; \
		$(POETRY) run quarto convert $(QUARTO_FILE) -o $(QUARTO_FILE:.qmd=.ipynb); \
		$(POETRY) run jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $(QUARTO_FILE:.qmd=.ipynb); \
		echo "Done! Code execution complete."; \
	fi

# Run all Quarto documents
.PHONY: run-all
run-all: setup-kernel
	@echo "Running all Quarto documents..."
	$(POETRY) run quarto convert $(DEFAULT_QUARTO_FILE) -o $(DEFAULT_QUARTO_FILE:.qmd=.ipynb)
	$(POETRY) run jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $(DEFAULT_QUARTO_FILE:.qmd=.ipynb)
	$(POETRY) run quarto convert $(JAX_QUARTO_FILE) -o $(JAX_QUARTO_FILE:.qmd=.ipynb)
	$(POETRY) run jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $(JAX_QUARTO_FILE:.qmd=.ipynb)
	@echo "Done! All code execution complete."

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f *.html
	rm -f *.ipynb
	rm -rf _files/
	rm -rf _freeze/

# Display help information
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup-kernel  - Set up Jupyter kernel with Poetry environment"
	@echo "  make render        - Render default Quarto document ($(DEFAULT_QUARTO_FILE))"
	@echo "  make render QUARTO_FILE=file.qmd - Render specific Quarto document"
	@echo "  make render-all    - Render all Quarto documents"
	@echo "  make view          - View default rendered HTML in browser"
	@echo "  make view QUARTO_FILE=file.qmd - View specific rendered HTML"
	@echo "  make run           - Run default Quarto document"
	@echo "  make run QUARTO_FILE=file.qmd - Run specific Quarto document"
	@echo "  make run-all       - Run all Quarto documents"
	@echo "  make clean         - Clean generated files"
	@echo "Clean complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Render the Quarto document to HTML"
	@echo "  make render   - Same as above"
	@echo "  make view     - Open the rendered HTML file in browser"
	@echo "  make run      - Convert to notebook and execute the Python code using Poetry's environment"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help message"