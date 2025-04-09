# Makefile for Risk-Neutral Monte Carlo Pricing Project
# Following functional programming principles with clean, elegant code

# Variables
QUARTO_FILE = final_project.qmd
POETRY = poetry
VENV_PATH = .venv
KERNEL_NAME = risk-neutral-monte-carlo

# Default target
.PHONY: all
all: render

# Ensure Jupyter kernel is set up to use Poetry's virtual environment
.PHONY: setup-kernel
setup-kernel:
	@echo "Setting up Jupyter kernel to use Poetry's virtual environment..."
	$(POETRY) run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Python ($(KERNEL_NAME))"
	@echo "Kernel setup complete."

# Render the Quarto document to HTML
.PHONY: render
render:
	echo "Rendering $(QUARTO_FILE) to HTML..."
	$(POETRY) run quarto render $(QUARTO_FILE)
	echo "Done! Output saved to $(QUARTO_FILE:.qmd=.html)"



# View the rendered HTML file in the default browser
.PHONY: view
view:
	echo "Opening $(QUARTO_FILE:.qmd=.html) in browser..."
	open $(QUARTO_FILE:.qmd=.html)

# Convert and execute the Quarto document
.PHONY: run
run: setup-kernel
	echo "Converting $(QUARTO_FILE) to notebook and executing..."
	$(POETRY) run quarto convert $(QUARTO_FILE) -o $(QUARTO_FILE:.qmd=.ipynb)
	$(POETRY) run jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=$(KERNEL_NAME) $(QUARTO_FILE:.qmd=.ipynb)
	echo "Done! Code execution complete."



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
	@echo "  make render        - Render the Quarto document ($(QUARTO_FILE))"
	@echo "  make view          - View rendered HTML in browser"
	@echo "  make run           - Run the Quarto document"
	@echo "  make clean         - Clean generated files"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Note: The document includes enhanced results presentation with:"
	@echo "  - Visual indicators and color-coding for performance metrics"
	@echo "  - Normalized columns for variance reduction and time factors"
	@echo "  - Consistent formatting and clear section organization"