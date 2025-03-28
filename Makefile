# Makefile for Risk-Neutral Monte Carlo Pricing Project
# Following professional Python development workflow principles

# Variables
QUARTO_FILE = final_project.qmd
HTML_OUTPUT = final_project.html
POETRY = poetry

# Default target
.PHONY: all
all: render

# Render the Quarto document to HTML
.PHONY: render
render:
	@echo "Rendering $(QUARTO_FILE) to HTML..."
	$(POETRY) run quarto render $(QUARTO_FILE)
	@echo "Done! Output saved to $(HTML_OUTPUT)"

# View the rendered HTML file in the default browser
.PHONY: view
view:
	@echo "Opening $(HTML_OUTPUT) in browser..."
	open $(HTML_OUTPUT)

# Run the project as a module
.PHONY: run
run:
	$(POETRY) run python -m final_project

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f $(HTML_OUTPUT)
	rm -rf _files/
	rm -rf _freeze/
	@echo "Clean complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Render the Quarto document to HTML"
	@echo "  make render   - Same as above"
	@echo "  make view     - Open the rendered HTML file in browser"
	@echo "  make run      - Run the project as a Python module"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help message"