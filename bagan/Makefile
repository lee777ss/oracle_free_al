
# ===============================
# PDF Export Makefile (Windows)
# ===============================

NOTEBOOKS := $(shell find . -name "*.ipynb")
PYFILES   := $(shell find . -name "*.py")

.PHONY: all notebooks code clean

all: notebooks code
	@echo "✅ All PDFs generated."

# -------------------------------
# Convert Jupyter notebooks
# -------------------------------
notebooks:
	@echo "📘 Converting notebooks to PDF..."
	jupyter nbconvert --to pdf $(NOTEBOOKS)

# -------------------------------
# Convert Python files
# -------------------------------
code:
	@echo "📗 Converting Python files to PDF..."
	for f in $(PYFILES); do \
		pygmentize -f pdf -O linenos=1 -o "$${f%.py}.pdf" "$$f"; \
	done

# -------------------------------
# Clean generated PDFs
# -------------------------------
clean:
	@echo "🧹 Removing generated PDFs..."
	find . -name "*.pdf" -delete
