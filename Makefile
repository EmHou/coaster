# Define the Python interpreter
PYTHON ?= python3

# Define the virtual environment directory
VENV_DIR = .venv

# Define the path to the pip executable within the virtual environment
VENV_PIP = $(VENV_DIR)/bin/pip

# Define the path to the python executable within the virtual environment
VENV_PYTHON = $(VENV_DIR)/bin/python

# Phony targets are targets that don't represent files
.PHONY: setup clean

# Default target
all: setup

# Target to set up the virtual environment and install dependencies
setup: $(VENV_DIR)/touchfile
	@echo "Virtual environment up-to-date."

# Use a marker file to track if requirements are installed
$(VENV_DIR)/touchfile: requirements.txt $(VENV_DIR)/bin/activate
	@echo "Installing/updating requirements..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@touch $@

# Target to create the virtual environment if it doesn't exist
$(VENV_DIR)/bin/activate:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)

# Target to clean the virtual environment
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."

