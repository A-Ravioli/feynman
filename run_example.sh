#!/bin/bash

# Ensure the examples directory exists
mkdir -p examples

# Setup a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package
pip install -e .

# Run the double slit example
feynman run examples/double_slit.phys --visualize

# Deactivate the virtual environment
deactivate

echo "Example completed! Check the visualization that opened in your browser." 