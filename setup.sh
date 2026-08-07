#!/bin/bash

echo "Activating virtual environment..."
source "./venv/Scripts/activate"

echo "Building executable..."
# pyinstaller --onefile --noconsole main.py
# we will build without "noconsole" for now to see any errors in the console
pyinstaller --onefile main.py

echo "Build complete!"
read -p "Press Enter to exit..."