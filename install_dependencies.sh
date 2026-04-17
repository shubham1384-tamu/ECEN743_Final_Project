#!/bin/bash

# Installation script for ECEN743 Final Project dependencies
# Run this after cloning the repo: ./install_dependencies.sh

set -e  # Exit on any error

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing Linux-specific packages from linux_requirements.txt..."
# Note: This assumes you have sudo access or are in a container with package manager
# On TAMU HPRC, you may need to adjust or skip this if packages are pre-installed
# Uncomment the next line if you have sudo and need these packages:
# sudo apt-get update && sudo apt-get install -y $(cat linux_requirements.txt)

echo "Installing f110_gym (F1TENTH Gym environment)..."
# f110_gym is not on PyPI, so we install from GitHub
# But first, we need to handle its dependency gym==0.19.0 which has broken metadata

# Clone and patch gym==0.19.0
if [ ! -d "gym-0.19.0" ]; then
    git clone https://github.com/openai/gym.git gym-0.19.0
    cd gym-0.19.0
    git checkout 0.19.0
    # Patch the invalid requirement in setup.py
    sed -i 's/opencv-python>=3\./opencv-python>=3.0.0/' setup.py
    pip install -e .
    cd ..
else
    echo "gym-0.19.0 already cloned, skipping..."
fi

# Install f110_gym from GitHub
pip install git+https://github.com/F1TENTH/f110_gym.git

echo "Installation complete!"
echo "You can now run the project with: python llm_mpc.py --help"