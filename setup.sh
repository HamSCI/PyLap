#!/bin/bash
# PyLap Setup Script
# Uses a Python virtual environment for package management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# check if the program is already setup
if [[ ${PHARLAP_HOME} = "" ]]; then

echo "First Time Setup"
echo "Please enter the filepath of the project directory"
echo "Example - /home/username/pylap_project"
read x
echo ${x}

# Detect PHaRLAP version (look for pharlap_* directory)
PHARLAP_DIR=$(find "${x}" -maxdepth 1 -type d -name "pharlap_*" 2>/dev/null | head -1)
if [[ -n "${PHARLAP_DIR}" ]]; then
    export PHARLAP_HOME="${PHARLAP_DIR}"
    echo "Found PHaRLAP at: ${PHARLAP_HOME}"
else
    echo "Warning: PHaRLAP directory not found in ${x}"
    echo "Please ensure PHaRLAP is installed before running raytracing examples"
    export PHARLAP_HOME="${x}/pharlap_4.7.4"
fi
echo "export PHARLAP_HOME=${PHARLAP_HOME}" >> ~/.bashrc

if [[ -d "${x}/PyLap" ]] ;
then
export PYTHONPATH="${x}/PyLap" 
echo "export PYTHONPATH=${x}/PyLap" >> ~/.bashrc
echo "Pylap filepath found"
elif [[ -d "${x}/PyLap-main" ]] ;
then
export PYTHONPATH="${x}/PyLap-main" 
echo "export PYTHONPATH=${x}/PyLap-main" >> ~/.bashrc
echo "Pylap filepath found"
else 
echo "Pylap filepath not found"
fi

export DIR_MODELS_REF_DAT="${PHARLAP_HOME}/dat"
echo "export DIR_MODELS_REF_DAT=${DIR_MODELS_REF_DAT}" >> ~/.bashrc

# Install system dependencies (gfortran provides the Fortran runtime — Intel Fortran is no longer required)
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-tk python3-pil python3-pil.imagetk libqt5gui5 python3-pyqt5 
sudo apt-get install -y libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev
sudo apt-get install -y python3-pip python3-venv gfortran

# Create and activate virtual environment
echo "Creating Python virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install Python dependencies in venv
echo "Installing Python packages in virtual environment..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install IRI-2020 from GitHub (requires gfortran)
echo "Installing IRI-2020 from GitHub..."
pip install git+https://github.com/space-physics/iri2020.git

# Build and install PyLap
echo "Building PyLap..."
python3 setup.py install

# Add venv activation to bashrc
echo "# PyLap virtual environment" >> ~/.bashrc
echo "alias pylap-activate='source ${VENV_DIR}/bin/activate'" >> ~/.bashrc
echo ""
echo "Setup complete!"
echo "To activate the PyLap environment, run: source ${VENV_DIR}/bin/activate"
echo "Or use the alias: pylap-activate"

else
echo "Pylap is already setup"
echo "To activate the virtual environment: source ${SCRIPT_DIR}/venv/bin/activate"
echo "Or use the alias: pylap-activate"
echo ""
echo "If you wish to redo the setup:"
echo "1. Edit ~/.bashrc and remove the PyLap environment variables"
echo "2. Delete the venv directory: rm -rf ${SCRIPT_DIR}/venv"
echo "3. Run this script again"
fi
