#!/bin/bash

# TODO: untested script!

# Check for the -local argument
LOCAL=false
if [[ "$1" == "-local" ]]; then
    LOCAL=true
fi

# Run make set-up-env with or without the -local parameter
if $LOCAL; then
    echo "Running make set-up-env-local"
    make set-up-env-local
else
    echo "Running make set-up-env"
    make set-up-env
fi

# Move to the ./lib directory
cd ./lib || { echo "lib directory not found"; exit 1; }

# Install private-pgm-master
echo "Installing private-pgm-master..."
cd private-pgm-master || { echo "private-pgm-master directory not found"; exit 1; }
pip install . || { echo "Failed to install private-pgm-master"; exit 1; }
cd ..  # Return to ./lib directory

# Install querysnout-main
echo "Installing optimized_qbs in querysnout-main..."
cd querysnout-main/src/optimized_qbs || { echo "optimized_qbs directory not found in querysnout-main"; exit 1; }
python setup.py install || { echo "Failed to install optimized_qbs"; exit 1; }
cd ../../../  # Return to ./lib directory

# Install additional requirements from requirements_part2.txt
echo "Installing requirements from requirements_part2.txt..."
pip install -r requirements_part2.txt || { echo "Failed to install requirements from requirements_part2.txt"; exit 1; }

# Install reprosyn
echo "Installing reprosyn..."
cd reprosyn || { echo "reprosyn directory not found"; exit 1; }
pip install . || { echo "Failed to install reprosyn"; exit 1; }
cd ..  # Return to ./lib directory

# Install synthcity package
echo "Installing synthcity..."
pip install synthcity || { echo "Failed to install synthcity"; exit 1; }

echo "All dependencies installed successfully."
