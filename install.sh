#!/bin/bash
################################################################################
### This script installs the necessary packages inside an already created 
### environment, called $ENV_NAME (e.g. discrete_guidance)
################################################################################
ENV_NAME=$1
# Assume we are in an activated environment with python 3.9, e.g. created by
# conda create -n $ENV_NAME --yes python=3.9
    
# Install standard datascience modules
conda install -y numpy==1.26.2 matplotlib==3.8.2 pandas==2.1.3 scipy==1.11.4 tqdm jupyter seaborn

# Install pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install extra packages required for basic packages
conda install -y tensorboard ml-collections scikit-learn torchmetrics

# Install cheminformatics pipeline
conda install -y rdkit chembl_structure_pipeline

# Metrics for cifar10 evaluation
pip install pytorch-gan-metrics

############################################################################################################################################################
# Make the project itself a python package (to enable more convenient absolute imports)
# 1) Generate a setup.py file for the project with 'project' as name
# Remark: Enable echo to interpret backslash escapes (required for the two newline characters '\n\n' below)
echo -e "from setuptools import setup, find_packages\n\nsetup(name='${ENV_NAME}', version='1.0', packages=find_packages())" > setup.py

# 2) Install the project itself by its name and make it editable (=> use '-e' for editable)
# Remark: This requires the setup.py file created above
pip install -e .
############################################################################################################################################################

echo " "
echo "Installation done"
