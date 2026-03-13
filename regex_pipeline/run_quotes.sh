#!/bin/bash

#SBATCH --job-name=quotes_ex
#SBATCH --partition=ICF-Free
#SBATCH --nodelist=scotia08
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s9905758/slogs/%x_%j.out
#SBATCH --error=/home/s9905758/slogs/%x_%j.err

# Point to the specific cluster path
export CUDA_HOME=/opt/cuda-12.6.3
export LD_LIBRARY_PATH=/opt/cuda-12.6.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

nvidia-smi

# Set up this project's python dir and environment
cd /home/s9905758/regex_pipeline
source /home/s9905758/.venv_38/bin/activate

# Print info to the log for debugging
echo "Starting job at: $(date)"
echo "Using python: $(which python)"

# Make sure it looks for things in the python dir
export PYTHONPATH=$PYTHONPATH:/home/s9905758/regex_pipeline

# Execute
# -u ensures tqdm and print statements show up in slogs immediately
python -u run_df_extraction.py

# Check
echo "Job finished at: $(date)"