#!/bin/bash
#SBATCH --cluster=whale
#SBATCH --partition=long
#SBATCH --account=researcher
#SBATCH --job-name=flowmatch_if
#SBATCH --output=flowmatch_if.out
#SBATCH --error=flowmatch_if.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10000M
#SBATCH --time=08-00:00
#SBATCH --ntasks=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# label smoothing parameter here is used to match the equivalent used in PMPNN
python utils/train_flow_model.py --backbone_noise 0.1 --label_smoothing 0.0909 --path_for_training_data YOUR/PATH/TO/PROTEIN/MPNN/DATA/pdb_2021aug02

