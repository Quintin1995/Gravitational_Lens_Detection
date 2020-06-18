#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --mem=25000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
module load TensorFlow/2.0.0-foss-2019a-Python-3.7.2
module load matplotlib/3.0.3-fosscuda-2019a-Python-3.7.2
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4
python3 cnn.py
