#!/bin/bash

#SBATCH -p recon
#SBATCH -q recon-prio
#SBATCH -t 0-12:00:00
#SBATCH -c 16
#SBATCH -o output/out-%A_%a.out
#SBATCH -e output/out-%A_%a.out
#SBATCH --gres gpu:1

python ChamferSequenceError.py

