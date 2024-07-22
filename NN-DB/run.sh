#!/bin/bash
#SBATCH -n 16
#SBATCH -t 2:00:00
#SBATCH -p genoa


source activate ase-ovito

python get_dsoap.py

