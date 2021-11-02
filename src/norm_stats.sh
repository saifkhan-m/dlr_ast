#!/bin/bash
#SBATCH --output=output/out_norm_%j.txt
#SBATCH --gres=gpu:2              # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=42000M
#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)


python get_norm_stats.py
