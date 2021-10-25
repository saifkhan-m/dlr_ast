#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --gres=gpu:1              # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32000M
#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)


python remove0lenfile.py

