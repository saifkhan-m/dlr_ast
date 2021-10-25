#!/bin/bash
#SBATCH --output=output/out_predict_%j.txt
#SBATCH --gres=gpu:1            # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=84000
#SBATCH --job-name="ast-dlr_predict"
#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)


python TagAudio.py
