#!/bin/bash
#SBATCH --nodelist=gpu00[1-4]
#SBATCH --output=output/out_predict_%j.txt
#SBATCH --gres=gpu:1            # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=84000

#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)
#SBATCH --job-name="ast-predict"
python checkTagging.py