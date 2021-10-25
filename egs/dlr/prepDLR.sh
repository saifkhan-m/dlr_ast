#!/bin/bash
#SBATCH --output=out_%j.txt
#SBATCH --gres=gpu:4              # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32000M
#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)


python processdata.py
python prep_dlr.py
#python folderto16k.py