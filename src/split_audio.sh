#!/bin/bash
#SBATCH --output=output/split_out_%j.txt
#SBATCH --gres=gpu:4              # Anzahl GPUs (pro node)
#SBATCH --mail-user=mohd.khan@dlr.de
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=84000
#SBATCH --time=2-00:00            # Max-Time (DD-HH:MM)


python SplitAudioToChunks.py
