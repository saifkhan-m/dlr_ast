import numpy as np
import json
import os
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from os import walk
from tqdm import tqdm
def get_relevantfiles(data_path):
    eventfiles= []
    for (dirpath, dirnames, filenames) in walk(data_path):
        eventfiles.extend([f for f in filenames if 'nos'  in f])
    return  eventfiles

def create_16k(base_dir, dest_dir,files):
    for audio in tqdm(files):
        source_file=  base_dir + '/'+ audio
        dest_file= dest_dir + '/' + audio
        #print('sox -G -v 0.98 ' + source_file + ' -r 16000 ' + dest_file)
        try:
            os.system('sox -G -v 0.98 ' + source_file + ' -r 16000 ' + dest_file)
        except:
            print('sox -G -v 0.98 ' + source_file + ' -r 16000 ' + dest_file)

base_dir= "/home/projects/SocialMediaAnalysis/audioDLR/reduced_Data/Silence"
dest_dir= "/home/projects/SocialMediaAnalysis/audioDLR/reduced_Data/Silences"
files= get_relevantfiles(base_dir)
create_16k(base_dir, dest_dir,files)

