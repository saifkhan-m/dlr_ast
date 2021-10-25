# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

TYPE='red15'
def create_16k(base_dir, meta):
    for index, file in meta.iterrows():
        audio = file['filename']
        cat_fol=file['folder']
        splitDir = os.path.join(base_dir,'final_data16k', cat_fol)
        Path(splitDir).mkdir(parents=True, exist_ok=True)
        #print('sox ' + base_dir + '/final_data/'+cat_fol +'/'+ audio + ' -r 16000 ' + base_dir + '/final_data16k/' +cat_fol+ '/'+audio)
        if index % 1000==0:
            print('------------------------processed examples------------------', index)
        try:
            os.system('sox -G -v 0.98 ' + base_dir + '/final_data/'+cat_fol +'/'+ audio + ' -r 16000 ' + base_dir + '/final_data16k/' +cat_fol+ '/'+audio)
        except:
            print('sox ' + base_dir + '/final_data/'+cat_fol +'/'+ audio + ' -r 16000 ' + base_dir + '/final_data16k/' +cat_fol+ '/'+audio)


base_dir= "/home/projects/SocialMediaAnalysis/audioDLR"
meta = pd.read_csv('data/dlr_data_folder_meta_'+TYPE+'.csv' )
#base_path_16k = "/home/projects/SocialMediaAnalysis/audioDLR/final_data16k"
base_path_16k = "/home/projects/SocialMediaAnalysis/audioDLR/reduced_Data"
base_path_16k = "/home/projects/SocialMediaAnalysis/audioDLR/reduced_Data15"
#create_16k(base_dir, meta)

X = meta.loc[:, 'folder':'filename']
y = meta['label']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y)

X_train['label']= y_train
X_test['label']= y_test

train = X_train.values.tolist()
test = X_test.values.tolist()
train_wav_list = []
eval_wav_list = []


for val in train:
    cur_dict = {"wav": os.path.join(base_path_16k , val[0] ,val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
    train_wav_list.append(cur_dict)

for val in test:
    cur_dict = {"wav": os.path.join(base_path_16k , val[0] ,val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
    eval_wav_list.append(cur_dict)


with open('data/datafiles/dlr_train_data_'+TYPE +'.json', 'w') as f:
    json.dump({'data': train_wav_list}, f, indent=1)

with open('data/datafiles/dlr_eval_data_'+TYPE+'.json', 'w') as f:
    json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished DLR data Preparation')