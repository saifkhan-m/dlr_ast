# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np
import dataloader
from tqdm import tqdm
#TYPE='red'
#target_dlr=3072
TYPE='red10'
target_dlr= 1024
TYPE='red5'
target_dlr= 512
audio_conf = {'num_mel_bins': 128, 'target_length': target_dlr, 'freqm': 24, 'timem': 192, 'mixup': 0.0,'get_norm_stats':True,'mode':'train', 'dataset':'dlr' }

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset('/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/data/datafiles/dlr_train_data_'+TYPE+'.json', label_csv='/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/data/dlr_class_label_'+TYPE+'.csv',
                                audio_conf=audio_conf), batch_size=1000,  shuffle=False, num_workers=8, pin_memory=True)
#train_loader = torch.utils.data.DataLoader(
#    dataloader.AudiosetDataset('/home/khan_mo/thesis/important Git Lib/ast/egs/esc50/data/datafiles/esc_train_data_1.json',
#                               label_csv='/home/khan_mo/thesis/important Git Lib/ast/egs/esc50/data/esc_class_labels_indices.csv',
#                                audio_conf=audio_conf), batch_size=1000,  shuffle=False, num_workers=8, pin_memory=True)


mean=[]
std=[]
for i, (audio_input, labels) in tqdm(enumerate(train_loader)):
    l=labels.tolist()
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    #print(cur_mean, cur_std)
    #print(i)
print(f'Norm stat for TYPE: {TYPE} and target length: {target_dlr}')
print('mean',np.mean(mean))
print('STD', np.mean(std))
print('Norm Stats Done')