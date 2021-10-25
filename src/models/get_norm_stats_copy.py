# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

from src import dataloader

audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 24, 'timem': 192, 'mixup': 0.5,'mode':'train', }

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset('/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/data/datafiles/dlr_train_data.json', label_csv='/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/data/dlr_class_label.csv',
                                audio_conf=audio_conf), batch_size=1000,  shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))