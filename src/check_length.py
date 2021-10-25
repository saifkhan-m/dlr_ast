import os
import torchaudio
from  os import walk
import pandas as pd
import csv
import os

import csv

# CReate a csv file that gives the detail about the files that has length less tha 400
# Was implemented as a check to remove 0 length files-

# the later logic is to remove those files
meta = pd.read_csv('../egs/dlr/data/dlr_data_folder_meta_all.csv' )
X = meta.loc[:, 'folder':'filename']
y = meta['label']
train = X.values.tolist()

wavfiles = []


base_path_16k = "/home/projects/SocialMediaAnalysis/audioDLR/final_data16k"
for val in train:
    #cur_dict = {"wav": os.path.join(base_path_16k , val[0] ,val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
    wavfiles.append(os.path.join(base_path_16k , val[0] ,val[1]))

print(f' the total wavfiles: {len(wavfiles)}')
lst=[]
for wavfile in wavfiles:
    waveform, sr = torchaudio.load(wavfile)
    waveform = waveform - waveform.mean()
    waveform = waveform[-1, :]
    if len(waveform) < 400:
        lst.append([wavfile,len(waveform)])

with open("length.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerow([ 'filename', 'length'])
    csvWriter.writerows(lst)


with open("length.csv", 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i==0:
            continue
        os.remove(row[0])