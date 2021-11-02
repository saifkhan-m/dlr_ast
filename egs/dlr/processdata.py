import os
import csv
from os import walk
from os import walk
from util import get_heirarchy_of_categories

#TYPE='all'
TYPE='red15'
TYPE= 'red10'
seconds=10
TYPE= 'red5'
seconds=5
def create_filenamecsv(data_folder):
    class_folders = [x[0].split('/')[-1] for x in os.walk(data_folder)][1:]
    for_csv = []
    label_inc=0
    label_count = {}
    for index, folder in enumerate(class_folders):
        foldername = folder
        class_heirarchy=get_heirarchy_of_categories()
        if not class_heirarchy[folder]\
                or class_heirarchy[folder][0] in ['Strassenverkehr', 'Zuege-Bahnen','Grosse Fahrzeuge']:
                #Condition to only takes Autos, Flugzeug, Silence and Nebengerausch
            continue
        classlabel= class_heirarchy[folder][0]
        if classlabel in label_count:
            label = label_count[classlabel]
        else:
            
            label=label_inc
            label_count[classlabel]=label_inc
            label_inc += 1

        for (dirpath, dirnames, filenames) in walk(os.path.join(data_folder, folder)):
            wavfiles = [f for f in filenames if f.endswith('wav')]
        for wavfile in wavfiles:
            #wavfile= wavfile.replace(' ', '_').replace('(','').replace(')','').replace(',','')
            for_csv.append([foldername, wavfile, label])
    print(label_count)
    with open("data/dlr_data_folder_meta_"+TYPE+".csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(['folder', 'filename', 'label'])
        csvWriter.writerows(for_csv)

def create_class_labelcsv(data_folder):

    class_folders = [x[0].split('/')[-1] for x in os.walk(data_folder)][1:]
    for_csv = []
    label_inc = 0
    label_count = {}
    for index, folder in enumerate(class_folders):
        class_heirarchy = get_heirarchy_of_categories()
        if not class_heirarchy[folder] \
                or class_heirarchy[folder][0] in ['Strassenverkehr', 'Zuege-Bahnen','Grosse Fahrzeuge']:
                #Condition to only takes Autos, Flugzeug, Silence and Nebengerausch
            continue
        classlabel = class_heirarchy[folder][0]
        if classlabel in label_count:
            label = label_count[classlabel]
        else:
            label = label_inc
            label_count[classlabel] = label_inc
            label_inc += 1

        for (dirpath, dirnames, filenames) in walk(os.path.join(data_folder, folder)):
            wavfiles = [f for f in filenames if f.endswith('wav')]
        for wavfile in wavfiles:
            for_csv.append([label, '/m/21rwj' + str(label).zfill(2), folder])
    print(label_count)
    with open("data/dlr_class_label_"+TYPE+".csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(['index', 'mid', 'display_name'])
        csvWriter.writerows(for_csv)


data_fol = f'/home/projects/SocialMediaAnalysis/audioDLR/reduced_Data{seconds}'
create_class_labelcsv(data_fol)
create_filenamecsv(data_fol)

