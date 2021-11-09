import pandas as pd
from datetime import timedelta
from datetime import datetime
import librosa
import csv
import re, os
import time
from psds_eval import (PSDSEval, plot_psd_roc, plot_per_class_psd_roc)
import numpy as np

def getEventDF(event_result_path, comment_index):
    df = pd.read_csv(event_result_path
                     , encoding="ISO-8859-1"
                     , sep='\t'
                     # , names=['i1','start_time', 'end_time','LASmax_dt-Maximalpegel(innen) [dB(A)]','LASmax_dt-Pegelanstieg(maximal,innen) [dB(A)/s]','LASmax_dt-Leq3-Geraeusch(innen) [dB(A)]','LASmax_dt-SEL(innen) [dB(A)]','LASmax_dt-Leq3_1min(innen) [dB(A)]','LASmax_dt- SNR','LASmax_dt- MNR','Kommentar 1','Kommentar 2']
                     , usecols=[0, 1, 2, comment_index,comment_index+1]  # 11 for new #36 for veu
                     , skiprows=[0, -2, -1]
                     , header=None
                     , quoting=csv.QUOTE_NONE
                     )
    df[comment_index] = df[comment_index]+'$'+ df[comment_index+1]
    df = df.drop(comment_index+1, 1)
    df = df.rename(columns={x: y for x, y in zip(df.columns, range(0, len(df.columns)))})
    return df[:-1]


def convert_negative(tdelta):
    if tdelta.days < 0:
        tdelta = timedelta(
            days=0,
            seconds=tdelta.seconds,
            microseconds=tdelta.microseconds
        )
    return tdelta

def getAllTime(actual_start, start, end):
    FMT = '%H:%M:%S'
    tdelta1 = convert_negative(datetime.strptime(start, FMT) - datetime.strptime(actual_start, FMT))
    tdelta2 = convert_negative(datetime.strptime(end, FMT) - datetime.strptime(actual_start, FMT))
    return convert_to_seconds(str(tdelta1)), convert_to_seconds(str(tdelta2))

def convert_to_seconds(td):
    x = time.strptime(td, '%H:%M:%S')
    return int(timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())

def getSilenceTimes(start, end):
    pass

def get_start_time(filename):
    filename = filename.split("/")[-1]
    parentDir, datefolder, startTime, _ = filename.split("_")
    if len(startTime) != 6:
        raise ValueError("startime less than 6 digits")
    else:
        startTime = ':'.join(startTime[i:i + 2] for i in range(0, len(startTime), 2))
    return startTime

def getLabel(label):

    cats = {'Probandengeraeusche': ['Nebengeraeusche'], 'Nachbarschaftslaerm': ['Nebengeraeusche'],'Probandengeraeusch': ['Nebengeraeusche'],
            'Nachbarschaftslaer': ['Nebengeraeusche'], 'Nachbarschaftslärm': ['Nebengeraeusche'], 'Probandengeräusche':['Nebengeraeusche'],
            'Umdrehen': ['Nebengeraeusche'], 'Umdrehen im Bett':['Nebengeraeusche'],'Grosse Fahrzeuge':['Autos', 'Strassenverkehr'],
                'Raumknacken': ['Nebengeraeusche'], 'Auto': ['Autos', 'Strassenverkehr'],'Strassenverkehr': ['Autos', 'Strassenverkehr'],
                'Autos': ['Autos', 'Strassenverkehr'],
                'Motorrad': ['Autos'],
                'Transporter': ['Autos'], 'Flugzeug landend': ['Flugzeug'], 'Flugzeug startend': ['Flugzeug'],
                'Flugzeug': ['Flugzeug'], 'Gueterzug': ['Zuege-Bahnen'], 'Personenzug': ['Zuege-Bahnen'],
                'Straßenbahn': ['Zuege-Bahnen'], 'entgegenkommende_Gueterzug': ['Zuege-Bahnen'],
                'Gueterzug_langsam_fahrend': ['Zuege-Bahnen'], 'entgegenkommende_Personenzug': ['Zuege-Bahnen'],
                'Personenzug_bremsend': ['Zuege-Bahnen'], 'Bahn_Rangierfahrzeug_etc': ['Zuege-Bahnen'],
                'Personenzug_langsam_fahrend': ['Zuege-Bahnen'], 'Güterzug_langsam_fahrend': ['Zuege-Bahnen'],
                'Gueterzug_bremsend': ['Zuege-Bahnen'], 'Messung_Start': [], 'Messung_Ende': [], 'Umdrehen_im_Bett': [],
                'Vogelgezwitscher': [], 'Autobahn': [], 'Fahrzeugkolonne': [], 'lauter_Regen': [],
                'Flughafenbodenlärm': [],
                'Aufstehen_Toilettengang_etc': [], 'Husten_Raeuspern': [], 'Sirene_Polizei_Notarzt_Feuerwehr': [],
                'entgegenkommende_Auto': [], 'Wind': [], 'Schnarchen': [], 'Gewitter': [],
                'Zuege-Bahnen': ['Zuege-Bahnen'], 'Silence': ['Silence'], 'Nebengeraeusche': ['Nebengeraeusche']}
    if label in cats and len(cats[label])>0:
        return cats[label][0]
    else:
        print(label, 'does not exist in the dict' )
        return None

def getgroundtruthDF(file, actualStartTime, wavfile, event_index):
    endwav = int(librosa.get_duration(filename=wavfile))
    wavfile = wavfile.split('/')[-1]
    events = getEventDF(file, event_index)
    list_with_silence = []
    stf=True
    for i, row in events.iterrows():

        labels = re.sub(' +', ' ', row[3])
        labels = labels.split('$')
        labels = [lab for lab in labels if lab != ' ']
        labels[0] = labels[0].strip().split(' ', 1)[1]
        for label in labels:
            label = getLabel(label.strip())
            if not label:
                continue
            elif label in ['Flugzeug', 'Silence', 'Nebengeraeusche', 'Autos']:
                pass
            else:
                print(label,'Tag the label')
                continue

            if stf:
                silence_start, silence_end =getAllTime(actualStartTime, actualStartTime ,row[1])
                list_with_silence.append([wavfile, silence_start, silence_end, 'Silence'])
                start, end = getAllTime(actualStartTime, row[1], row[2])
                list_with_silence.append([wavfile, start, end, label])
                prev = row
                stf = False
                continue

            silence_start, silence_end =getAllTime(actualStartTime, prev[2],row[1])
            list_with_silence.append([wavfile, silence_start, silence_end, 'Silence'])
            start, end = getAllTime(actualStartTime, row[1], row[2])
            list_with_silence.append([wavfile, start, end, label])
        prev = row
    start, end = getAllTime(actualStartTime, row[2], row[2])
    list_with_silence.append([wavfile, start, endwav, 'Silence'])
    return pd.DataFrame(list_with_silence, columns=['filename', 'onset', 'offset', 'event_label'])

def get_metadataDF(wavFile):
    duration_seconds = librosa.get_duration(filename=wavFile)
    return pd.DataFrame([[wavFile.split('/')[-1], int(duration_seconds)]], columns=['filename', 'duration'])

def get_predictDF(pred_filename, wavfile):
    events =  pd.read_csv(pred_filename, usecols=[2, 3, 6])
    events['audio_start'] = events['audio_start'].apply(convert_to_seconds)
    events['audio_end'] = events['audio_end'].apply(convert_to_seconds)

    events['filename'] = pd.Series([wavfile.split('/')[-1]]*len(events))
    events = events.rename(columns={'audio_end': 'offset', 'audio_start': 'onset', 'label': 'event_label'})
    return events



def getMetrics(pred_filename, groundtruth, wavFile, event_index):
    dtc_threshold = 0.5
    gtc_threshold = 0.5
    cttc_threshold = 0.3
    alpha_ct = 0.0
    alpha_st = 0.0
    max_efpr = 100
    #event_index=11
    #pred_filename = '/home/khan_mo/kratos/thesis/important Git Lib/ast/src/predict/predictCSV/pred_133-0020_201031_231556_indoors-tagSeconds-30_events.csv'
    #groundtruth = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_single_Events_results_FLuiD.txt'
    #wavFile = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_231556_indoors.wav'
    startTime = get_start_time(wavFile)
    gt_table = getgroundtruthDF(groundtruth, startTime, wavFile, event_index)

    meta_table = get_metadataDF(wavFile)

    predictionDF = get_predictDF(pred_filename, wavFile)

    loglist = []
    # Instantiate PSDSEval
    psds_eval = PSDSEval(dtc_threshold, gtc_threshold, cttc_threshold,
                         ground_truth=gt_table, metadata=meta_table)
    info = {"name": f"Op {1}", "threshold": 0.1}
    psds_eval.add_operating_point(predictionDF, info=info)

    # Calculate the PSD-Score
    psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
    loglist.append(f"PSD-Score: {psds.value:.5f}\n")
    
    class_constraints = list()
    for cls in psds_eval.class_names[:-1]:
        # find the op. point with maximum f1-score, value field is ignored for
        # F1-score criteria
        class_constraints.append({"class_name": cls,
                                  "constraint": "fscore",
                                  "value": None})
    class_constraints_table = pd.DataFrame(class_constraints)
    selected_ops = psds_eval.select_operating_points_per_class(
        class_constraints_table, alpha_ct=0., beta=1.)
    
    
    for k in range(len(class_constraints)):
        loglist.append(f"For class ''{class_constraints_table.class_name[k]}'', the highest "
              f"F1-score is achieved at:")
        loglist.append(f"\tProbability Threshold: {selected_ops.threshold[k]}, "
              f"TPR: {selected_ops.TPR[k] * 100:.2f}, "
              f"FPR: {selected_ops.FPR[k] * 100:.2f}, "
              f"eFPR: {selected_ops.eFPR[k] * 100:.2f}, "
              f"F1-score: {selected_ops.Fscore[k] * 100:.2f}")
        
    return loglist