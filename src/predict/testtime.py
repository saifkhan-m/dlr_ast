from datetime import datetime
import ntpath
from datetime import timedelta
import pandas as pd
import csv
def get_dataframe(data_dir):
    df = pd.read_csv(data_dir
                     , sep='\t'
                     # , names=['i1','start_time', 'end_time','LASmax_dt-Maximalpegel(innen) [dB(A)]','LASmax_dt-Pegelanstieg(maximal,innen) [dB(A)/s]','LASmax_dt-Leq3-Geraeusch(innen) [dB(A)]','LASmax_dt-SEL(innen) [dB(A)]','LASmax_dt-Leq3_1min(innen) [dB(A)]','LASmax_dt- SNR','LASmax_dt- MNR','Kommentar 1','Kommentar 2']
                     , usecols=[0, 1, 2, 11]
                     , skiprows=[0, -2, -1]
                     , header=None
                     )
    return df[:-1]
data= '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_single_Events_results_FLuiD.txt'

cls_labels=['Flugzeug', 'Silence', 'Nebengeraeusche', 'Autos']
def do_windowing( predicted_events):
    predictions = [row[4] for row in predicted_events]
    window_predictions = [-1] * len(predicted_events)
    window_predictions[0], window_predictions[1] = predictions[0], predictions[1]
    mask = [0.3, 0.75, 1.0, 0.75, 0.3]
    for i in range(2, len(predictions) - 2):
        window = [predictions[i - 2], predictions[i - 1], predictions[i], predictions[i + 1], predictions[i + 2]]
        dct = {p: 0 for p in window}
        for w in range(len(window)):
            dct[window[w]] += mask[w]
        window_predictions[i] = max(dct, key=dct.get)
    window_predictions[-1], window_predictions[-2] = predictions[-1], predictions[-2]
    for it in range(len(predicted_events)):
        predicted_events[it][4] = window_predictions[it]
        predicted_events[it][5] = cls_labels[int(window_predictions[it])]
    return predicted_events

def test_window():
    with open("predictCSV/pred_133-0020_201031_231556_indoors-tagSeconds-30_events.csv", "r") as my_csv:
        csv_reader = list(csv.reader(my_csv, delimiter=','))
        do_windowing(csv_reader[1:])


def get_actual_audio_time( time):
    #time in seconds to actual and audio time
    FMT = '%H:%M:%S'
    actual_time= None
    audio_time=str(datetime.strptime(str(time), FMT).time())
    return actual_time, audio_time
def get_duration():
    time=timedelta(seconds=5)
    timeend = timedelta(seconds=150000)
    FMT = '%H:%M:%S'
    actual_start, audio_start = get_actual_audio_time(time)
    actual_end, audio_end = get_actual_audio_time(timeend)
    duration = datetime.strptime(audio_end, FMT)- datetime.strptime(audio_start, FMT)
    print(str(duration))

import itertools as it


def find_ranges(lst,n=1):
    """Return ranges for `n` or more repeated values.
    https://stackoverflow.com/questions/44790869/find-indexes-of-repeated-elements-in-an-array-python-numpy
    """
    groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if len(idx_g) >=n)
    return [[sub[0][0], sub[-1][0]] for sub in repeated]

lst = [34,2,3,22,22,22,22,22,22,18,90,5,-55,-19,22,6,6,6,6,6,6,6,6,23,53,1,5,-42,82]

#print(list(find_ranges(lst, 1)))
def frequency():
    
    from collections import Counter
    
    # initializing list
    test_list = [9, 4, 5]
    # using most_common to
    # get most frequent element
    test_list = Counter(test_list)
    res = test_list.most_common(1)[0][0]
    print(res)
