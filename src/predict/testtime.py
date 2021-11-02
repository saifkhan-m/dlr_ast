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

with open("predictCSV/pred_133-0020_201031_231556_indoors-tagSeconds-30_events.csv", "r") as my_csv:
    csv_reader = list(csv.reader(my_csv, delimiter=','))
    do_windowing(csv_reader[1:])
