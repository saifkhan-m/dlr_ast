import os.path
from datetime import datetime
import ntpath
from datetime import timedelta
from Model_prediction import Model_prediction
import soundfile as sf
import csv
import argparse
import torch
import torchaudio
from tqdm import tqdm
import pandas as pd
import operator
from src.utilities.util import get_args
import itertools as it
from collections import Counter
import metrics

cls_labels=['Flugzeug', 'Silence', 'Nebengeraeusche', 'Autos']
class TagAudio:
    def __init__(self, longAudioPath, model_path, args, event_result_path=None, overlap=False,windowing=False):
        self.path = longAudioPath
        self.args=args
        self.waveform, self.sr = self.getWaveform_SR(self.path)
        self.modelPrediction= Model_prediction(model_path, args)
        self.startTime = self.get_start_time(longAudioPath)
        self.event_result_path=event_result_path
        self.overlap=overlap
        self.overlap_times = 3
        self.windowing=windowing
        if self.event_result_path is not None:
            self.event_df = self.getEventDF(self.event_result_path)
    
    def getEventDF(self, event_result_path):
        df = pd.read_csv(event_result_path
                         , encoding="ISO-8859-1"
                         , sep='\t'
                         # , names=['i1','start_time', 'end_time','LASmax_dt-Maximalpegel(innen) [dB(A)]','LASmax_dt-Pegelanstieg(maximal,innen) [dB(A)/s]','LASmax_dt-Leq3-Geraeusch(innen) [dB(A)]','LASmax_dt-SEL(innen) [dB(A)]','LASmax_dt-Leq3_1min(innen) [dB(A)]','LASmax_dt- SNR','LASmax_dt- MNR','Kommentar 1','Kommentar 2']
                         , usecols=[0, 1, 2, self.args.column_number]# 11 for new #36 for veu
                         , skiprows=[0, -2, -1]
                         , header=None
                         , quoting=csv.QUOTE_NONE
                         )
        df = df.rename(columns={x: y for x, y in zip(df.columns, range(0, len(df.columns)))})
        return df[:-1]
    
    def getWaveform_SR(self, longAudioPath):
        waveform, sr= torchaudio.load(longAudioPath)
        req_sr=16000
        waveform = torchaudio.transforms.Resample(sr, req_sr)(waveform[0, :].view(1, -1))
        return waveform, req_sr
    
    def  process_overlaped_predictions(self,overlaped_predictions):
        final_length=len(overlaped_predictions[0])
        for i in range(len(overlaped_predictions)):
            if len(overlaped_predictions[i])>final_length:
                overlaped_predictions[i]=overlaped_predictions[i][:final_length]
            elif len(overlaped_predictions[i])< final_length:
                last= final_length-len(overlaped_predictions[i])
                overlaped_predictions[i].extend(overlaped_predictions[0][-last:])

        for i in range(final_length):
            pred_times=[ overlaped_predictions[j][i][2] for j in range(self.overlap_times)]
            pred_times=[ ele for ele in pred_times if ele!=-1 ]

            pred_times = Counter(pred_times)
            final_prediction = pred_times.most_common(1)[0][0]
            overlaped_predictions[0][i][2] = final_prediction
        return overlaped_predictions[0]
                
    
    def tag_overlap(self,tagSeconds):
        overlaped_predictions= []
        for times in range(self.overlap_times):
            start = int(times*tagSeconds/3)
            starttflag=True
            predictions = []
            for i in tqdm(range(start*self.sr, len(self.waveform[0]), self.sr*tagSeconds)):
                prediction=self.modelPrediction.predict_example(self.waveform[:,i:i+self.sr*tagSeconds], self.sr, melbins=128)
                if starttflag:
                    starttflag=False
                    for sta in range(0, start, int(tagSeconds/3)):
                        time=timedelta(seconds=sta)
                        timeend = timedelta(seconds=(sta + (tagSeconds / 3)))
                        predictions.append([time, timeend, -1])
                for times1 in range(self.overlap_times):
                    time=timedelta(seconds=i/self.sr+(times1*tagSeconds/3))
                    timeend=timedelta(seconds=i/self.sr+((times1+1)*(tagSeconds/3)))
                    predictions.append([time, timeend, prediction])
            overlaped_predictions.append(predictions)
        overlaped_predictions= self.process_overlaped_predictions(overlaped_predictions)
        return overlaped_predictions

    def tag_normal(self,tagSeconds):
        predictions = []
        for i in tqdm(range(0, len(self.waveform[0]), self.sr*tagSeconds)):
            prediction=self.modelPrediction.predict_example(self.waveform[:,i:i+self.sr*tagSeconds], self.sr, melbins=128)

            time=timedelta(seconds=i/self.sr)
            timeend = timedelta(seconds=(i / self.sr)+tagSeconds)
            predictions.append([time, timeend, prediction])
        return predictions

    def process_predictions(self, predictions):
        predicted_events = []
        event_counter = 0
        predictions_list = [int(predictions[i][2]) for i in range(len(predictions))]
        prediction_ranges = self.find_ranges(predictions_list)
        for rnge in prediction_ranges:
            start_i, end_i = rnge[0], rnge[1]
            final_prediction = predictions[start_i][2]
            time, timeend = predictions[start_i][0], predictions[end_i][1],

            actual_start, audio_start = self.get_actual_audio_time(time)
            actual_end, audio_end = self.get_actual_audio_time(timeend)
            duration = self.getDuration(audio_end, audio_start)

            label = cls_labels[final_prediction]
            predicted_event = [actual_start, actual_end, audio_start, audio_end, duration, final_prediction, label]
            if self.event_result_path is not None:
                # print(actual_start, actual_end, event_counter)
                if self.checkIfinBetween(actual_start, actual_end, self.event_df.iloc[event_counter][1]):
                    actual_event, counter = self.getAllTime(actual_start, actual_end, event_counter)
                    predicted_event.extend(actual_event)
                    event_counter += counter

            predicted_events.append(predicted_event)
            if event_counter >= len(self.event_df):
                break
        return predicted_events

    def tag(self, tagSeconds):
        if self.overlap:
            #self.waveform[0] = self.waveform[0][self.sr * tagSeconds * 10]
            predictions= self.tag_overlap(tagSeconds)
        else:
            predictions= self.tag_normal(tagSeconds)

        predicted_events = self.process_predictions(predictions)

        filename = self.path.split('/')[-1].split('.')[0]
        filename = filename+'_ts'+ str(tagSeconds)
        plainPredictionFile = "PLAIN" + filename + ".csv"
        if self.windowing:
            predicted_events = self.do_windowing(predicted_events)
            filename = 'windowing_'  + filename
        if self.overlap:
            filename = 'overlap_' + filename
            plainPredictionFile = 'overlap_'+plainPredictionFile
       
        self.writePlain(plainPredictionFile, predictions)
        pred_filename = self.writePredFile(filename+ ".csv", predicted_events)
        print('pred_filename : ', pred_filename)
        if self.event_result_path is not None:
            metriclog = metrics.getMetrics(pred_filename, self.event_result_path, self.path, self.args.column_number)
            self.writeMetricLog(pred_filename ,metriclog)
    
    def writeMetricLog(self, filename,metriclog):
        filename = filename.split('.')[0]+'.txt'
        with open(filename, 'w') as f:
            f.write('\n'.join(metriclog))

    def writePlain(self, filename, predictions):
        filename = os.path.join('predictCSV/plain', filename)
        with open(filename, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerow(['timestart','timeend', 'cat'])
            csvWriter.writerows(predictions)
            
    def writePredFile(self,filename, predicted_events):
        filename = os.path.join('predictCSV/eventfiles', filename)
        with open(filename, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            if self.event_result_path is not None:
                csvWriter.writerow(['actual_start', 'actual_end', 'audio_start', 'audio_end', 'duration', 'prediction', 'label', 'event_start', 'event_end', 'event_label'])
            else:
                csvWriter.writerow(['actual_start', 'actual_end', 'audio_start', 'audio_end', 'duration', 'prediction', 'label'])
            csvWriter.writerows(predicted_events)
            print(f'{filename}" written')
        return filename
            
    def find_ranges(self, lst, n=1):
        """Return ranges for `n` or more repeated values.
        https://stackoverflow.com/questions/44790869/find-indexes-of-repeated-elements-in-an-array-python-numpy
        """
        groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
        repeated = (idx_g for k, idx_g in groups if len(idx_g) >= n)
        return [[sub[0][0], sub[-1][0]] for sub in repeated]

    def getDuration(self, audio_end, audio_start):
        FMT = '%H:%M:%S'
        return str(datetime.strptime(audio_end, FMT) - datetime.strptime(audio_start, FMT))
    
    def do_windowing_weighted(self, predicted_events):
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

    def do_windowing_weighted(self, predicted_events):
        pass
    def convert_to_time(self, date_strimg):
        FMT = '%H:%M:%S'
        return datetime.strptime(date_strimg, FMT).time()

    def getAllTime(self, actual_start, actual_end,counter):
        st, et, lb='', '', ''
        count=0
        for i, row in self.event_df[counter:].iterrows():
            if self.checkIfinBetween(actual_start, actual_end, row[1]):
                st += row[1]+'||'
                et += row[2]+'||'
                lb += row[3].strip()+'||'
                count+=1
            else:
                break
        return [st.strip('||'), et.strip('||'), lb.strip('||')], count

    def checkIfinBetween(self, begin_time, end_time, check_time):
        # If check time is not given, default to current UTC time
        begin_time, end_time, check_time = self.convert_to_time(begin_time), self.convert_to_time(end_time), self.convert_to_time(check_time)
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else: # crosses midnight
            return check_time >= begin_time or check_time <= end_time
        
    def get_actual_audio_time(self, time):
        #time in seconds to actual and audio time
        FMT = '%H:%M:%S'
        actual_time = datetime.strptime(self.startTime, FMT) + time
        actual_time= str(actual_time.time())
        audio_time=str(datetime.strptime(str(time), FMT).time())
        return actual_time, audio_time


    def get_start_time(self,filename):
        filename = filename.split("/")[-1]
        parentDir, datefolder, startTime, _ = filename.split("_")

        if len(startTime) != 6:
            raise ValueError("startime less than 6 digits")
        else:
            startTime = ':'.join(startTime[i:i + 2] for i in range(0, len(startTime), 2))
        return startTime

if __name__ =="__main__":
    #sf.write(processedPath, sig, samplerate)
    #model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-f10-t10-p-b4-lr1e-05-26Sep1337fold/models/best_audio_model.pth'
    model_path = '/home/khan_mo/kratos/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered15-30Sep0024fold/models/best_audio_model.pth'
    #model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered-30Sep0027fold/models/best_audio_model.pth'
    #model_path = '/home/khan_mo/thesis/models_pred/30sec30sep/best_audio_model.pth'
    model_path = '/home/khan_mo/thesis/models_pred/models/best_audio_model_seconds_15.pth'


    audio_file = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_231556_indoors.wav'
    event_file = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_single_Events_results_FLuiD.txt'

    #veu
    #audio_file = '/home/khan_mo/thesis/models_pred/full_audio/087-0291_160805_231502_indoors.wav'
    #event_file = '/home/khan_mo/thesis/models_pred/full_audio/VP087-0291_160805_single_Events_results.txt'
    audio_file='/home/khan_mo/thesis/thesis-khan/audio/dataset/audio/background/street/133-0020_201031_231556_indoorsstreet-sceene.wav'
    arguments = get_args(30)
    arguments.dlr_target=1580
    tagaud=TagAudio(audio_file,
                    model_path,
                    arguments,
                    event_result_path=None,
                    overlap=True)
    tagaud.tag(30)