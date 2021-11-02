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


class TagAudio:
    def __init__(self, longAudioPath, model_path, args, event_result_path=None):
        self.path = longAudioPath
        self.args=args
        self.waveform, self.sr = self.getWaveform_SR(self.path)
        self.modelPrediction= Model_prediction(model_path, args)
        self.startTime = self.get_start_time(longAudioPath)
        self.event_result_path=event_result_path
        if self.event_result_path is not None:
            self.event_df = self.getEventDF(self.event_result_path)
    
    def getEventDF(self, event_result_path):
        df = pd.read_csv(event_result_path
                         , sep='\t'
                         # , names=['i1','start_time', 'end_time','LASmax_dt-Maximalpegel(innen) [dB(A)]','LASmax_dt-Pegelanstieg(maximal,innen) [dB(A)/s]','LASmax_dt-Leq3-Geraeusch(innen) [dB(A)]','LASmax_dt-SEL(innen) [dB(A)]','LASmax_dt-Leq3_1min(innen) [dB(A)]','LASmax_dt- SNR','LASmax_dt- MNR','Kommentar 1','Kommentar 2']
                         , usecols=[0, 1, 2, 11]
                         , skiprows=[0, -2, -1]
                         , header=None
                         )
        df = df.rename(columns={x: y for x, y in zip(df.columns, range(0, len(df.columns)))})
        return df[:-1]
    
    def getWaveform_SR(self, longAudioPath):
        waveform, sr= torchaudio.load(longAudioPath)
        req_sr=16000
        waveform = torchaudio.transforms.Resample(sr, req_sr)(waveform[0, :].view(1, -1))
        return waveform, req_sr

    def tag(self, tagSeconds, windowing=False):
        predictions=[]
        count=0
        prev=None
        predicted_events= []
        event_counter=0
        timWindowCount = 0
        for i in tqdm(range(0, len(self.waveform[0]), self.sr*tagSeconds)):
            prediction=self.modelPrediction.predict_example(self.waveform[:,i:i+self.sr*tagSeconds], self.sr, melbins=128)
            predictions.append(prediction)
            time=timedelta(seconds=i/self.sr)
            timeend = timedelta(seconds=(i / self.sr)+tagSeconds)

            if timWindowCount ==0:

                actual_start, audio_start = self.get_actual_audio_time(time)
                
            if prev is not None and prev !=prediction:
                
                actual_end, audio_end = self.get_actual_audio_time(timeend)
                label= cls_labels[prev]
                predicted_event = [actual_start, actual_end, audio_start, audio_end, prediction, label]
                if self.event_result_path is not None:
                    if self.checkIfinBetween(actual_start, actual_end, self.event_df.iloc[event_counter][1]):
                        actual_event, counter=self.getAllTime(actual_start,actual_end, event_counter)
                        predicted_event.extend(actual_event)
                        event_counter+=counter
                
                predicted_events.append(predicted_event)
                timWindowCount=0
                prev = prediction
                count += 1
                continue

            #predictions.append([ time, timeend,prediction])
            count+=1
            timWindowCount+=1
            #prev=prediction

        if windowing:
            predicted_events = self.do_windowing(predicted_events)

        filename=self.path.split('/')[-1].split('.')[0]
        # with open("predictCSV/pred_"+filename+'-tagSeconds-'+str(tagSeconds)+".csv", "w+") as my_csv:
        #     csvWriter = csv.writer(my_csv, delimiter=',')
        #     csvWriter.writerow(['timestart','timeend', 'cat'])
        #     csvWriter.writerows(predictions)
        with open("predictCSV/pred_"+filename+'-tagSeconds-'+str(tagSeconds)+"_events.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            if self.event_result_path is not None:
                csvWriter.writerow(['actual_start', 'actual_end', 'audio_start', 'audio_end', 'prediction', 'label', 'event_start', 'event_end', 'event_label'])
            else:
                csvWriter.writerow(['actual_start', 'actual_end', 'audio_start', 'audio_end', 'prediction', 'label'])
            csvWriter.writerows(predicted_events)
        print(f'pred_{filename}-tagSeconds-{str(tagSeconds)}_events.csv" written')

    def do_windowing(self, predicted_events):
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
    model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered15-30Sep0024fold/models/best_audio_model.pth'
    model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered-30Sep0027fold/models/best_audio_model.pth'
    model_path = '/home/khan_mo/thesis/models_pred/30sec30sep/best_audio_model.pth'

    audio_file = '/home/projects/SocialMediaAnalysis/audioDLR/full_audio/133-0020_201031_231556_indoors.wav'
    audio_file = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_231556_indoors.wav'
    event_file = '/home/khan_mo/thesis/models_pred/full_audio/133-0020_201031_single_Events_results_FLuiD.txt'

    arguments = get_args()
    tagaud=TagAudio(audio_file,
                    model_path,
                    arguments,
                    event_file)
    tagaud.tag(30)