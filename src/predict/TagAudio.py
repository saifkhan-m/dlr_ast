import datetime
import ntpath
from Model_prediction import Model_prediction
import soundfile as sf
import csv
import argparse
import torch
import torchaudio
from tqdm import tqdm
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
dateTimeObj = datetime.datetime.now()
timestampStr = dateTimeObj.strftime("%d%b%H%M")

te_data='../egs/dlr/data/datafiles/dlr_eval_data_all.json'
tr_data='../egs/dlr/data/datafiles/dlr_train_data_all.json'
args.n_class=4
args.label_csv= '../egs/dlr/data/dlr_class_label_all.csv'

norm_stats = [-5.4024234, 4.9392357]
args.norm_mean = norm_stats[0]
args.norm_std=  norm_stats[1]
args.model='ast'
args.dataset='dlr'
args.imagenet_pretrain=True
args.audioset_pretrain=True
args.bal=None
if args.audioset_pretrain == True:
  args.lr=1e-5
else:
  args.lr=1e-4

args.freqm=24
args.timem=96
args.mixup=0
args.n_epochs=10
args.batch_size = 4
args.fstride=10
args.tstride=10
base_exp_dir = f'../egs/dlr/exp/test-{args.dataset}-f{args.fstride}-t{args.tstride}-p-b{args.batch_size}-lr{args.lr}-{timestampStr}'
args.save_model =True
args.exp_dir= base_exp_dir+'fold'
args.data_val = te_data
args.data_train = tr_data
#target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128, 'dlr':3072}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path='../../egs/dlr/exp/test-dlr-f10-t10-p-b4-lr1e-05-18Sep0933fold/models/best_audio_model.pth'

#args.dlr_target=1536
args.dlr_target=3072
class TagAudio:
    def __init__(self, longAudioPath, model_path,args):
        self.path = longAudioPath
        self.args=args
        self.waveform, self.sr = self.getWaveform_SR(self.path)
        self.modelPrediction= Model_prediction(model_path, args)
        
    def getWaveform_SR(self, longAudioPath):
        waveform, sr= torchaudio.load(longAudioPath)
        req_sr=16000
        waveform = torchaudio.transforms.Resample(sr, req_sr)(waveform[0, :].view(1, -1))
        return waveform, req_sr

    def tag(self, tagSeconds):
        predictions=[]
        count=0
        prev=None
        for i in tqdm(range(0, len(self.waveform[0]), self.sr*tagSeconds)):
            prediction=self.modelPrediction.predict_example(self.waveform[:,i:i+self.sr*tagSeconds], self.sr, melbins=128)
            time=str(datetime.timedelta(seconds=i/self.sr))
            timeend = str(datetime.timedelta(seconds=(i / self.sr)+tagSeconds))

            predictions.append([ time, timeend,prediction])
            count+=1

            prev=prediction

        filename=self.path.split('/')[-1].split('.')[0]
        with open("predictCSV/pred_"+filename+'-tagSeconds-'+str(tagSeconds)+".csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerow(['timestart','timeend', 'cat'])
            csvWriter.writerows(predictions)
        print(f'pred_{filename}-tagSeconds-{str(tagSeconds)}.csv" written')

#sf.write(processedPath, sig, samplerate)
#model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-f10-t10-p-b4-lr1e-05-26Sep1337fold/models/best_audio_model.pth'
model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered15-30Sep0024fold/models/best_audio_model.pth'
model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered-30Sep0027fold/models/best_audio_model.pth'
tagaud=TagAudio('/home/projects/SocialMediaAnalysis/audioDLR/full_audio/087-0291_160805_231502_indoors.wav',
                model_path,
                args)
tagaud.tag(30)