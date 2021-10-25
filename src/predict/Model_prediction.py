import argparse
from datetime import datetime
import os
import ast
import pickle
import sys
import time
import torch
import torchaudio
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src import models
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model_prediction:
    def __init__(self, model_path, args):
        self.args=args
        self.target_len_ip = args.dlr_target
        self.model= self.get_model(model_path, args)


    def get_model(self,path, args):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(path, map_location=device)

        audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride,
                                      input_fdim=128,
                                      input_tdim=self.target_len_ip, imagenet_pretrain=args.imagenet_pretrain,
                                      audioset_pretrain=args.audioset_pretrain, model_size='base384', path=None)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd)
        audio_model.eval()
        audio_model.to(device)
        print("model loaded")
        return audio_model

    def _wav2fbank(self, waveform, sr, melbins ):

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                      window_type='hanning', num_mel_bins=melbins, dither=0.0,
                                                      frame_shift=10)
        except AssertionError:
            print('Wav not loaded -------****************')
            fbank = None
        target_length = self.target_len_ip
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]


        return fbank

    def process_example(self,waveform, sr,melbins=128):

        waveform = waveform - waveform.mean()
        fbank=self._wav2fbank( waveform, sr, melbins)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.args.freqm)
        timem = torchaudio.transforms.TimeMasking(self.args.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.args.freqm != 0:
            fbank = freqm(fbank)
        if self.args.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = (fbank - self.args.norm_mean) / (self.args.norm_std * 2)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank
    def getWaveform(self, file):
        waveform, sr = torchaudio.load(file)
        return waveform,sr

    def predict_example(self, waveform, sr, melbins=128, fromFile=None):
        if fromFile is not None:
            waveform, sr =  self.getWaveform( fromFile)

        audio_input = self.process_example(waveform, sr, melbins)
        audio_input = audio_input.unsqueeze(0)
        audio_input = audio_input.to(device)
        with torch.no_grad():
            audio_output = self.model(audio_input)
        audio_output = torch.sigmoid(audio_output)
        predictions = audio_output.to('cpu').detach()
        return np.argmax(predictions, 1).tolist()[0]

if __name__ =="__main__":
    predict_file='/home/khan_mo/kratos/thesis/important Git Lib/ast/egs/dlr/data/dlrdata/audio16k/Flugzeug/8_split_23588_Flugzeug.wav'
    mp = Model_prediction(path, args)
    print(mp.predict_example( None,16000, melbins=128, fromFile=predict_file))
    print(mp.predict_example( None,16000, melbins=128, fromFile=predict_file))
    print(mp.predict_example( None,16000, melbins=128, fromFile=predict_file))
    print(mp.predict_example( None,16000, melbins=128, fromFile=predict_file))
