from Model_prediction import Model_prediction
import sys
#sys.path.append("../src")
from src.utilities.util import get_args
from TagAudio import TagAudio

def get_model(tag_seconds, where='/home/projects/SocialMediaAnalysis/audioDLR/models_pred/models/'):
    return where+f'best_audio_model_seconds_{tag_seconds}.pth'

def get_audio_event(dataset, where='/home/projects/SocialMediaAnalysis/audioDLR/models_pred/full_audio/'):
    if dataset=='new':
        audio_file = where + '133-0020_201031_231556_indoors.wav'
        event_file = where + '133-0020_201031_single_Events_results_FLuiD.txt'
        return audio_file, event_file
    elif dataset== "veu":
        audio_file = where + '087-0291_160805_231502_indoors.wav'
        event_file = where + 'VP087-0291_160805_single_Events_results.txt'
        return audio_file, event_file
    else :
        return None
overlap_flag=True
datasets = ['new','veu']
for dataset in datasets:
    if dataset=='new':
        column_num = 11#11 #36
    if dataset=='veu':
        column_num = 36#11 #36

    tag_seconds_arr= [5,10,15, 30]
    for tag_seconds in tag_seconds_arr:
        if overlap_flag and (tag_seconds==5 or tag_seconds==10):
            continue
        args= get_args(tag_seconds)
        args.column_number= column_num
        #model_path = get_model(tag_seconds, where='/home/khan_mo/thesis/models_pred/models/')
        model_path = get_model(tag_seconds)

        #audio_file, event_file = get_audio_event(dataset, where='/home/khan_mo/thesis/models_pred/full_audio/')
        audio_file, event_file = get_audio_event(dataset)
        print(audio_file)
        print(event_file)
        print(model_path)

        tagaud =TagAudio(audio_file,
                        model_path,
                        args,
                        event_result_path=event_file, overlap=overlap_flag)
        tagaud.tag(tag_seconds)