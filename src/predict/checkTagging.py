from Model_prediction import Model_prediction
import sys
#sys.path.append("../src")
from src.utilities.util import get_args


args= get_args()
from TagAudio import TagAudio

# model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-f10-t10-p-b4-lr1e-05-26Sep1337fold/models/best_audio_model.pth'
model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered15-30Sep0024fold/models/best_audio_model.pth'


model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered-30Sep0027fold/models/best_audio_model.pth'
audio_file = '/home/projects/SocialMediaAnalysis/audioDLR/full_audio/133-0020_201031_231556_indoors.wav'
event_file = '/home/projects/SocialMediaAnalysis/audioDLR/full_audio/133-0020_201031_single_Events_results_FLuiD.txt'
#model_path = '/home/khan_mo/thesis/models_pred/30sec30sep/best_audio_model.pth'
#audio_file = '/home/khan_mo/data/full_audio/087-0291_160805_231502_indoors.wav'

tagaud =TagAudio(audio_file,
                model_path,
                args,
                event_file)
tagaud.tag(30)