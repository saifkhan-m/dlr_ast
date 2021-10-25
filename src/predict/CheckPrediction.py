from Model_prediction import Model_prediction
import sys
#sys.path.append("../src")
from src.utilities.util import get_args
import json
predict_file='/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/data/dlrdata/audio16k/Flugzeug/8_split_23588_Flugzeug.wav'
model_path = '/home/khan_mo/thesis/important Git Lib/ast/egs/dlr/exp/test-dlr-b4-e10-ered-30Sep0027fold/models/audio_model.10.pth'

dataset_json_file= "../../egs/dlr/data/datafiles/dlr_eval_data_red.json"
with open(dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']
check_list=[]
for i in range(0,len(data)):
    label = data[i]['labels']
    if label=='/m/21rwj00':
        check_list.append([data[i]['wav'],label])

args= get_args()
mp = Model_prediction(model_path, args)
count=0
for file, label in check_list:
    prediction = mp.predict_example( None,16000, melbins=128, fromFile=file)
    if prediction==int(label[-1]):
        count+=1
    print(f'output is : {prediction} for label {label}')
print(f'Total examples are{len(check_list)}')
print(f'accuracy is {count/len(check_list)}')