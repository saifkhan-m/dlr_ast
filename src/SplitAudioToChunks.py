from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from pathlib import Path
from os import walk
from tqdm import tqdm
from utilities import *
# This file will create chunks based on the value of seconds variable. It will create chunks of the given value
home_folder= '/home/projects/SocialMediaAnalysis/audioDLR/final_data16k_copy'

seconds=15
dest_folder= '/home/projects/SocialMediaAnalysis/audioDLR/final_data16k'+str(seconds)+'chunks'
subfolders = [ f.path for f in os.scandir(home_folder) if f.is_dir() ]

for folder in subfolders:
    cat_folder = folder.split('/')[-1]
    class_heirarchy = get_heirarchy_of_categories()
    if not class_heirarchy[cat_folder] \
            or class_heirarchy[cat_folder][0] in ['Strassenverkehr', 'Zuege-Bahnen', 'Grosse Fahrzeuge']:
        # Condition to only takes Autos, Flugzeug, Silence and Nebengerausch
        continue
    cat= class_heirarchy[cat_folder][0]
    folder_path = os.path.join(dest_folder, cat)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    wavfiles = []
    for (dirpath, dirnames, filenames) in walk(folder):
        wavfiles.extend([os.path.join(dirpath, f).replace('\\', '/') for f in filenames if f.endswith('wav')])

    for file in tqdm(wavfiles):
        myaudio = AudioSegment.from_file(file, "wav")
        chunk_length_ms = seconds * 1000  # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
        filename = file.split('/')[-1].split('.')[0]
        # Export all of the individual chunks as wav files

        for i, chunk in enumerate(chunks):
            chunk_name = os.path.join(folder_path, filename + "_chunk{0}.wav".format(i))
            chunk.export(chunk_name, format="wav")