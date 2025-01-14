import os

import json
import librosa
import numpy as np

DATA_PATH = "jim2012Chords/Guitar_Only" # Directory name of the file containing the .wav files
JSON_PATH = "data.json" # Filename where the encoded data is written

SAMPLE_RATE = 44100
HOP_LENGTH = 512

def generate_features(data_path, json_path, hop_length = HOP_LENGTH):
  """
  generate_features(data_path, json_path, n_mfcc, n_fft, hop_length, num_segment) 
  generates a dictionary containing the MFCC and associated labels for the .wav
  files contained in sub-folders within the data_path folder, and writes all the
  collected informations in the json file denoted by json_path. The MFCC is generated
  with arguments n_mfcc, n_fft, hop_length, where each 30s .wav file is split into
  num_segment segments.
  """

  # Dictionary data-type to encode information extracted from .wav file
  data = {
    "mapping": [],
    "chroma_vector": [],
    "labels": []
  }

  # Looping through all genres
  for i, (dir_path, _, file_name) in enumerate(os.walk(data_path)):

    # Saving genre label from directory path
    if dir_path is not data_path:
      path_components = dir_path.split("/")
      chord = path_components[-1]
      data["mapping"].append(chord)

      print("Processing {}\n".format(chord))
    
    for f in file_name:
      # loading an individual file
      if f == '.DS_Store':
        continue
      file_path = os.path.join(dir_path, f)
      signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

      chroma = librosa.feature.chroma_stft(y = signal, sr = sample_rate, hop_length = hop_length)
      chroma = chroma.T

      data['chroma_vector'].append(chroma.tolist())
      data['labels'].append(i - 1)

  # Accessing the json file for writing collected data
  with open(json_path, "w") as fp:
    json.dump(data, fp, indent = 4)

# Initiating the data collection process
if __name__ == "__main__":
  generate_features(DATA_PATH, JSON_PATH, HOP_LENGTH)