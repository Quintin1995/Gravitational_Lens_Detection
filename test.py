import glob
import os
import json


model_folder = "29_05_2020_14h_46m_02s_500chunks/"
model_folder = os.path.join("models", model_folder)

h5_file = glob.glob(model_folder + "*.h5")[0]
param_dump_file = glob.glob(model_folder + "*.json")[0]
with open(param_dump_file, 'r') as f:
    distros_dict = json.load(f)

print(h5_file)
x=23