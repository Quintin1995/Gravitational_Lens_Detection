TEST_DIC_PATH_SOURCES = "data/train_dic_sources.p"
TEST_DIC_PATH_LENSES = "data/train_dic_lenses.p"
TEST_DIC_PATH_NEG = "data/train_dic_neg.p"

import numpy as np
import glob
import os
import pickle
import random

# -----------------sources-----------------------------


path_source_folders = "data/training/sources/"
len_folders = len(glob.glob(path_source_folders + "*"))
path_sources = [
    path_source_folders + str(folder + 1) + "/" + str(folder + 1) + ".fits"
    for folder in range(len_folders)
]
source_names = [
    str(folder + 1) + "/" + str(folder + 1) + ".fits" for folder in range(len_folders)
]

cutout_dict = {}
for key, sourcename in zip(list(range(len(source_names))), source_names):
    cutout_dict[key] = {}
    if "lens" in sourcename:
        cutout_dict[key]["name"] = sourcename
    else:
        cutout_dict[key]["name"] = sourcename

d2 = dict(list(cutout_dict.items()))
pickle.dump(d2, open(TEST_DIC_PATH_SOURCES, "wb"))

# ----------lenses--------------------------

lenslist = []
fitsfiles = sorted(glob.glob("data/training/lenses/*_r_*.fits"))
sourceNames = set(fitsfiles)


for source in sourceNames:
    lenslist.append(source)

cutout_dict = {}
for key, sourcename in zip(list(range(len(lenslist))), lenslist):
    cutout_dict[key] = {}
    cutout_dict[key]["name"] = sourcename

d1 = dict(list(cutout_dict.items()))
pickle.dump(d1, open(TEST_DIC_PATH_LENSES, "wb"))


# ----------neg--------------------------

lenslist = []
fitsfiles = sorted(glob.glob("data/training/negatives/*_r_*.fits"))
sourceNames = set(fitsfiles)

for source in sourceNames:
    lenslist.append(source)

cutout_dict = {}
for key, sourcename in zip(list(range(len(lenslist))), lenslist):
    cutout_dict[key] = {}
    cutout_dict[key]["name"] = sourcename

d1 = dict(list(cutout_dict.items()))
pickle.dump(d1, open(TEST_DIC_PATH_NEG, "wb"))
