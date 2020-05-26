TRAIN_IDS_PATH_LENS = "data/train_ids_lens.npy"
TRAIN_IDS_PATH_SOURCE = "data/train_ids_source.npy"
TRAIN_IDS_PATH_NEG = "data/train_ids_neg.npy"

import numpy as np
import os
import csv
import pickle

# -------------LENS

train_ids = []
dic = "data/train_dic_lenses.p"
cutout_dic = pickle.load(open(dic, "rb"))
for i in cutout_dic:
    train_ids.append(i)

train_ids_lens = np.array(train_ids)
print("Saving %s" % TRAIN_IDS_PATH_LENS)
np.save(TRAIN_IDS_PATH_LENS, train_ids_lens)


# -------------SOURCE
train_ids = []
dic = "data/train_dic_sources.p"
cutout_dic = pickle.load(open(dic, "rb"))
for i in cutout_dic:
    train_ids.append(i)

train_ids_source = np.array(train_ids)
print("Saving %s" % TRAIN_IDS_PATH_SOURCE)
np.save(TRAIN_IDS_PATH_SOURCE, train_ids_source)

# -------------NEG
train_ids = []
dic = "data/train_dic_neg.p"
cutout_dic = pickle.load(open(dic, "rb"))
for i in cutout_dic:
    train_ids.append(i)

train_ids_neg = np.array(train_ids)
print("Saving %s" % TRAIN_IDS_PATH_NEG)
np.save(TRAIN_IDS_PATH_NEG, train_ids_neg)
