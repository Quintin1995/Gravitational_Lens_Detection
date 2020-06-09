import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

root_dir_models = "models"

path_model_1 = os.path.join(root_dir_models, "08_06_2020_16h_11m_59s_approx1100chunks/")
path_model_2 = os.path.join(root_dir_models, "07_06_2020_19h_33m_32s_500chunks_baseline/")

f_beta_results_file_1_csv = glob.glob(path_model_1 + "f_beta_results.csv")[0]
f_beta_results_file_2_csv = glob.glob(path_model_2 + "f_beta_results.csv")[0]

dat1 = pd.read_csv(f_beta_results_file_1_csv)
dat2 = pd.read_csv(f_beta_results_file_2_csv)

axis1, = plt.plot(dat1['p_threshold'], dat1['f_beta'], label="1100 chunks")
axis2, = plt.plot(dat2['p_threshold'], dat2['f_beta'], label="500  chunks")
plt.legend(handles=[axis1, axis2])
plt.xlabel("p_threshold")
plt.ylabel("F")
plt.title("F beta scores")
plt.show()
