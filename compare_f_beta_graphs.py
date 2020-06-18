import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

root_dir_models = "models"

path_model_1 = os.path.join(root_dir_models, "resnet_single_newtr_last_last_weights_only/")
path_model_2 = os.path.join(root_dir_models, "14_06_2020_14h_15m_44s_2000chunks/")
path_model_3 = os.path.join(root_dir_models, "08_06_2020_16h_11m_59s_approx1100chunks/")

f_beta_results_file_1_csv_list = glob.glob(path_model_1 + "f_beta_results.csv")
f_beta_results_file_2_csv_list = glob.glob(path_model_2 + "f_beta_results.csv")
f_beta_results_file_3_csv_list = glob.glob(path_model_3 + "f_beta_results.csv")

if len(f_beta_results_file_1_csv_list) != 0 and len(f_beta_results_file_2_csv_list):
    f_beta_results_file_1_csv = f_beta_results_file_1_csv_list[0]
    f_beta_results_file_2_csv = f_beta_results_file_2_csv_list[0]
    f_beta_results_file_3_csv = f_beta_results_file_3_csv_list[0]
else:
    print("f_beta scores must be calculated first by means of running calc_f_beta_score.py")

dat1 = pd.read_csv(f_beta_results_file_1_csv)
dat2 = pd.read_csv(f_beta_results_file_2_csv)
dat3 = pd.read_csv(f_beta_results_file_3_csv)

axis1, = plt.plot(dat1['p_threshold'], dat1['f_beta'], label="Enrico Resnet")
axis2, = plt.plot(dat2['p_threshold'], dat2['f_beta'], label="2000 chunks")
axis3, = plt.plot(dat3['p_threshold'], dat3['f_beta'], label="1100 chunks")
plt.legend(handles=[axis1, axis2, axis3])
plt.xlabel("p_threshold")
plt.ylabel("F")
plt.title("F beta scores")
plt.show()
