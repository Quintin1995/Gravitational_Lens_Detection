import numpy as np
import matplotlib.pyplot as plt
import math

# Count the number of true positive predictions
def count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, threshold, beta_squarred):
    TP = 0 #true positive
    TN = 0 #true negative
    FP = 0 #false positive
    FN = 0 #false negative

    for idx, pred in enumerate(prediction_vector):
        if pred >= threshold and y_test[idx] >= threshold:
            TP += 1
        if pred < threshold and y_test[idx] < threshold:
            TN += 1
        if pred >= threshold and y_test[idx] < threshold:
            FP += 1
        if pred < threshold and y_test[idx] >= threshold:
            FN += 1

    tot_count = TP + TN + FP + FN
    
    precision = TP/(TP + FP) if TP + FP != 0 else 0
    recall    = TP/(TP + FN) if TP + FN != 0 else 0
    fp_rate   = FP/(FP + TN) if FP + TN != 0 else 0
    accuracy  = (TP + TN) / len(prediction_vector) if len(prediction_vector) != 0 else 0

    F_beta    = (1+beta_squarred) * ((precision * recall) / ((beta_squarred * precision) + recall)) if ((beta_squarred * precision) + recall) else 0

    if tot_count != len(prediction_vector):
        print("Total count {} of (TP, TN, FP, FN) is not equal to the length of the prediction vector: {}".format(tot_count, len(prediction_vector)))

    print("Total Count {}\n\tTP: {}, TN: {}, FP: {}, FN: {}".format(tot_count, TP, TN, FP, FN))
    
    print("precision = {}".format(precision))
    print("recall    = {}".format(recall))
    print("fp_rate   = {}".format(fp_rate))
    print("accuracy  = {}".format(accuracy))
    print("F beta    = {}".format(F_beta))

    return TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta

############### begin script ###############
beta_squarred = 0.001
labels   = np.array([1.0 , 0.0   , 0.0     ,1.0    ,0.0  ,1.0  ,1.0 ,1.0,0.0,1.0,])
pred_vec = np.array([0.87, 0.0558, 0.000584,0.92587,0.657,0.765,0.99,0.1,0.1,0.98])

stepsize = 0.01
threshold_range = np.arange(0.0,1.0+stepsize,stepsize)

f_betas = []
for p_threshold in threshold_range:
    (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(pred_vec, labels, p_threshold, beta_squarred)
    f_betas.append(F_beta)

threshold_range = list(threshold_range)

plt.plot(threshold_range, f_betas)
plt.xlabel("p threshold")
plt.ylabel("F")
plt.title("Beta = {}".format(math.sqrt(beta_squarred)))
plt.show()