from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Flatten,
    Reshape,
    Input,
    Activation,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    GaussianNoise,
    MaxPooling1D,
    Lambda,
    Concatenate,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers, metrics
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    LearningRateScheduler,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import resnet
import matplotlib.pyplot as plt
import time, os, glob, sys, datetime
import numpy as np
import load_data
import augmentation as ra
import argparse
import pickle
from datetime import datetime
import json
from parameters import Parameters
from utils import *
import csv
import matplotlib.pyplot as plt
# import tensorflow as tf

# print(tf.config.list_physical_devices('GPU'))

# load the settings dictionary in order to start a run.
settings = load_run_yaml("runs/run.yaml")

# load all the parameters from settings dictionary into a parameter class.
params = Parameters(settings)

ra.augmentation_setup(params)



############# functions #####################
def main(
    model="resnet",
    mode=params.mode,
    num_chunks=params.num_chunks,
    chunk_size=params.chunk_size,
    input_sizes=params.input_sizes,
    batch_size=params.batch_size,
    nbands=params.nbands,
    model_name=params.model_name,
):
    #create a model (neural network)
    multi_model = call_model(params, model=model)
    print("Model loaded: {}".format(model), flush=True)
    
    if mode == "train":
        #create a csv logger that will store the history of the .fit function into a .csv file
        with open(params.full_path_of_history, 'w', newline='') as history_file:
            writer = csv.writer(history_file)
            writer.writerow(["chunk", "loss", "binary_accuracy"])

            loss = optimizers.Adam(lr=params.learning_rate)
            multi_model.compile(
                optimizer=loss,
                loss="binary_crossentropy",
                metrics=[metrics.binary_accuracy],
            )

            if nbands == 3:
                augmented_data_gen_pos = ra.realtime_augmented_data_gen_pos_col(
                    params = params,
                    num_chunks=params.num_chunks,
                    chunk_size=chunk_size,
                    target_sizes=input_sizes,
                    augmentation_params=params.default_augmentation_params,
                )
                augmented_data_gen_neg = ra.realtime_augmented_data_gen_neg_col(
                    params = params,
                    num_chunks=params.num_chunks,
                    chunk_size=chunk_size,
                    target_sizes=input_sizes,
                    augmentation_params=params.default_augmentation_params,
                )

            else:
                augmented_data_gen_pos = ra.realtime_augmented_data_gen_pos(
                    params = params,
                    range_min=params.range_min,
                    range_max=params.range_max,
                    num_chunks=params.num_chunks,
                    chunk_size=chunk_size,
                    target_sizes=input_sizes,
                    normalize=params.normalize,
                    resize=params.resize,
                    augmentation_params=params.default_augmentation_params,
                )
                augmented_data_gen_neg = ra.realtime_augmented_data_gen_neg(
                    params = params,
                    num_chunks=params.num_chunks,
                    chunk_size=chunk_size,
                    target_sizes=input_sizes,
                    normalize=params.normalize,
                    resize=params.resize,
                    augmentation_params=params.default_augmentation_params,
                )

            train_gen_neg = load_data.buffered_gen_mp(
                augmented_data_gen_neg, buffer_size=params.buffer_size
            )
            train_gen_pos = load_data.buffered_gen_mp(
                augmented_data_gen_pos, buffer_size=params.buffer_size
            )

            loss_per_chunk = []
            bin_acc_per_chunk = []
            actual_begin_time = time.time()
            try:
                for chunk in range(params.num_chunks):
                    start_time = time.time()
                    chunk_data_pos, chunk_length = next(train_gen_pos)
                    y_train_pos = chunk_data_pos.pop()
                    X_train_pos = chunk_data_pos

                    chunk_data_neg, _ = next(train_gen_neg)
                    y_train_neg = chunk_data_neg.pop()
                    X_train_neg = chunk_data_neg

                    if False:               #just to view some images positive or negative
                        imgs = chunk_data_pos[0]
                        imgs = np.squeeze(imgs)
                        for img in imgs:
                            plt.imshow(img/255.0)
                            plt.show()
                    

                    X_train = np.concatenate((X_train_pos[0], X_train_neg[0]))
                    y_train = np.concatenate((y_train_pos, y_train_neg))
                    y_train = y_train.astype(np.int32)
                    y_train = np.expand_dims(y_train, axis=1)
                    batches = 0
                    start_chunk_processing_time = time.time()
                    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                        X_batch, y_batch = batch

                        history = multi_model.fit(X_batch / 255.0 - params.avg_img, y_batch)
                        batches += 1
                    print("Chunck neural net time: {0:.3f} seconds".format(time.time() - start_chunk_processing_time), flush=True)
                    #write results to csv for later use
                    writer.writerow([str(chunk), str(history.history["loss"][0]), str(history.history["binary_accuracy"][0])])
                    
                    #store loss and accuracy in list
                    loss_per_chunk.append(history.history["loss"][0])
                    bin_acc_per_chunk.append(history.history["binary_accuracy"][0])

                    # plot loss and accuracy on interval
                    if chunk % params.chunk_plot_interval == 0:
                        save_loss_and_acc_figure(loss_per_chunk, bin_acc_per_chunk, params)

                    #empty the train data
                    X_train = None
                    y_train = None
                    print("Chunck {}/{} has been trained".format(chunk+1, params.num_chunks), flush=True)
                   

            except KeyboardInterrupt:
                multi_model.save_weights(params.full_path_of_weights)
                print("interrupted by KEYBOARD!", flush=True)
                print("saved weights to: {}".format(params.full_path_of_weights), flush=True)
            end_time = time.time()

            multi_model.save_weights(params.full_path_of_weights)
            print("\nSaved weights to: {}".format(params.full_path_of_weights), flush=True)
            print("\nSaved results to: {}".format(params.full_path_of_history), flush=True)
            final_time = end_time - actual_begin_time
            print("\nTotal time employed ", load_data.hms( final_time), flush=True)

    if mode == "predict":
        if nbands == 3:
            augmented_data_gen_test_fixed = ra.realtime_fixed_augmented_data_test_col(params = params, target_sizes=input_sizes)  # ,normalize=normalize)
        else:
            augmented_data_gen_test_fixed = ra.realtime_fixed_augmented_data_test(params = params, target_sizes=input_sizes)

        #load a trained model
        multi_model.load_weights(params.full_path_predict_weights)

        predictions = []
        test_batches = 0
        if params.augm_pred == True:            #seems to be a boolean to control whether you want the test data to be augmented or not when performing a prediction on test data.
            start_time = time.time()
            for e, (chunk_data_test, chunk_length_test) in enumerate(augmented_data_gen_test_fixed):
                X_test = chunk_data_test
                X_test = X_test[0]
                X_test = X_test / 255.0 - params.avg_img
                pred1 = multi_model.predict(X_test)
                pred2 = multi_model.predict(np.array([np.flipud(image) for image in X_test]))
                pred3 = multi_model.predict(np.array([np.fliplr(np.flipud(image)) for image in X_test]))
                pred4 = multi_model.predict(np.array([np.fliplr(image) for image in X_test]))
                preds = np.mean([pred1, pred2, pred3, pred4], axis=0)
                preds = preds.tolist()
                predictions = predictions + preds
                print("done with predict chunk: {}".format(e), flush=True)
        else:
            for e, (chunk_data_test, chunk_length_test) in enumerate(
                augmented_data_gen_test_fixed
            ):
                X_test = chunk_data_test
                X_test = X_test[0]
                X_test = X_test / 255.0 - params.avg_img
                pred1 = multi_model.predict(X_test)
                preds = pred1.tolist()
                predictions = predictions + preds

        with open("pred_" + params.model_name + ".pkl", "wb") as f:
            pickle.dump([[ra.test_data], [predictions]], f, pickle.HIGHEST_PROTOCOL)
        
        objects = []
        with (open("pred_" + params.model_name + ".pkl", "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break

        f = open("pred_my_model.csv", "w")
        x = str(objects[0])
        f.write(x)
        f.write("\n")
        f.close()

############### end functions #######################


if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs["model"] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs["mode"] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs["model_name"] = sys.argv[3]
    main(**kwargs)

