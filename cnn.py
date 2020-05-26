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


# load the settings dictionary in order to start a run.
settings = load_run_yaml("runs/run.yaml")

# load all the parameters from settings dictionary into a parameter class.
p = Parameters(settings)

############### extra parameters #########################
default_augmentation_params = {
    "zoom_range": (1 / 1.1, 1.0),
    "rotation_range": (0, 180),
    "shear_range": (0, 0),
    "translation_range": (-4, 4),
    "do_flip": True,
}

test_path = ra.test_path
test_data = ra.test_data





############# functions #####################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx : start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_resnet():
    model = resnet.ResnetBuilder.build_resnet_18(p.input_shape, 1)  # 18
    return model


def call_model(model="resnet"):
    if model == "resnet":
        multi_model = build_resnet()

    return multi_model


def main(
    model="resnet",
    mode="train",
    num_chunks=p.num_chunks,
    chunk_size=p.chunk_size,
    input_sizes=p.input_sizes,
    batch_size=p.batch_size,
    nbands=p.nbands,
    model_name=p.model_name,
):
    
    multi_model = call_model(model=model)

    if mode == "train":

        loss = optimizers.Adam(lr=p.learning_rate)

        multi_model.compile(
            optimizer=loss,
            loss="binary_crossentropy",
            metrics=[metrics.binary_accuracy],
        )

        if nbands == 3:
            augmented_data_gen_pos = ra.realtime_augmented_data_gen_pos_col(
                num_chunks=p.num_chunks,
                chunk_size=chunk_size,
                target_sizes=input_sizes,
                augmentation_params=default_augmentation_params,
            )
            augmented_data_gen_neg = ra.realtime_augmented_data_gen_neg_col(
                num_chunks=p.num_chunks,
                chunk_size=chunk_size,
                target_sizes=input_sizes,
                augmentation_params=default_augmentation_params,
            )

        else:
            augmented_data_gen_pos = ra.realtime_augmented_data_gen_pos(
                range_min=p.range_min,
                range_max=p.range_max,
                num_chunks=p.num_chunks,
                chunk_size=chunk_size,
                target_sizes=input_sizes,
                normalize=p.normalize,
                resize=p.resize,
                augmentation_params=default_augmentation_params,
            )
            augmented_data_gen_neg = ra.realtime_augmented_data_gen_neg(
                num_chunks=p.num_chunks,
                chunk_size=chunk_size,
                target_sizes=input_sizes,
                normalize=p.normalize,
                resize=p.resize,
                augmentation_params=default_augmentation_params,
            )

        train_gen_neg = load_data.buffered_gen_mp(
            augmented_data_gen_neg, buffer_size=p.buffer_size
        )
        train_gen_pos = load_data.buffered_gen_mp(
            augmented_data_gen_pos, buffer_size=p.buffer_size
        )
        actual_begin_time = time.time()
        try:
            for chunk in range(p.num_chunks):
                start_time = time.time()
                chunk_data_pos, chunk_length = next(train_gen_pos)
                y_train_pos = chunk_data_pos.pop()
                X_train_pos = chunk_data_pos

                chunk_data_neg, _ = next(train_gen_neg)
                y_train_neg = chunk_data_neg.pop()
                X_train_neg = chunk_data_neg

                X_train = np.concatenate((X_train_pos[0], X_train_neg[0]))
                y_train = np.concatenate((y_train_pos, y_train_neg))
                y_train = y_train.astype(np.int32)
                y_train = np.expand_dims(y_train, axis=1)
                train_batches = 0
                batches = 0
                for batch in iterate_minibatches(
                    X_train, y_train, batch_size, shuffle=True
                ):
                    X_batch, y_batch = batch
                    train = multi_model.fit(X_batch / 255.0 - p.avg_img, y_batch)
                    batches += 1
                X_train = None
                y_train = None
                print("Chunck {}/{} has been trained".format(chunk, p.num_chunks))
                print("Chunck took {0:.3f} seconds".format(time.time() - start_time))

        except KeyboardInterrupt:
            # multi_model.save(model_name+'_last.h5')
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
            multi_model.save_weights(
                model_name + "_weights_only{}.h5".format(dt_string)
            )
            print("interrupted!")
            print("saved weights")

        end_time = time.time()

        # multi_model.save(model_name+'_last.h5')
        multi_model.save_weights(model_name + "_weights_only.h5")
        print("time employed ", end_time - actual_begin_time)

    if mode == "predict":
        if nbands == 3:
            augmented_data_gen_test_fixed = ra.realtime_fixed_augmented_data_test_col(
                target_sizes=input_sizes
            )  # ,normalize=normalize)
        else:
            augmented_data_gen_test_fixed = ra.realtime_fixed_augmented_data_test(
                target_sizes=input_sizes
            )

        multi_model.load_weights(model_name + "_weights_only.h5")

        predictions = []
        test_batches = 0
        if p.augm_pred == True:
            start_time = time.time()
            for e, (chunk_data_test, chunk_length_test) in enumerate(
                augmented_data_gen_test_fixed
            ):
                X_test = chunk_data_test
                X_test = X_test[0]
                X_test = X_test / 255.0 - p.avg_img
                pred1 = multi_model.predict(X_test)
                pred2 = multi_model.predict(
                    np.array([np.flipud(image) for image in X_test])
                )
                pred3 = multi_model.predict(
                    np.array([np.fliplr(np.flipud(image)) for image in X_test])
                )
                pred4 = multi_model.predict(
                    np.array([np.fliplr(image) for image in X_test])
                )
                preds = np.mean([pred1, pred2, pred3, pred4], axis=0)
                preds = preds.tolist()
                predictions = predictions + preds

        else:
            for e, (chunk_data_test, chunk_length_test) in enumerate(
                augmented_data_gen_test_fixed
            ):
                X_test = chunk_data_test
                X_test = X_test[0]
                X_test = X_test / 255.0 - avg_img
                pred1 = multi_model.predict(X_test)
                preds = pred1.tolist()
                predictions = predictions + preds

        with open("pred_" + model_name + ".pkl", "wb") as f:
            pickle.dump([[ra.test_data], [predictions]], f, pickle.HIGHEST_PROTOCOL)

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

objects = []
with (open("pred_my_model.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

f = open("pred_my_model.csv", "w")
x = str(objects[0])
f.write(x)
f.close()

print("Dati salvati nel file pred_my_model.csv")
print("Translation:\nData saved in the file pred_my_model.csv")