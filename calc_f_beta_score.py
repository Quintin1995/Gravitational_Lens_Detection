import glob
import numpy as np
import skimage
import load_data
import multiprocessing as mp
from PIL import Image
import resnet
import matplotlib.pyplot as plt
import os
import json
import math
import csv

### begin model to be loaded for prediction ###
model_folder = "18_06_2020_12h_15m_09s_5000chunks/"
model_folder = os.path.join("models", model_folder)
h5_file = glob.glob(model_folder + "*.h5")[0]

# file name and path of figure with f_beta score
full_path_fBeta_figure = os.path.join(model_folder, "f_beta_graph.png")

# file name and path of csv where f_beta results will be stored in csv format. This includes scores, such as TP, TN, FP, FN, accuracy, recall, precision.
f_beta_full_path = os.path.join(model_folder, "f_beta_results.csv")

this_model_has_param_file = True

if this_model_has_param_file:
    param_dump_filename = glob.glob(model_folder + "*.json")[0]
    with open(param_dump_filename, 'r') as f:
        param_dict = json.load(f)
if not this_model_has_param_file:
    param_dict = {}
    param_dict["nbands"] = 3
### end - model to be loaded for prediction ###

#location of lenses, which are galaxies without lensing features
lenses_path = "data/test_data/lenses/"
test_data = glob.glob(lenses_path + "*_r_*.fits")
num_test = len(test_data)

loadsize = 100
num_sources = load_data.num_sources
num_lenses = load_data.num_lenses
num_neg = load_data.num_neg
NUM_PROCESSES = 2
default_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
}
CHUNK_SIZE = 25000
IMAGE_WIDTH = 101
IMAGE_HEIGHT = 101
center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2.0 - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
tform_identity = (
    skimage.transform.AffineTransform()
)  # this is an identity transform by default
ds_transforms_default = [tform_identity]
ds_transforms = ds_transforms_default  # CHANGE THIS LINE to select downsampling transforms to be used



######### CLASSES #########

class LoadAndProcessFixedTest(object):
    def __init__(self, ds_transforms, augmentation_transforms, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed_test(
            img_index,
            self.ds_transforms,
            self.augmentation_transforms,
            self.target_sizes,
        )


class LoadAndProcessNeg(object):  ##USATA
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_neg(
            img_index, self.ds_transforms, self.augmentation_params, self.target_sizes
        )


class LoadAndProcessSource(object):  ##USATA
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_source(
            img_index, self.ds_transforms, self.augmentation_params, self.target_sizes
        )


class LoadAndProcessLens(object):  ##USATA
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_lens(
            img_index, self.ds_transforms, self.augmentation_params, self.target_sizes
        )




######### FUNCTIONS #########
def load_and_process_image_fixed_test(
    img_index, ds_transforms, augmentation_transforms, target_sizes=None
):
    img_id = test_data[img_index]
    img = load_data.load_fits_test(img_id)
    if param_dict["nbands"] == 3:
        img = np.dstack((img, img, img))
    if param_dict["nbands"] == 1:
        img = np.expand_dims(img, axis=2)

    return [img]


def load_and_process_image_neg(
    img_index, ds_transforms, augmentation_params, target_sizes=None
):  ##USATA
    img_id = load_data.train_ids_neg[img_index]
    img = load_data.load_fits_neg(img_id)
    if param_dict["nbands"] == 3:
        img = np.dstack((img, img, img))
    # if param_dict["nbands"] == 1:
    #     img = np.expand_dims(img, axis=2)

    img_a = perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes)

    return img_a


def fast_warp(img, tf, output_shape=(53, 53), mode="reflect"):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    img = img.astype(np.float32)
    m = tf.params.astype(np.float32)
    img_wf = np.empty(
        (output_shape[0], output_shape[1], param_dict["nbands"]), dtype="float32"
    )
    for k in range(param_dict["nbands"]):
        img_wf[..., k] = skimage.transform._warps_cy._warp_fast(
            img[..., k], m, output_shape=output_shape, mode=mode
        ).astype(np.float32)
    return img_wf


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(
        scale=(1 / zoom, 1 / zoom),
        rotation=np.deg2rad(rotation),
        shear=np.deg2rad(shear),
        translation=translation,
    )
    tform = tform_center + tform_augment + tform_uncenter
    return tform


def random_perturbation_transform(
    zoom_range, rotation_range, shear_range, translation_range, do_flip=False
):

    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.uniform(
        *rotation_range
    )  # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = np.random.uniform(*shear_range)

    # # flip
    if do_flip and (np.random.randint(2) > 0):  # flip half of the time
        shear += 180
        rotation += 180

    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(
        np.random.uniform(*log_zoom_range)
    )  # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform(zoom, rotation, shear, translation)


def select_indices(num, num_selected):
    selected_indices = np.arange(num)
    np.random.shuffle(selected_indices)
    selected_indices = selected_indices[:num_selected]
    return selected_indices


def perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes=None):
    if target_sizes is None:
        target_sizes = [(53, 53) for _ in range(len(ds_transforms))]

    tform_augment = random_perturbation_transform(**augmentation_params)

    result = []
    for tform_ds, target_size in zip(ds_transforms, target_sizes):
        result.append(
            fast_warp(img, tform_ds + tform_augment, output_shape=target_size, mode="reflect").astype("float32")
        )  # crop here?

    return result


def load_and_process_image_source(
    img_index, ds_transforms, augmentation_params, target_sizes=None
):  ##USATA
    img_id = load_data.train_ids_source[img_index]
    img = load_data.load_fits_source(img_id)
    if param_dict["nbands"] == 3:
        img = np.dstack((img, img, img))
    # if param_dict["nbands"] == 1:
    #     img = np.expand_dims(img, axis=2)

    img_a = perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes)

    return img_a


def load_and_process_image_lens(
    img_index, ds_transforms, augmentation_params, target_sizes=None
):  ##USATA
    img_id = load_data.train_ids_lens[img_index]
    img = load_data.load_fits_lens(img_id)
    if param_dict["nbands"] == 3:
        img = np.dstack((img, img, img))
    # if param_dict["nbands"] == 1:
    #     img = np.expand_dims(img, axis=2)

    img_a = perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes)
    
    return img_a


def realtime_augmented_data_gen_pos(
    num_chunks=None,
    chunk_size=CHUNK_SIZE,
    augmentation_params=default_augmentation_params,  # keep
    ds_transforms=ds_transforms_default,
    target_sizes=None,
    processor_class=LoadAndProcessSource,
    processor_class2=LoadAndProcessLens,
    normalize=True,
    resize=False,
    range_min=0.02,
    range_max=0.5,
):
    """
    new version, using Pool.imap instead of Pool.map, to avoid the data structure conversion
    from lists to numpy arrays afterwards.
    """
    n = 0
    while True:
        if num_chunks is not None and n >= num_chunks:
            break
        selected_indices_sources = select_indices(num_sources, chunk_size)
        selected_indices_lenses = select_indices(num_lenses, chunk_size)

        labels = np.ones(chunk_size)

        process_func = processor_class(
            ds_transforms, augmentation_params, target_sizes
        )  # SOURCE
        process_func2 = processor_class2(
            ds_transforms, augmentation_params, target_sizes
        )  # LENS

        target_arrays_pos = [
            np.empty((chunk_size, size_x, size_y, param_dict["nbands"]), dtype="float32")
            for size_x, size_y in target_sizes
        ]

        pool1 = mp.Pool(NUM_PROCESSES)
        gen = pool1.imap(process_func, selected_indices_sources, chunksize=loadsize)

        pool2 = mp.Pool(NUM_PROCESSES)
        gen2 = pool2.imap(process_func2, selected_indices_lenses, chunksize=loadsize)

        k = 0
        for source, lens in zip(gen, gen2):
            source = np.array(source)
            lens = np.array(lens)
            imageData = lens + source / np.amax(source) * np.amax(lens) * np.random.uniform(range_min, range_max)
            scale_min = 0
            scale_max = imageData.max()
            imageData.clip(min=scale_min, max=scale_max)
            indices = np.where(imageData < 0)
            imageData[indices] = 0.0
            new_img = np.sqrt(imageData)
            if normalize:
                new_img = new_img / new_img.max() * 255.0
            target_arrays_pos[0][k] = new_img
            k += 1

        pool1.close()
        pool1.join()
        pool2.close()
        pool2.join()
        target_arrays_pos.append(labels.astype(np.int32))

        yield target_arrays_pos, chunk_size
        n += 1




def realtime_augmented_data_gen_neg(
    num_chunks=None,
    chunk_size=CHUNK_SIZE,
    augmentation_params=default_augmentation_params,  # keep
    ds_transforms=ds_transforms_default,
    target_sizes=None,
    processor_class=LoadAndProcessNeg,
    normalize=True,
    resize=False,
    resize_shape=(60, 60),
):
    """
    new version, using Pool.imap instead of Pool.map, to avoid the data structure conversion
    from lists to numpy arrays afterwards.
    """

    n = 0
    while True:
        if num_chunks is not None and n >= num_chunks:
            break
        selected_indices = select_indices(num_neg, chunk_size)
        labels = np.zeros(chunk_size)
        process_func = processor_class(ds_transforms, augmentation_params, target_sizes)

        target_arrays = [
            np.empty((chunk_size, size_x, size_y, param_dict["nbands"]), dtype="float32")
            for size_x, size_y in target_sizes
        ]
        pool = mp.Pool(NUM_PROCESSES)
        gen = pool.imap(
            process_func, selected_indices, chunksize=loadsize
        )  # lower chunksize seems to help to keep memory usage in check

        for k, imgs in enumerate(gen):
            for i, image in enumerate(imgs):
                scale_min = 0
                scale_max = image.max()
                image.clip(min=scale_min, max=scale_max)
                indices = np.where(image < 0)
                image[indices] = 0.0
                new_img = np.sqrt(image)
                if normalize:
                    new_img = new_img / new_img.max() * 255.0
                target_arrays[i][k] = new_img
        pool.close()
        pool.join()

        target_arrays.append(labels.astype(np.int32))

        yield target_arrays, chunk_size
        n += 1



def realtime_fixed_augmented_data_test(
    ds_transforms=ds_transforms_default,
    augmentation_transforms=[tform_identity],  # keep
    chunk_size=5514,
    target_sizes=None,
    processor_class=LoadAndProcessFixedTest,
):
    """
    by default, only the identity transform is in the augmentation list, so no augmentation occurs (only ds_transforms are applied).
    """
    selected_indices = np.arange(num_test)
    labels = np.zeros(chunk_size)
    num_ids_per_chunk = chunk_size // len(augmentation_transforms)  # number of datapoints per chunk - each datapoint is multiple entries!
    num_chunks = int(np.ceil(len(selected_indices) / float(num_ids_per_chunk)))

    process_func = processor_class(ds_transforms, augmentation_transforms, target_sizes)

    for n in range(num_chunks):
        indices_n = selected_indices[ n * num_ids_per_chunk : (n + 1) * num_ids_per_chunk ]
        current_chunk_size = len(indices_n) * len( augmentation_transforms )  # last chunk will be shorter!

        target_arrays = [
            np.empty(
                (current_chunk_size, size_x, size_y, param_dict["nbands"]),
                dtype="float32",
            )
            for size_x, size_y in target_sizes
        ]

        pool = mp.Pool(NUM_PROCESSES)
        gen = pool.imap(
            process_func, indices_n, chunksize=100
        )  # lower chunksize seems to help to keep memory usage in check

        for k, imgs_aug in enumerate(gen):
            for i, imgs in enumerate(imgs_aug):
                target_arrays[i][k] = imgs

        pool.close()
        pool.join()

        target_arrays.append(labels.astype(np.int32))

        yield target_arrays, current_chunk_size



def call_model(model="resnet"):
    if model == "resnet":
        multi_model = build_resnet()
    return multi_model


def build_resnet():
    model = resnet.ResnetBuilder.build_resnet_18((101,101,param_dict["nbands"]), 1)
    return model



# Count the number of true positive, true negative, false positive and false negative in for a prediction vector relative to the label vector.
def count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, threshold, beta_squarred, verbatim = False):
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
    
    if verbatim:
        if tot_count != len(prediction_vector):
            print("Total count {} of (TP, TN, FP, FN) is not equal to the length of the prediction vector: {}".format(tot_count, len(prediction_vector)), flush=True)

        print("Total Count {}\n\tTP: {}, TN: {}, FP: {}, FN: {}".format(tot_count, TP, TN, FP, FN), flush=True)
        
        print("precision = {}".format(precision), flush=True)
        print("recall    = {}".format(recall), flush=True)
        print("fp_rate   = {}".format(fp_rate), flush=True)
        print("accuracy  = {}".format(accuracy), flush=True)
        print("F beta    = {}".format(F_beta), flush=True)

    return TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta
######### END FUNCTIONS #########


######### SCRIPT #########
resize = False
normalize = True
pos_chunk_size = 11598
range_min =  0.02
range_max =  0.30
num_chunks = 500
input_sizes = [(101,101)]

augmented_data_gen_pos = realtime_augmented_data_gen_pos(range_min=range_min, range_max=range_max, num_chunks=num_chunks, chunk_size=pos_chunk_size, target_sizes=input_sizes, normalize=normalize, resize=resize, augmentation_params=default_augmentation_params)
augmented_data_gen_neg = realtime_augmented_data_gen_neg(num_chunks=num_chunks, chunk_size=num_neg, target_sizes=input_sizes, normalize=normalize, resize=resize, augmentation_params=default_augmentation_params)
augmented_data_gen_test_fixed = realtime_fixed_augmented_data_test(target_sizes=input_sizes)

if True:
    print("num_neg: {}".format(num_neg), flush=True)
    print("num_sources: {}".format(num_sources), flush=True)
    print("num_lenses: {}".format(num_lenses), flush=True) # is the test data in this case aswell, to be counted as negatives, because there are no lensing features in them.

if True:
    pos_data, labels_pos = next(augmented_data_gen_pos)[0]
    neg_data, labels_neg = next(augmented_data_gen_neg)[0]
    neg_data_lenses, labels_neg_lenses = next(augmented_data_gen_test_fixed)[0]             #this is the lenses set, meaning: that these are images of galaxies without any lensing features (no source is applied to it). the naming is confusing, i know, but i just went with the terminology already used in the rest of the existing code.

if True:
    print("pos data labels: {}".format(str(labels_pos)), flush=True)
    print("neg data labels: {}".format(str(labels_neg)), flush=True)
    print("neg data lenses labels: {}".format(str(labels_neg_lenses)), flush=True)

if True:
    #the data is not normalized, therefore divide by 255.0
    x_test = np.concatenate([pos_data, neg_data, neg_data_lenses], axis=0)/255.0
    print(x_test.shape, flush=True)
    y_test = np.concatenate([labels_pos, labels_neg, labels_neg_lenses], axis=0)
    print(y_test.shape, flush=True)

# load a keras model
multi_model = call_model(model="resnet")

#load the weights
multi_model.load_weights(h5_file)

prediction_vector = multi_model.predict(x_test)
print("Length prediction vector: {}".format(len(prediction_vector)), flush=True)

beta_squarred = 0.03
stepsize = 0.01
threshold_range = np.arange(stepsize,1.0,stepsize)

f_betas = []
with open(f_beta_full_path, 'w', newline='') as f_beta_file:
    writer = csv.writer(f_beta_file)
    writer.writerow(["p_threshold", "TP", "TN", "FP", "FN", "precision", "recall", "fp_rate", "accuracy", "f_beta"])
    for p_threshold in threshold_range:
        (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, p_threshold, beta_squarred)
        f_betas.append(F_beta)
        writer.writerow([str(p_threshold), str(TP), str(TN), str(FP), str(FN), str(precision), str(recall), str(fp_rate), str(accuracy), str(F_beta)])

print("saved csv with f_beta scores to: ".format(f_beta_full_path), flush=True)

plt.plot(list(threshold_range), f_betas)
plt.xlabel("p threshold")
plt.ylabel("F")
plt.title("F_beta score - Beta = {0:.2f}".format(math.sqrt(beta_squarred)))

plt.savefig(full_path_fBeta_figure)
print("figure saved: {}".format(full_path_fBeta_figure), flush=True)
plt.show()