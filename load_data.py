import numpy as np
from scipy import ndimage
import scipy
import glob
import itertools
import threading
import skimage.transform
import skimage.io
import gzip
import os
import queue
import multiprocessing as mp
import pickle
from astropy.io import fits
import subprocess
import math
import pyfits
import time
from PIL import Image
import HumVI_online_lensing as rgb

# import humvi

preprocess = False

train_ids_lens = np.load("data/train_ids_lens.npy")
train_ids_source = np.load("data/train_ids_source.npy")
train_ids_neg = np.load("data/train_ids_neg.npy")
# ,train_ids_real_lenses = np.load("data/train_ids_real_lenses.npy")

dic_train_neg = "data/train_dic_neg.p"
cutout_dict_train_neg = pickle.load(open(dic_train_neg, "rb"))
num_neg = len(cutout_dict_train_neg)

dic_train_lens = "data/train_dic_lenses.p"
cutout_dict_train_lens = pickle.load(open(dic_train_lens, "rb"))
num_lenses = len(cutout_dict_train_lens)

dic_train_source = "data/train_dic_sources.p"
cutout_dict_train_source = pickle.load(open(dic_train_source, "rb"))
num_sources = len(cutout_dict_train_source)

dic_train_neg = "data/train_dic_real_lenses.p"
# cutout_dict_train_real_lenses=pickle.load( open( dic_train_neg, "rb" ) )
# num_real_lenses=len(cutout_dict_train_real_lenses)


nx = 101
ny = 101
f1 = pyfits.open("data/PSF_KIDS_129.0_1.5_i.fits")  # PSF
d1 = f1[0].data
d1 = np.asarray(d1)
nx_, ny_ = np.shape(d1)
PSF_i = np.zeros((nx, ny))  # output
dx = (nx - nx_) // 2  # shift in x
dy = (ny - ny_) // 2  # shift in y
for ii in range(nx_):  # iterating over input array
    for jj in range(ny_):
        PSF_i[ii + dx][jj + dy] = d1[ii][jj]

f1 = pyfits.open("data/PSF_KIDS_133.4_ 2.5_g.fits")  # PSF
d1 = f1[0].data
d1 = np.asarray(d1)
nx_, ny_ = np.shape(d1)
PSF_g = np.zeros((nx, ny))  # output
dx = (nx - nx_) // 2  # shift in x
dy = (ny - ny_) // 2  # shift in y
for ii in range(nx_):  # iterating over input array
    for jj in range(ny_):
        PSF_g[ii + dx][jj + dy] = d1[ii][jj]

f1 = pyfits.open("data/PSF_KIDS_175.0_-0.5_r.fits")  # PSF
d1 = f1[0].data
d1 = np.asarray(d1)
nx_, ny_ = np.shape(d1)
PSF_r = np.zeros((nx, ny))  # output
dx = (nx - nx_) // 2  # shift in x
dy = (ny - ny_) // 2  # shift in y
for ii in range(nx_):  # iterating over input array
    for jj in range(ny_):
        PSF_r[ii + dx][jj + dy] = d1[ii][jj]

seds = np.loadtxt("data/SED_colours_2017-10-03.dat")

Rg = 3.30
Rr = 2.31
Ri = 1.71

perc_range = (0.02, 0.30)


def load_fits_source(img_id):
    path = "data/training/sources/"
    image = None
    while image is None:
        try:
            image = fits.getdata(path + cutout_dict_train_source[img_id]["name"])
            
            image = image.astype(np.float32)    #is important
            image = scipy.signal.fftconvolve(image, PSF_r, mode="same")
            hdulist = pyfits.open(path + cutout_dict_train_source[img_id]["name"])
            prihdr = hdulist[0].header
            # ein_rad= prihdr['LENSER']
            # mags = prihdr['MAG']
            # if ein_rad < 1.:
            #    image=None
            #    img_id=img_id+1
        except IOError:
            img_id = img_id + 1
            pass

    # shift_x=np.random.randint(-4, 4)
    # shift_y=np.random.randint(-4, 4)
    # image=np.roll(image, shift_x, axis=1)
    # image=np.roll(image, shift_y, axis=0)

    img = np.expand_dims(image, axis=2)
    return img


def load_fits_lens(img_id):
    image = fits.getdata(cutout_dict_train_lens[img_id]["name"])
    image = image.astype("float32")
    image = np.expand_dims(image, axis=2)
    return image


def load_fits_neg(img_id):
    image = fits.getdata(cutout_dict_train_neg[img_id]["name"])
    image = image.astype("float32")
    image = np.expand_dims(image, axis=2)
    return image


# def load_fits_val(path):
#        image=np.load(path)
#        img=np.expand_dims(image, axis=2)
#        return image


def load_fits_test(path, normalize=True):  # THIS IS USED FOR TEST TIME
    image = fits.getdata(path)
    image = image.astype("float32")
    img = np.array(image, copy=True)
    scale_min = 0
    scale_max = img.max()
    img = img.clip(min=scale_min, max=scale_max)
    indices = np.where(img < 0)
    img[indices] = 0.0
    imageData = np.sqrt(img)
    if normalize:
        imageData = imageData / imageData.max() * 255.0
    new_img = np.flipud(imageData)
    if preprocess:
        new_img = ((new_img / 255.0) - 0.5) * 2

    return new_img


def load_fits_pos_col(img_id_lens, img_id_src, perc_range=perc_range):

    lens_r = cutout_dict_train_lens[img_id_lens]["name"]
    lens_g = (
        cutout_dict_train_lens[img_id_lens]["name"].split("_r_")[0]
        + "_g_"
        + cutout_dict_train_lens[img_id_lens]["name"].split("_r_")[1]
    )  #!!!
    lens_i = (
        cutout_dict_train_lens[img_id_lens]["name"].split("_r_")[0]
        + "_i_"
        + cutout_dict_train_lens[img_id_lens]["name"].split("_r_")[1]
    )
    lens_r_data = fits.getdata(cutout_dict_train_lens[img_id_lens]["name"])

    perc = np.random.uniform(perc_range[0], perc_range[1])

    path = "data/training/sources/"
    image = None
    while image is None:
        try:
            image = fits.getdata(path + cutout_dict_train_source[img_id_src]["name"])
            hdulist = pyfits.open(path + cutout_dict_train_source[img_id_src]["name"])
            prihdr = hdulist[0].header
            ein_rad = prihdr["LENSER"]
            if ein_rad < 1.0:
                image = None
                img_id_src = img_id_src + 1
        except IOError:
            img_id_src = np.random.randint(0, num_sources)
            pass

    index = np.random.randint(0, seds.shape[0])
    ext_range = abs(np.random.normal(0, 0.1))
    r_mag = seds[index][3] + Rr * ext_range + np.random.uniform(-1, 1)
    g_mag = seds[index][2] + Rg * ext_range + np.random.uniform(-1, 1)
    i_mag = seds[index][4] + Ri * ext_range + np.random.uniform(-1, 1)

    gmr = g_mag - r_mag
    rmi = r_mag - i_mag

    flux_g = 10 ** (-0.4 * gmr)
    flux_i = 10 ** (0.4 * rmi)

    image_r = image
    image_g = image * flux_g
    image_i = image * flux_i

    image_r = scipy.signal.fftconvolve(image_r, PSF_r, mode="same")
    image_g = scipy.signal.fftconvolve(image_g, PSF_g, mode="same")
    image_i = scipy.signal.fftconvolve(image_i, PSF_i, mode="same")

    lens_r_data[np.isnan(lens_r_data)] = 0
    lens_r_data[np.isinf(lens_r_data)] = 0

    lens_max = np.max(lens_r_data[47:55, 47:55])
    source_r = image_r / np.max(image_r) * lens_max * perc
    source_g = image_g / np.max(image_r) * lens_max * perc
    source_i = image_i / np.max(image_r) * lens_max * perc

    image = rgb.rgb_composer(lens_i, lens_r, lens_g, source_i, source_r, source_g)
    final_img = np.asarray(image)

    return final_img


# def load_fits_lens_col(img_id):
#        path = "data/training/lenses/"
#        lens_r = path+cutout_dict_train_lens[img_id]['name']
#        lens_g = path+cutout_dict_train_lens[img_id]['name'].split('_r_')[0]+'_g_'+cutout_dict_train_lens[img_id]['name'].split('_r_')[1] #!!!
#        lens_i = path+cutout_dict_train_lens[img_id]['name'].split('_r_')[0]+'_i_'+cutout_dict_train_lens[img_id]['name'].split('_r_')[1]
#        image=rgb.rgb_composer(lens_i,lens_r,lens_g)
#        image=np.asarray(image)
#        return image


def load_fits_neg_col(img_id):
    image_r = cutout_dict_train_neg[img_id]["name"]
    image_g = (
        cutout_dict_train_neg[img_id]["name"].split("_r_")[0]
        + "_g_"
        + cutout_dict_train_neg[img_id]["name"].split("_r_")[1]
    )
    image_i = (
        cutout_dict_train_neg[img_id]["name"].split("_r_")[0]
        + "_i_"
        + cutout_dict_train_neg[img_id]["name"].split("_r_")[1]
    )
    image = humvi.compose(image_i, image_r, image_g)
    image = np.asarray(image)

    return image


def load_fits_test_col(path, preprocess=preprocess):
    image_r = path
    image_g = path.split("_r_")[0] + "_g_" + path.split("_r_")[1]  #!!!
    image_i = path.split("_r_")[0] + "_i_" + path.split("_r_")[1]
    image = humvi.compose(image_i, image_r, image_g)
    image = np.asarray(image)
    if preprocess:
        image = ((image / 255.0) - 0.5) * 2

    return image


def buffered_gen_mp(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = mp.Queue(maxsize=buffer_size)

    def _buffered_generation_process(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                # print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = next(source_gen)
            except StopIteration:
                # print "DEBUG: OUT OF DATA, CLOSING BUFFER"
                buffer.close()  # signal that we're done putting data in the buffer
                break

            buffer.put(data)

    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()

    while True:
        try:
            # yield buffer.get()
            # just blocking on buffer.get() here creates a problem: when get() is called and the buffer
            # is empty, this blocks. Subsequently closing the buffer does NOT stop this block.
            # so the only solution is to periodically time out and try again. That way we'll pick up
            # on the 'close' signal.
            try:
                yield buffer.get(True, timeout=sleep_time)
            except queue.Empty:
                if not process.is_alive():
                    break  # no more data is going to come. This is a workaround because the buffer.close() signal does not seem to be reliable.

                # print "DEBUG: queue is empty, waiting..."
                pass  # ignore this, just try again.

        except IOError:  # if the buffer has been closed, calling get() on it will raise IOError.
            # this means that we're done iterating.
            # print "DEBUG: buffer closed, stopping."
            break


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)
