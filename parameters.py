import os
from utils import *

class Parameters(object):
    def __init__(self, settings):
        #hyper parameters
        self.model_name      = settings["model_name"]  # for example "first_model" must be something unique
        self.num_chunks      = settings["num_chunks"]
        self.chunk_size      = settings["chunk_size"]
        self.batch_size      = settings["batch_size"]
        self.num_batch_augm  = settings["num_batch_augm"]
        self.input_sizes     = [(settings["img_rows"], settings["img_cols"])]

        #other parameters
        self.nbands          = settings["nbands"]
        self.normalize       = settings["normalize"]   # normalize the images to max of 255 (valid for single-band only)
        self.augm_pred       = settings["augm_pred"]
        self.learning_rate   = settings["learning_rate"]
        self.resize          = settings["resize"]
        
        #more parameters
        self.range_min       = settings["range_min"]
        self.range_max       = settings["range_max"]

        #more parameters
        self.buffer_size     = settings["buffer_size"]
        self.avg_img         = settings["avg_img"]

        #even more params
        self.input_shape     = (self.input_sizes[0][0], self.input_sizes[0][1], 3)

        #path stuff
        self.root_dir_models        = settings["root_dir_models"]
        self.model_folder           = get_time_string()     #A model will be stored in a folder with just a date&time as folder name
        self.model_path             = os.path.join(self.root_dir_models, self.model_folder)     #path of model
        self.make_model_dir()       #create directory for all data concerning this model.
        
        #weights .h5 file
        self.weights_extension      = ".h5"                 #Extension for saving weights
        self.filename_weights       = self.model_name + "_weights_only" + self.weights_extension
        self.full_path_of_weights   = os.path.join(self.model_path, self.filename_weights)

        #csv logger file to store the callback of the .fit function. It stores the history of the training session.
        self.history_extension      = ".log"                 #Extension for history callback
        self.filename_history       = self.model_name + "_history" + self.history_extension
        self.full_path_of_history   = os.path.join(self.model_path, self.filename_history)

        #output path of .png
        self.figure_extension      = ".png"                 #Extension for figure 
        self.filename_figure       = self.model_name + "_results" + self.figure_extension
        self.full_path_of_figure   = os.path.join(self.model_path, self.filename_figure)


    def make_model_dir(self):
        try:
            os.mkdir(self.model_path)
        except OSError:
            print ("Creation of the directory %s failed" % self.model_path)
        else:
            print ("Successfully created the directory %s " % self.model_path)