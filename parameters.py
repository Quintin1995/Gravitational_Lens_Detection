import os
from utils import *
import json

class Parameters(object):
    def __init__(self, settings):
        # Hyper parameters
        self.model_name      = settings["model_name"]  # for example "first_model" must be something unique
        self.num_chunks      = settings["num_chunks"]
        self.chunk_size      = settings["chunk_size"]
        self.batch_size      = settings["batch_size"]
        self.num_batch_augm  = settings["num_batch_augm"]
        self.input_sizes     = [(settings["img_rows"], settings["img_cols"])]

        # Determine whether the user wants to train or predict
        self.mode            = settings["mode"]

        # Path to weights file for prediction (the referenced file should be a .h5 file that is trained)
        self.full_path_predict_weights = os.path.join(settings["path_trained_folder"], settings["filename_trained_weights"])

        # Other parameters
        self.nbands          = settings["nbands"]
        self.normalize       = settings["normalize"]   # normalize the images to max of 255 (valid for single-band only)
        self.augm_pred       = settings["augm_pred"]
        self.learning_rate   = settings["learning_rate"]
        self.resize          = settings["resize"]
        
        # More parameters
        self.range_min       = settings["range_min"]
        self.range_max       = settings["range_max"]

        # More parameters
        self.buffer_size     = settings["buffer_size"]
        self.avg_img         = settings["avg_img"]

        # Even more params
        self.input_shape     = (self.input_sizes[0][0], self.input_sizes[0][1], 3)

        # Default Augmentation Params           This dictionary holds all default data augmentation parameters
        self.default_augmentation_params = {
            "zoom_range": (settings["zoom_range_min"], settings["zoom_range_max"]),
            "rotation_range": (settings["rotation_range_min"], settings["rotation_range_max"]),
            "shear_range": (settings["shear_range_min"], settings["shear_range_max"]),
            "translation_range": (settings["translation_range_min"], settings["translation_range_max"]),
            "do_flip": settings["do_flip"],
        }

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
        self.history_extension      = ".csv"                 #Extension for history callback
        self.filename_history       = self.model_name + "_history" + self.history_extension
        self.full_path_of_history   = os.path.join(self.model_path, self.filename_history)

        #output path of .png
        self.figure_extension      = ".png"                 #Extension for figure 
        self.filename_figure       = self.model_name + "_results" + self.figure_extension
        self.full_path_of_figure   = os.path.join(self.model_path, self.filename_figure)

        #output path of .json       dumps all parameters into a json file
        self.param_dump_extension  = ".json"                #Extension for the paramters being written to a file
        self.filename_param_dump   = self.model_name + "_param_dump" + self.param_dump_extension
        self.full_path_param_dump  = os.path.join(self.model_path, self.filename_param_dump)

        #store all parameters of this object into a json file
        self.write_parameters_to_file()


    def make_model_dir(self):
        try:
            os.mkdir(self.model_path)
        except OSError:
            print ("Creation of the directory %s failed" % self.model_path)
        else:
            print ("Successfully created the directory: %s " % self.model_path)


    #write all the paramters defined in parameters class to a file
    def write_parameters_to_file(self):
        with open(self.full_path_param_dump, 'w') as outfile:
            json_content = self.toJSON()
            outfile.write(json_content)
            print("Wrote all run parameters to directory: {}".format(self.full_path_param_dump))


    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True,indent=4)
