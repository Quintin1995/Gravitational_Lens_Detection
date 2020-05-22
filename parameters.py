



class Parameters(object):
    def __init__(self):
        self.input_sizes = [(101,101)] # size of the input images

        self.default_augmentation_params = {         
            'zoom_range': (1/1.1, 1.),
            'rotation_range': (0, 180),
            'shear_range': (0, 0),
            'translation_range': (-4, 4),
            'do_flip': True,
        }
        self.num_chunks      = 2000
        self.chunk_size      = 1280
        self.batch_size      = 64 
        self.num_batch_augm  = 20 
        self.nbands          = 1
        self.normalize       = True   # normalize the images to max of 255 (valid for single-band only)
        self.augm_pred       = True
        self.model_name      = 'baseline_model'
        self.learning_rate   = 0.0001 #for the neural network
        self.resize=False
            
        self.range_min=0.02
        self.range_max=0.30