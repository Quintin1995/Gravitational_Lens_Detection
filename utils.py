from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import numpy as np
import resnet



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


def build_resnet(parameters):
    model = resnet.ResnetBuilder.build_resnet_18(parameters.input_shape, 1)  # 18
    return model


def call_model(parameters, model="resnet"):
    if model == "resnet":
        multi_model = build_resnet(parameters)
    return multi_model


def load_run_yaml(yaml_run_path):
    #opens run.yaml and load all the settings into a dictionary.
    with open(yaml_run_path) as file:
        settings = yaml.load(file)
        print("\nSettings: {}".format(yaml_run_path))
        for i in settings:
            print(str(i) + ": " + str(settings[i]))
        print("\nAll settings loaded.\n\n")
        return settings


def get_time_string():
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")


# Define a nice plot function for the accuracy and loss over time
# History is the object returns by a model.fit()
def plot_history(history, settings):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(settings.full_path_of_figure)
    print("\nsaved figure to: " + settings.full_path_of_figure)