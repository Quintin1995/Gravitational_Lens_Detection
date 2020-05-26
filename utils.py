from datetime import datetime
import matplotlib.pyplot as plt

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