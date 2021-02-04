import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class PlotLr(keras.callbacks.Callback):
    def __init__(self, logging, output_folder, current=1):
        self.logging = logging
        self.output_folder = output_folder
        self.current = current
        
    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.fig = plt.figure()
        self.logs = []
   
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('lr'))
        self.logging.info("Epoch {} -> Learning Rate: {:.4f}"
                          .format(self.i, logs.get('lr')))
        self.i += 1
        plt.plot(self.x, self.losses, label="learning rate")
        plt.legend()
        plt.savefig(self.output_folder+"lr-{}.png".format(self.current))
        plt.show();