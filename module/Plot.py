import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, logging, output_folder, current=1):
        self.logging = logging
        self.output_folder = output_folder
        self.current = current
        
    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
   
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.logging.info("Epoch {} -> Train Loss: {:.4f} Valid Loss: {:.4f}"
                          .format(self.i, logs.get('loss'), logs.get('val_loss')))
        self.logging.info("Epoch {} -> Train Acc: {:.4f} Valid Acc: {:.4f}"
                          .format(self.i, logs.get('acc'), logs.get('val_acc')))
        self.i += 1
        plt.plot(self.x, self.losses, 'r--', label="Train_Loss")
        plt.plot(self.x, self.val_losses, 'b-', label="Valid_Loss")
        plt.legend()
        plt.savefig(self.output_folder+"loss-{}.png".format(self.current))
        plt.show();