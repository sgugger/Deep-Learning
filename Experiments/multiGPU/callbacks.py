from fastai.conv_learner import *


class LogResults(Callback):
    """
    Callback to log all the results of the training:
    - at the end of each epoch: training loss, validation loss and metrics
    """
    
    def __init__(self, learn, fname):
        super().__init__()
        self.learn, self.fname = learn, fname
        
    def on_train_begin(self):
        self.logs, self.epoch, self.n = "", 0, 0
        names = ["epoch", "trn_loss", "val_loss", "accuracy"]
        layout = "{!s:10} " * len(names)
        self.logs += layout.format(*names) + "\n"
    
    def on_batch_end(self, metrics):
        self.loss = metrics
    
    def on_epoch_end(self, metrics):
        self.save_stats(self.epoch, [self.loss] + metrics)
        self.epoch += 1
        
    def save_stats(self, epoch, values, decimals=6):
        layout = "{!s:^10}" + " {!s:10}" * len(values)
        values = [epoch] + list(np.round(values, decimals))
        self.logs += layout.format(*values) + "\n"

    def on_train_end(self):
        with open(self.fname, 'a') as f: f.write(self.logs)
