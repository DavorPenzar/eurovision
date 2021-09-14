# -*- coding: utf-8 -*-

import tensorflow as _tf
import tensorflow.keras as _keras
import tensorflow.keras.callbacks as _callbacks

class DelayedEarlyStopping (_callbacks.EarlyStopping):
    def __new__ (cls, *args, **kwargs):
        instance = super(DelayedEarlyStopping, cls).__new__(cls)

        instance.delay = None

        return instance

    def __init__ (self, delay, *args, **kwargs):
        super(DelayedEarlyStopping, self).__init__(*args, **kwargs)

        self.delay = int(delay)

    def on_epoch_end(self, epoch, logs = None):
        if epoch > self.delay:
            super().on_epoch_end(epoch, logs)
