import numpy as np
from tensorflow.keras.utils import to_categorical
def Preprocess(args):
    (x_train, y_train), (x_test,y_test) = args[0]
    x_train= x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    if x_train.shape[-1]!=3:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test,y_test)