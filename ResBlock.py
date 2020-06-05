import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def _act(activation):
    if activation=="leakyreLU":
        act=lambda :layers.LeakyReLU(0.2)
    elif activation=="prelu":
        act=lambda  :layers.PReLU()
    else:
        act=lambda :layers.Activation(activation)
    return act

def ResBlock(filter, strides=1,activation="LeakyReLU"):
    reg = lambda: l2(1e-4)
    act=_act(activation)
    def f(x):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2D(filter, strides=strides, kernel_size=1, padding="same", kernel_regularizer=reg(),
                          kernel_initializer='he_normal')(x)

        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2D(filter, kernel_size=3, padding="same", kernel_regularizer=reg(), kernel_initializer='he_normal')(
            x)

        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2D(filter * 4, kernel_size=1, padding="same", kernel_regularizer=reg(),
                          kernel_initializer='he_normal')(x)

        if strides != 1:
            shortcut = layers.Conv2D(filter * 4, strides=strides, kernel_size=1, padding="same")(shortcut)

        return layers.Add()([x, shortcut])
    return f

def TransResBlock(filter, strides=1,activation="leakyreLU"):
    reg = lambda: l2(1e-14)
    act=_act(activation)
    def f(x):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2DTranspose(filter, strides=strides, kernel_size=1, padding="same", kernel_regularizer=reg(),
                          kernel_initializer='he_normal')(x)

        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2DTranspose(filter, kernel_size=3, padding="same", kernel_regularizer=reg(), kernel_initializer='he_normal')(
            x)

        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2DTranspose(filter * 4, kernel_size=1, padding="same", kernel_regularizer=reg(),
                          kernel_initializer='he_normal')(x)

        if strides != 1:
            shortcut = layers.Conv2DTranspose(filter * 4, strides=strides, kernel_size=1, padding="same")(shortcut)

        return layers.Add()([x, shortcut])
    return f

def SRResBlock(filter,activation="prelu"):
    reg = lambda: l2(1e-14)
    act = _act(activation)
    def f(x):
        shortcut=x
        x = layers.Conv2D(filter,kernel_size=3,padding="same",kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = act()(x)
        x = layers.Conv2D(filter, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x,shortcut])
        return x
    return f
def PixelShuffle(filter,r):
    def f(x):
        shape=x.shape[1:]
        x = layers.Reshape((r,r,filter,shape[0],shape[1]))(x)
        x = layers.Permute((3,4,1,5,2))(x)
        x = layers.Reshape((shape[0]*r,shape[1]*r,filter))(x)
        return x
    return f
def UpsampleBlock(filter,r=2,activation="prelu"):
    act=_act(activation)
    def f(x):
        x = layers.Conv2D(filter*r*r,kernel_size=3,padding="same")(x)
        x = PixelShuffle(filter,r)(x)
        x = act()(x)
        return x
    return f