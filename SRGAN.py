from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.utils import to_categorical
from GAN_tools import DataPreprocess, generate_pic
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from ResBlock import *
from GAN_recoder import recoder

reg = lambda: L1L2(1e-14)
def Build_Generator(img_size,noise_size,):
    filter=256
    kernel_size=33-img_size[0]
    noise_input = layers.Input((noise_size,))
    x = layers.Dense(4*4*filter)(noise_input)
    x = layers.Reshape((4,4,filter))(x)
    x = layers.Conv2D(filter,kernel_size=3,padding="same")(x)
    x = layers.PReLU()(x)
    shortcut=x
    for i in range(5):
        x = SRResBlock(filter)(x)
    x = layers.Add()([x,shortcut])
    for i in range(3):
        filter=filter//2
        x = UpsampleBlock(filter)(x)
    x = layers.Conv2D(img_size[-1],kernel_size,activation="sigmoid")(x)
    generator=Model(noise_input,x,name="generator")
    return generator
def Build_Discriminator(img_size):
    filter=64
    discriminator_input=layers.Input(img_size)
    x=discriminator_input
    for i in range(3):
        filter*=2
        x = layers.Conv2D(filter, kernel_size=5, strides=2, padding="same")(x)
        # x=layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    discriminator_out =layers.Dense(1, activation='sigmoid',name="fake_out")(x)
    discriminator=Model(discriminator_input,discriminator_out,name="discriminator")
    return discriminator
def main():
    img_size=(32,32,3)
    noise_size=100

    generator=Build_Generator(img_size,noise_size)
    generator.summary()

    discriminator=Build_Discriminator(img_size)
    discriminator.trainable=True
    discriminator.compile(optimizer=Adam(),loss="binary_crossentropy",metrics=["acc"])
    discriminator.summary()

    discriminator.trainable = False
    noise_input = layers.Input((noise_size,))
    generated = generator(noise_input)
    discriminated = discriminator(generated)

    model = Model(noise_input, discriminated,name="model")
    model.summary()
    model.compile(optimizer=Adam(),loss="binary_crossentropy")

    epoch = 5000
    batch = 100
    (x_train, y_train), (x_test, y_test) = DataPreprocess(cifar10.load_data())

    def update_func():
        mask = np.random.randint(0, x_train.shape[0], batch)
        noise = np.random.normal(0, 1, (batch, noise_size))
        sup_imgs = x_train[mask]
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(sup_imgs, np.ones((batch)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch)))
        g_loss = model.train_on_batch(noise, np.ones((batch)))
        return [d_loss_real, d_loss_fake, g_loss]

    labels = ["d_real_loss", "d_real_acc", "d_fake_loss", "d_fake_acc", "g_loss"]
    graph = {"acc": ["d_real_acc", "d_fake_acc"],
             "loss": ["d_real_loss", "d_fake_loss", "g_loss"]}
    rec = recoder("SRGAN_cifar10",
                  [generator, discriminator, model],
                  update_func,
                  epoch=epoch,
                  labels=labels,
                  plot_values=graph)
    rec.run()
if __name__ == '__main__':
    main()
