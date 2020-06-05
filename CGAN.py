from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from GAN_tools import DataPreprocess, generate_pic_with_label
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2

reg = lambda: L1L2(1e-14)
def Build_Generator(img_size,noise_size,):
    kernel_size=33-img_size[0]
    noise_input = layers.Input((noise_size,))
    label_input=layers.Input((10,))
    x = layers.Concatenate()([noise_input,label_input])
    x = layers.Dense(4*4*256,activation="relu")(x)
    x = layers.Reshape((4,4,256))(x)
    for i in [256, 128, 64]:
        x = layers.Conv2DTranspose(i, (5, 5), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.UpSampling2D()(x)
    x = layers.Conv2D(img_size[-1], kernel_size=kernel_size,activation="sigmoid")(x)
    generator=Model([noise_input,label_input],x)
    return generator
def Build_Discriminator(img_size):
    discriminator_input=layers.Input(img_size)
    label_input=layers.Input((10,))
    x = layers.Conv2D(16, (3, 3), padding='same')(discriminator_input)
    for i in [32, 64, 128]:
        x = layers.AveragePooling2D((2,2))(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv2D(i, (3, 3), padding='same',kernel_regularizer=reg())(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Concatenate()([x,label_input])
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(0.1)(x)
    discriminator_out =layers.Dense(1, activation='sigmoid',name="fake_out")(x)
    discriminator=Model([discriminator_input,label_input],discriminator_out,name="discriminator")
    return discriminator
def main():
    img_size=(28,28,1)
    noise_size=100

    generator=Build_Generator(img_size,noise_size)
    generator.summary()

    discriminator=Build_Discriminator(img_size)
    discriminator.trainable=True
    discriminator.compile(optimizer=Adam(),loss="binary_crossentropy",metrics=["acc"])
    discriminator.summary()

    discriminator.trainable = False
    noise_input = layers.Input((noise_size,))
    label_input=layers.Input((10,))
    generated = generator([noise_input,label_input])
    discriminated = discriminator([generated,label_input])
    model = Model([noise_input,label_input], discriminated)
    model.summary()
    model.compile(optimizer=Adam(),loss="binary_crossentropy")

    epoch=8000
    batch=128
    (x_train, y_train), (x_test, y_test) = DataPreprocess(mnist.load_data())

    for i in range(epoch):
        print("epoch:", i + 1, "/", epoch)
        mask = np.random.randint(0, x_train.shape[0], batch)
        noise = np.random.normal(0, 1, (batch,noise_size))
        sup_imgs = x_train[mask]
        sup_class= y_train[mask]
        gen_imgs = generator.predict([noise,sup_class])
        d_loss_real = discriminator.train_on_batch([sup_imgs,sup_class], np.ones((batch)))
        d_loss_fake = discriminator.train_on_batch([gen_imgs,sup_class], np.zeros((batch)))

        input_noise = np.random.normal(0, 1, (batch, 100))
        g_loss = model.train_on_batch([input_noise,sup_class], np.ones((batch)))

        print("discriminator real loss:", d_loss_real)
        print("discriminator fake loss:", d_loss_fake)
        print("generator loss:", g_loss)
        if i % 100 == 0:
            generate_pic_with_label(generator, filepath="CGANv2",filename=str(i // 100))

if __name__ == '__main__':
    main()
