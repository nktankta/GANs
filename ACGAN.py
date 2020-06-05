from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from GAN_tools import DataPreprocess, generate_pic
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from GAN_recoder import recoder

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
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D()(x)
    x = layers.Conv2D(img_size[-1], kernel_size=kernel_size,activation="sigmoid")(x)
    generator=Model([noise_input,label_input],x,name="generator")
    return generator
def Build_Discriminator(img_size):
    discriminator_input=layers.Input(img_size)
    x = layers.Conv2D(16, (3, 3), padding='same')(discriminator_input)
    for i in [32 , 64, 128,256]:
        x = layers.AveragePooling2D((2,2))(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv2D(i, (3, 3), padding='same',kernel_regularizer=reg())(x)
    x = layers.GlobalAveragePooling2D()(x)
    discriminator_out =layers.Dense(1, activation='sigmoid',name="fake_out")(x)
    class_out=layers.Dense(10,activation="softmax",name="class_out")(x)
    discriminator=Model([discriminator_input],[discriminator_out,class_out],name="discriminator")
    return discriminator
def main():
    img_size=(28,28,1)
    noise_size=100

    generator=Build_Generator(img_size,noise_size)
    generator.summary()

    discriminator=Build_Discriminator(img_size)
    discriminator.trainable=True
    loss={"fake_out":"binary_crossentropy",
          "class_out":"categorical_crossentropy"}
    discriminator.compile(optimizer=Adam(),loss=loss,metrics=["acc"])
    discriminator.summary()

    discriminator.trainable = False
    noise_input = layers.Input((noise_size,))
    label_input=layers.Input((10,))
    generated = generator([noise_input,label_input])
    discriminated = discriminator(generated)
    model = Model([noise_input,label_input], discriminated,name="model")
    model.summary()

    loss={"discriminator":"binary_crossentropy",
          "discriminator_1":"categorical_crossentropy"}
    model.compile(optimizer=Adam(),loss=loss)

    epoch=5000
    batch=128
    (x_train, y_train), (x_test, y_test) = DataPreprocess(mnist.load_data())

    def update_func():
        mask = np.random.randint(0, x_train.shape[0], batch)
        noise = np.random.normal(0, 1, (batch, noise_size))
        sup_imgs = x_train[mask]
        sup_class = y_train[mask]
        gen_imgs = generator.predict([noise, sup_class])
        d_loss_real = discriminator.train_on_batch([sup_imgs], [np.ones((batch)),sup_class])
        d_loss_fake = discriminator.train_on_batch([gen_imgs], [np.zeros((batch)),sup_class])
        d_loss=0.5*np.add(d_loss_fake,d_loss_real)

        input_noise = np.random.normal(0, 1, (batch, 100))
        g_loss = model.train_on_batch([input_noise, sup_class], [np.ones((batch)),sup_class])

        return [d_loss,g_loss]
    labels=["d_loss","d_binary_loss","d_categorical_loss","d_binary_acc","d_categorical_acc","g_loss","g_binary_loss","g_categorical_loss"]
    graph={"acc":["d_binary_acc","d_categorical_acc"],
           "d_losses":["d_loss","d_binary_loss","d_categorical_loss"],
           "loss":["d_loss","g_loss"]
           }
    rec=recoder("ACGAN",
                [generator,discriminator,model],
                update_func,
                epoch=epoch,
                labels=labels,
                plot_values=graph,
                isUselabel=True)
    rec.run()

if __name__ == '__main__':
    main()
