from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import cifar10,mnist
from tensorflow.keras.utils import to_categorical
from GAN_tools import DataPreprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from GAN_recoder import recoder
from tensorflow.keras.preprocessing import image
from ResBlock import ResBlock


reg = lambda: L1L2(1e-14)
initializer="he_normal"
def Build_Generator(img_size,noise_size):
    def bn_act(activation="relu",momentum=0.8,**kwargs):
        def f(x):
            x=layers.BatchNormalization(momentum=momentum)(x)
            if activation=="leakyrelu":
                x=layers.LeakyReLU(kwargs["alpha"])(x)
            else:
                x=layers.Activation(activation)(x)
            return x
        return f

    kernel_size=33-img_size[0]
    noise_input = layers.Input((noise_size,))
    label_input=layers.Input((10,))
    x = layers.Concatenate()([noise_input,label_input])
    x = layers.Dense(4*4*256)(x)
    x = layers.Reshape((4,4,256))(x)
    x = bn_act(activation="leakyrelu",alpha=0.2)(x)
    for i in [256, 128, 64]:
        x = layers.UpSampling2D((2,2))(x)
        x = layers.Conv2DTranspose(i, (5, 5), padding='same')(x)
        x = bn_act(activation="leakyrelu",alpha=0.2)(x)
    x = layers.Conv2D(img_size[-1], kernel_size=kernel_size,activation="sigmoid")(x)
    generator=Model([noise_input,label_input],x,name="generator")
    return generator
def Build_Discriminator(img_size):
    discriminator_input=layers.Input(img_size)
    label_input=layers.Input((10,))
    x = layers.GaussianNoise(0.1)(discriminator_input)
    for i in [64,128,256]:
        x = layers.Conv2D(i,kernel_size=5,strides=2,padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    product = layers.Dense(10)(x)
    product = layers.Dot(1)([product,label_input])
    sum_out =layers.Dense(1,name="fake_out")(x)
    discriminator_out=layers.Add(name="discriminator_out")([product,sum_out])
    discriminator_out=layers.Activation("sigmoid")(discriminator_out)
    discriminator=Model([discriminator_input,label_input],discriminator_out,name="discriminator")
    return discriminator
def main():
    img_size=(32,32,3)
    noise_size=100

    generator=Build_Generator(img_size,noise_size)
    generator.summary()

    discriminator=Build_Discriminator(img_size)
    discriminator.trainable=True
    loss="binary_crossentropy"
    discriminator.compile(optimizer=Adam(),loss=loss,metrics=["acc"])
    discriminator.summary()

    discriminator.trainable = False
    noise_input = layers.Input((noise_size,))
    label_input=layers.Input((10,))
    generated = generator([noise_input,label_input])
    discriminated = discriminator([generated,label_input])
    model = Model([noise_input,label_input], discriminated,name="model")
    model.summary()

    loss="binary_crossentropy"
    model.compile(optimizer=Adam(),loss=loss)

    epoch=50000
    batch=100
    (x_train, y_train), (x_test, y_test) = DataPreprocess(cifar10.load_data())
    """
    params = {
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 5,
        "zoom_range": 0.2,
        "channel_shift_range": 5,
        "brightness_range": [0.5, 1.0],
        "rescale": 1. / 255
    }
    datagen = image.ImageDataGenerator(**params)
    train_iter = datagen.flow(x=x_train, y=y_train, batch_size=batch)
    """

    real=np.ones((batch))
    fake=np.zeros((batch))
    def update_func():
        noise = np.random.normal(0, 1, (batch, noise_size))
        mask = np.random.randint(0,x_train.shape[0],batch)
        sup_imgs ,sup_class = x_train[mask],y_train[mask]
        gen_imgs = generator.predict([noise, sup_class])
        d_loss_real = discriminator.train_on_batch([sup_imgs,sup_class], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs,sup_class], fake)
        d_loss=0.5*np.add(d_loss_fake,d_loss_real)

        input_noise = np.random.normal(0, 1, (batch, 100))
        g_loss = model.train_on_batch([input_noise, sup_class], real)

        return [d_loss,g_loss]
    labels=["d_loss","d_acc","g_loss"]
    graph={"loss":["d_loss","g_loss"]
           }
    rec=recoder("Projection_ACGAN_cifar10",
                [generator,discriminator,model],
                update_func,
                epoch=epoch,
                labels=labels,
                plot_values=graph,
                isUselabel=True)
    rec.run()

if __name__ == '__main__':
    main()
