from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from GAN_tools import DataPreprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from GAN_recoder import recoder
from tensorflow import keras

def main():
    img_size=(32,32,3)
    noise_size=100

    generator=keras.models.load_model("GAN_data/Projection_ACGAN/models/generator_vfinal.h5")
    generator.summary()

    discriminator=keras.models.load_model("GAN_data/Projection_ACGAN/models/discriminator_vfinal.h5")
    discriminator.summary()

    model =keras.models.load_model("GAN_data/Projection_ACGAN/models/model_vfinal.h5")
    model.summary()

    epoch=100000
    batch=128
    (x_train, y_train), (x_test, y_test) = DataPreprocess(cifar10.load_data())

    def update_func():
        mask = np.random.randint(0, x_train.shape[0], batch)
        noise = np.random.normal(0, 1, (batch, noise_size))
        sup_imgs = x_train[mask]
        sup_class = y_train[mask]
        gen_imgs = generator.predict([noise, sup_class])
        d_loss_real = discriminator.train_on_batch([sup_imgs,sup_class], np.ones((batch)))
        d_loss_fake = discriminator.train_on_batch([gen_imgs,sup_class], np.zeros((batch)))
        d_loss=0.5*np.add(d_loss_fake,d_loss_real)

        input_noise = np.random.normal(0, 1, (batch, 100))
        g_loss = model.train_on_batch([input_noise, sup_class], [np.ones((batch))])

        return [d_loss,g_loss]
    labels=["d_loss","d_acc","g_loss"]
    graph={"loss":["d_loss","g_loss"]
           }
    rec=recoder("Projection_ACGAN_additional_train",
                [generator,discriminator,model],
                update_func,
                epoch=epoch,
                labels=labels,
                plot_values=graph,
                isUselabel=True)
    rec.run()

if __name__ == '__main__':
    main()
