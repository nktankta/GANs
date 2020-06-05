from abc import ABCMeta, abstractmethod
import numpy as np
from DataPreprocess import Preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
class GAN(metaclass=ABCMeta):
    def __init__(self):
        self.isUseLabel=False
        self.epoch=3000
        self.batch=128
        self.D_loss="binary_crossentropy"
        self.G_loss="binary_crossentropy"
        self.D_optimizer=Adam()
        self.G_optimizer=Adam()
    @abstractmethod
    def Build_Generator(self,img_size,noise_size,*args):
        pass

    @abstractmethod
    def Build_Discriminator(self,img_size,outputs):
        pass

    def Training(self,*args,noise_size=100,output_shape=[1]):
        (x_train, y_train), (x_test, y_test)=Preprocess(args)
        img_size=x_train.shape[1:]
        classes=y_train.shape[-1]
        if self.isUseLabel:
            self.Generator = self.Build_Generator(img_size,noise_size,classes)
        else:
            self.Generator = self.Build_Generator(img_size,noise_size)
        self.Discriminator=self.Build_Discriminator(img_size,output_shape)


        inputs=[]
        noise_input=layers.Input((noise_size,))
        inputs.append(noise_input)
        if self.isUseLabel:
            label_input=layers.Input((classes,))
            inputs.append(label_input)
        generated=self.Generator(inputs)
        discriminated=self.Discriminator(generated)
        self.Discriminator.trainable = False
        self.Model=Model(inputs,discriminated)

        self.info()
        self.Discriminator.trainable = False
        self.Model.compile(loss=self.G_loss, optimizer=self.G_optimizer)

        self.Discriminator.trainable = True
        self.Discriminator.compile(loss=self.D_loss, optimizer=self.D_optimizer, metrics=["accuracy"])

        epoch=self.epoch
        batch=self.batch

        for i in range(epoch):
            print("epoch:", i+1, "/", epoch)
            mask = np.random.randint(0, x_train.shape[0], batch)
            noise = np.random.normal(0, 1, (batch, 100))
            sup_imgs = x_train[mask]
            category = y_train[mask]
            generate_mask = to_categorical(np.random.randint(0, 10, batch))
            gen_imgs = self.Generator.predict([noise, generate_mask])
            d_loss_real = self.Discriminator.train_on_batch([sup_imgs], [np.ones((batch)),category])
            d_loss_fake = self.Discriminator.train_on_batch([gen_imgs], [np.zeros((batch)),generate_mask])

            input_noise = np.random.normal(0, 1, (batch, 100))
            g_loss = self.Model.train_on_batch([input_noise, generate_mask], [np.ones((batch)),generate_mask])


            print("discriminator real loss:", d_loss_real)
            print("discriminator fake loss:", d_loss_fake)
            print("generator loss:", g_loss)
            if i%100==0:
                if self.isUseLabel:
                    self.generate_pic_with_label(img_size,noise_size,filename=str(i//100))
                else:
                    self.generate_pic(img_size,noise_size,filename=str(i//100))

    def mnist_train(self):
        self.Training(mnist.load_data(),output_shape=[1,10])

    def generate_pic(self,img_size,noise_size,filename="test"):
        noise = np.random.normal(0, 1, (100, noise_size))
        gen_imgs = self.Generator.predict(noise)
        x, y, z = img_size
        if z==1:
            gen_imgs=gen_imgs.reshape(-1,x,y)
            img = np.zeros((10 * x, 10 * y))
        else:
            img = np.zeros((10 * x, 10 * y, z))
        gen_imgs=gen_imgs*255
        for i in range(10):
            for j in range(10):
                img[i * x:(i + 1) * x, j * y:(j + 1) * y] = gen_imgs[i * 10 + j]
        Image.fromarray(img.astype(np.uint8)).save("./GAN_pic/DCGAN_{}.png".format(filename))
    def generate_pic_with_label(self,img_size,noise_size,filename="test"):
        noise = np.concatenate([np.random.normal(0, 1, (10, 100))] * 10)
        label = to_categorical(np.concatenate([[i] * 10 for i in range(10)]))
        gen_imgs = self.Generator.predict([noise, label])
        x, y, z = img_size
        if z == 1:
            gen_imgs = gen_imgs.reshape(-1, x, y)
            img = np.zeros((10 * x, 10 * y))
        else:
            img = np.zeros((10 * x, 10 * y, z))
        gen_imgs = gen_imgs * 255
        for i in range(10):
            for j in range(10):
                img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = gen_imgs[i * 10 + j]
        Image.fromarray(img.astype(np.uint8)).save("./GAN_pic/DCGAN_{}.png".format(filename))

    def info(self):
        print("UseLabel:",self.isUseLabel)
        print("epoch:",self.epoch)
        print("batch:",self.batch)
        print()
        print("Generator summary")
        self.Generator.summary()
        print()
        print("Discriminator summary")
        self.Discriminator.summary()
        print()
        print("Model summary")
        self.Model.summary()