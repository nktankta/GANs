from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import pickle

def generate_pic(generator, filepath="test",filename="test",isUsePicFolder=True):
    if isUsePicFolder:
        filepath="GAN_pic/{0}".format(filepath)
    img_size = tuple(generator.output.shape[1:].as_list())
    noise_size = generator.input.shape[1:].as_list()[0]
    noise = np.random.normal(0, 1, (100, noise_size))
    gen_imgs = generator.predict(noise)
    x, y, z = img_size
    if z == 1:
        gen_imgs = gen_imgs.reshape(-1, x, y)
        img = np.zeros((10 * x, 10 * y))
    else:
        img = np.zeros((10 * x, 10 * y, z))
    if np.min(gen_imgs)>=0:
        gen_imgs = gen_imgs * 255
    else:
        gen_imgs = gen_imgs*127.5+127.5
    for i in range(10):
        for j in range(10):
            img[i * x:(i + 1) * x, j * y:(j + 1) * y] = gen_imgs[i * 10 + j]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    Image.fromarray(img.astype(np.uint8)).save("{0}/{1}.png".format(filepath,filename))


def generate_pic_with_label(generator,filepath="test", filename="test",isUsePicFolder=True):
    if isUsePicFolder:
        filepath="GAN_pic/{0}".format(filepath)
    img_size = tuple(generator.output.shape[1:].as_list())
    inputs=generator.inputs
    noise_size = inputs[0].shape[1:].as_list()[0]
    classes = inputs[1].shape[1:].as_list()[0]
    noise = np.concatenate([np.random.normal(0, 1, (10, noise_size))] * classes)
    label = to_categorical(np.concatenate([[i] * 10 for i in range(classes)]))
    gen_imgs = generator.predict([noise, label])
    x, y, z = img_size
    if z == 1:
        gen_imgs = gen_imgs.reshape(-1, x, y)
        img = np.zeros((10 * x, 10 * y))
    else:
        img = np.zeros((10 * x, 10 * y, z))
    if np.min(gen_imgs) >= 0:
        gen_imgs = gen_imgs * 255
    else:
        gen_imgs = gen_imgs * 127.5 + 127.5
    for i in range(classes):
        for j in range(10):
            img[i * x:(i + 1) * x, j * y:(j + 1) * y] = gen_imgs[i * 10 + j]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    Image.fromarray(img.astype(np.uint8)).save("{0}/{1}.png".format(filepath,filename))

def DataPreprocess(args,useTanh=False):
    (x_train, y_train), (x_test,y_test) = args
    if useTanh:
        x_train = x_train.astype(np.float32) / 127.5 -1
        x_test = x_test.astype(np.float32) / 127.5 -1
    else:
        x_train= x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    if x_train.shape[-1]!=3:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test,y_test)

def save_func(models,modelpath, savemodel=False):
    if type(models) is not list:
        models=[models]
    def save_model(version):
        if savemodel:
            for model in models:
                model.save("{0}/{1}_{2}.h5".format(modelpath,model.name,version))
        else:
            for model in models:
                model.save_weights("{0}/{1}_{2}.hdf5".format(modelpath, model.name, version))
    return save_model

def load_weights(models,path,version):
    if type(models) is not list:
        models = [models]
    for model in models:
        model.load_weights("{0}/{1}_{2}.hdf5".format(path,model.name,version))

def draw_pic(img):
    shape = img.shape
    w = min(shape[0], 10)
    h = (shape[0] - 1) // 10 + 1
    draw_img = np.zeros((w * shape[1], h * shape[2], shape[3]), dtype=np.uint8)
    if np.max(img[0]) <= 1:
        if np.min(img[0]) < 0:
            img = img * 127.5 + 127.5
        else:
            img = img * 255
    for i in range(w):
        c = img[np.arange(i, shape[0], 10)].transpose((1, 0, 2, 3)).reshape((shape[1], -1, 3))
        draw_img[i * shape[1]:(i +1)* shape[1] , :c.shape[1]] = c
    plt.imshow(draw_img)

def save_pic(img,path,name):
    shape = img.shape
    w = min(shape[0], 10)
    h = (shape[0] - 1) // 10 + 1
    draw_img = np.zeros((w * shape[1], h * shape[2], shape[3]), dtype=np.uint8)
    if np.max(img[0])<=1:
        if np.min(img[0])<0:
            img=img*127.5+127.5
        else:
            img=img*255
    for i in range(w):
        c = img[np.arange(i, shape[0], 10)].transpose((1, 0, 2, 3)).reshape((shape[1], -1, 3))
        draw_img[i * shape[1]:(i +1)* shape[1] , :c.shape[1]] = c
    Image.fromarray(draw_img.astype(np.uint8)).save("{0}/{1}.png".format(path, name))

def save(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)

def load(name):
    with open(name, "rb") as f:
        return pickle.load(f)
