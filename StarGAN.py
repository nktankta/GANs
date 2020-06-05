import tensorflow as tf
from tensorflow.keras import models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image
import pickle
import tensorflow.python.debug as tf_debug
from tensorflow.python.debug import has_inf_or_nan
from tensorflow.keras.regularizers import l2

tf.compat.v1.disable_eager_execution()
debug=False

def set_debugger_session():
    sess = tf.compat.v1.keras.backend.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    tf.compat.v1.keras.backend.set_session(sess)
if debug:set_debugger_session()

picpath= "Stargan/pic"
modelpath="Stargan/models"

def load(name):
    with open(name,"rb") as f:
        return pickle.load(f)
def resBlock():
    def f(x):
        shape=x.shape
        shortcut=x
        x = layers.Conv2D(shape[3],kernel_size=3,padding="same")(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(shape[3], kernel_size=3, padding="same")(x)
        x = tfa.layers.InstanceNormalization()(x)
        return layers.Add()([x,shortcut])
    return f

def downSampling(*args,**kwargs):
    def f(x):
        x = layers.Conv2D(*args,**kwargs)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        return x
    return f

def upSampling(*args,**kwargs):
    def f(x):
        x = layers.UpSampling2D((2,2))(x)
        x = layers.Conv2D(*args,**kwargs)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        return x
    return f

def build_generator(img_shape,isSkip=False):
    c_size=3
    c_output=(img_shape[0],img_shape[1],c_size)
    inp_img=layers.Input(img_shape)
    inp_c=layers.Input((c_size,))

    c = layers.RepeatVector(img_shape[0]*img_shape[1])(inp_c)
    c = layers.Reshape(c_output)(c)

    x=layers.Concatenate()([inp_img,c])

    x1=downSampling(64,kernel_size=7,padding="same")(x)
    x2=downSampling(128,kernel_size=5,padding="same",strides=2)(x1)
    x3=downSampling(256,kernel_size=5,padding="same",strides=2)(x2)

    x=x3
    for i in range(6):
        x=resBlock()(x)

    if isSkip: x = layers.Concatenate()([x,x3])
    x = upSampling(128,kernel_size=5,padding="same")(x)
    if isSkip: x = layers.Concatenate()([x,x2])
    x = upSampling(64, kernel_size=5, padding="same")(x)
    if isSkip: x = layers.Concatenate()([x,x1])
    x = layers.Conv2D(3, kernel_size=7, padding="same",activation="tanh")(x)

    model=Model([inp_img,inp_c],x,name="generator")
    return model

def build_discriminator(img_shape):
    discriminator_inp=layers.Input(img_shape)
    filter=32
    repeat=5

    x=discriminator_inp

    for i in range(repeat):
        x=layers.Conv2D(filter,kernel_size=3,strides=1 if i==0 else 2,padding="same",use_bias=False)(x)
        x=layers.LeakyReLU(0.01)(x)
        filter*=2

    src_out=layers.Conv2D(1,kernel_size=3,padding="same",use_bias=False)(x)
    #src_out=layers.Flatten()(src_out)

    kernel_size=img_shape[0]//(2**(repeat-1))

    cls_out=layers.Conv2D(3,kernel_size=kernel_size,padding="valid",use_bias=False)(x)
    cls_out=layers.Flatten()(cls_out)
    cls_out=layers.Activation("tanh")(cls_out)

    model=Model(discriminator_inp,[src_out,cls_out],name="discriminator")
    return model

def reconstruction_loss(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))


def build_discriminator_updatefunction(generator,discriminator,grad_penalty_weight=10,isOneCenter=True):
    g_inp=layers.Input(generator.inputs[0].shape[1:])
    cls_input=layers.Input(generator.inputs[1].shape[1:])
    f_img=generator([g_inp,cls_input])
    def noise_dis(x):
        #x=layers.GaussianNoise(0.2)(x)
        return discriminator(x)
    f_src_out,f_cls_out=noise_dis(f_img)

    r_img=layers.Input(discriminator.input.shape[1:])
    r_src_out,r_cls_out=noise_dis(r_img)

    e_inp=K.placeholder(shape=(None,1,1,1))
    a_img=layers.Input(discriminator.input.shape[1:],
                       tensor=e_inp*f_img+(1-e_inp)*r_img)
    a_out,_=noise_dis(a_img)

    loss_real=K.mean(r_src_out)
    loss_fake=K.mean(f_src_out)

    loss_cls=K.mean(K.abs(cls_input-r_cls_out))

    grads=K.gradients(a_out,[a_img])[0]
    grads_sqr=K.square(grads)
    grads_sqr_sum=K.sum(grads_sqr,axis=np.arange(1,len(grads_sqr.shape)))
    grad_l2_norm=tf.sqrt(grads_sqr_sum)
    if isOneCenter:
        grad_penalty=K.mean(K.square(1-grad_l2_norm))
    else:
        grad_penalty=K.mean(grad_l2_norm)
    loss=loss_fake - loss_real + loss_cls + grad_penalty_weight*grad_penalty

    training_updates=Adam(lr=1e-4,beta_1=0.5).get_updates(loss,discriminator.trainable_weights)
    train_func=K.function([r_img,g_inp,cls_input,e_inp],
                          [loss_real,loss_fake,loss_cls],
                          training_updates)
    return train_func

def build_generator_updatefunction(generator,discriminator):
    g_inp=layers.Input(generator.inputs[0].shape[1:])
    cls_inp=layers.Input(generator.inputs[1].shape[1:])
    random_cls_inp=layers.Input(generator.inputs[1].shape[1:])

    img=generator([g_inp,random_cls_inp])
    src_out,cls_out=discriminator(img)
    reconstruct_img=generator([img,cls_inp])

    id_img=generator([g_inp,cls_inp])

    rec_loss=reconstruction_loss(g_inp,reconstruct_img)

    src_loss=K.mean(src_out)
    cls_loss=K.mean(K.abs(cls_out-random_cls_inp))

    id_loss=reconstruction_loss(g_inp,id_img)

    loss=-src_loss+cls_loss+10*rec_loss+1*id_loss

    training_updates=Adam(lr=1e-4,beta_1=0.5).get_updates(loss,generator.trainable_weights)
    train_func=K.function([g_inp,cls_inp,random_cls_inp],
                          [src_loss,cls_loss,rec_loss],
                          training_updates)
    return train_func

def save_picture(generator,img,cls,name,size=(10,10)):
    shape=img.shape
    idx = np.random.randint(0, img.shape[0], size[0]*size[1])
    using_img = img[idx]
    idx = np.random.randint(0, img.shape[0], size[0]*size[1])
    using_cls = cls[idx]

    gen_img=generator.predict([using_img,using_cls]).reshape((size[0],size[1],shape[1],shape[2],shape[-1]))
    gen_img = gen_img.transpose((0,2,1,3,4)).reshape((size[0]*shape[1],size[1]*shape[2],shape[-1]))*127.5+127.5

    using_img=using_img.reshape((size[0],size[1],shape[1],shape[2],shape[3]))
    using_img = using_img.transpose((0, 2, 1, 3, 4)).reshape((size[0] * shape[1], size[1] * shape[2], shape[-1])) * 127.5+127.5

    one=np.ones((1,1,shape[1],shape[2],1))
    using_cls=using_cls.reshape((size[0],size[1],1,1,shape[-1]))
    using_cls_pic=one*using_cls
    using_cls_pic=using_cls_pic.transpose((0,2,1,3,4)).reshape((size[0]*shape[1],size[1]*shape[2],shape[-1]))*127.5+127.5
    Image.fromarray(using_img.astype(np.uint8)).save("{0}/{1}_inp.png".format(picpath,name))
    Image.fromarray(using_cls_pic.astype(np.uint8)).save("{0}/{1}_cls.png".format(picpath,name))
    Image.fromarray(gen_img.astype(np.uint8)).save("{0}/{1}_gen.png".format(picpath,name))

def save_model(generator,discriminator,version,savemodel=False):
    if savemodel:
        generator.save("{0}/generator_{1}.h5".format(modelpath, version))
        discriminator.save("{0}/discriminator_{1}.h5".format(modelpath, version))
    else:
        generator.save_weights("{0}/generator_{1}.hdf5".format(modelpath, version))
        discriminator.save_weights("{0}/discriminator_{1}.hdf5".format(modelpath, version))

def reconstruction_error(generator,img,cls):
    id_img=generator.predict([img,cls])
    return np.mean(np.abs(img-id_img),axis=(1,2,3))

def train(generator,discriminator,imgs,clses,epochs=1000,batch_size=10,update_weight=5,save_span=100,version=0):
    generator.trainable = True
    discriminator.trainable = False
    g_update=build_generator_updatefunction(generator,discriminator)

    generator.trainable = False
    discriminator.trainable = True
    d_update=build_discriminator_updatefunction(generator,discriminator,grad_penalty_weight=1,isOneCenter=False)

    for epoch in range(epochs):
        print("epoch: {0}/{1}".format(epoch+1,epochs))
        for i in range(update_weight):
            idx=np.random.randint(0,clses.shape[0],batch_size)
            using_img=imgs[idx]
            using_cls=clses[idx]
            epsilon=np.random.uniform(size=(batch_size,1,1,1))
            loss_real,loss_fake,loss_cls=d_update([using_img,using_img,using_cls,epsilon])
            print("loss_real:",loss_real)
            print("loss_fake:",loss_fake)
            print("loss_cls:",loss_cls)
        idx=np.random.randint(0,clses.shape[0],batch_size)
        random_cls=clses[idx]
        loss_gen,loss_cls,loss_rec=g_update([using_img,using_cls,random_cls])
        print("loss_gen:",loss_gen)
        print("loss_cls:",loss_cls)
        print("loss_rec:",loss_rec)
        if (epoch+1)%save_span==0:
            version+=1
            print("version:",version)
            save_picture(generator,imgs,clses,version)
            save_model(generator,discriminator,version)


if __name__ == '__main__':
    #img=load("animeface.pkl")
    #cls=load("haircolor.pkl")
    #cls=load("hair_predict_v0.pkl")
    img,cls=load("traindata.pkl")
    img=img/127.5-1
    cls=cls/127.5-1

    generator=build_generator(img.shape[1:])
    discriminator=build_discriminator(img.shape[1:])
    version=141
    if version!=0:
        v="final"
        if v=="Interrupt" or v=="final":
            generator = models.load_model("{0}/generator_{1}.h5".format(modelpath,v))
            discriminator = models.load_model("{0}/discriminator_{1}.h5".format(modelpath,v))
        else:
            generator.load_weights("{0}/generator_{1}.hdf5".format(modelpath,version))
            discriminator.load_weights("{0}/discriminator_{1}.hdf5".format(modelpath,version))

    generator.summary()
    discriminator.summary()
    try:
        train(generator,discriminator,img,cls,epochs=10000,save_span=100,version=version,update_weight=15)
    except KeyboardInterrupt as e:
        save_model(generator,discriminator,"Interrupt",savemodel=True)
    finally:
        save_picture(generator,img,cls,"final")
        save_model(generator, discriminator, "final", savemodel=True)
        save_model(generator,discriminator,"final")