from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from GAN_tools import *
import numpy as np
from tensorflow.keras.preprocessing import image

def conv_bn_relu(*args,**kwargs):
    def f(x):
        x = Conv2D(*args,**kwargs)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    return f
def conc_up_conv_bn_relu(*args,**kwargs):
    def f(a):
        x = a[0]
        x = UpSampling2D((2,2))(x)
        x = Conv2D(*args,**kwargs)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x,a[1]])
        return x
    return f
def build_generator(img_shape):
    reg = lambda: l2(1e-4)
    kernel_size=3
    filter=16

    img_input = Input(img_shape)
    x1 = Conv2D      (filter    , kernel_size=kernel_size, strides=1,kernel_regularizer=reg(),padding="same",activation="relu")(img_input)
    x2 = conv_bn_relu(filter * 2, kernel_size=kernel_size, strides=2,kernel_regularizer=reg(),padding="same")(x1)
    x3 = conv_bn_relu(filter * 4, kernel_size=kernel_size, strides=2,kernel_regularizer=reg(),padding="same")(x2)
    x4 = conv_bn_relu(filter * 8, kernel_size=kernel_size, strides=2,kernel_regularizer=reg(),padding="same")(x3)
    x5 = conv_bn_relu(filter * 8, kernel_size=kernel_size, strides=2,kernel_regularizer=reg(),padding="same")(x4)

    x1 = conv_bn_relu(filter//4 , kernel_size=1, strides=1,kernel_regularizer=reg(),padding="same")(x1)
    x2 = conv_bn_relu(filter//2 , kernel_size=1, strides=1,kernel_regularizer=reg(),padding="same")(x2)
    x3 = conv_bn_relu(filter    , kernel_size=1, strides=1,kernel_regularizer=reg(),padding="same")(x3)
    x4 = conv_bn_relu(filter*2  , kernel_size=1, strides=1,kernel_regularizer=reg(),padding="same")(x4)

    x4 = conc_up_conv_bn_relu(filter * 8, kernel_size=kernel_size,kernel_regularizer=reg(),padding="same")([x5, x4])
    x3 = conc_up_conv_bn_relu(filter * 4, kernel_size=kernel_size,kernel_regularizer=reg(),padding="same")([x4, x3])
    x2 = conc_up_conv_bn_relu(filter * 2, kernel_size=kernel_size,kernel_regularizer=reg(),padding="same")([x3, x2])
    x1 = conc_up_conv_bn_relu(filter    , kernel_size=kernel_size,kernel_regularizer=reg(),padding="same")([x2, x1])

    out = Conv2D(3,kernel_size=kernel_size,padding="same",activation="tanh")(x1)
    model=Model(img_input,out,name="generator")
    return model

def build_discriminator(img_shape):
    kernel_size=5
    filter=64
    feature_size=100
    x = img_inp = Input(img_shape)
    x = Conv2D(filter,kernel_size=kernel_size,strides=2,padding="same")(x)
    x = LeakyReLU(0.2)(x)

    for i in range(3):
        filter*=2
        x = Conv2D(filter,kernel_size=kernel_size,strides=2,padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Conv2D(feature_size,kernel_size=x.shape[1],padding="valid")(x)
    feature_out = Flatten()(x)
    src_out = Dense(1,activation="sigmoid",name="fake_out")(feature_out)
    model=Model(img_inp,[feature_out,src_out],name="discriminator_feature")
    model2=Model(img_inp,src_out,name="discriminator")
    model2.compile(optimizer=Adam(lr=1e-3),loss="binary_crossentropy",metrics=["acc"])
    return model,model2

def build_compiled_model(generator,discriminator):
    generator.trainable=True
    discriminator.trainable=False

    img_inp = Input(generator.input.shape[1:])
    gen_img = generator(img_inp)
    r_feature,r_src = discriminator(img_inp)
    f_feature,f_src = discriminator(gen_img)
    model = Model(img_inp,[gen_img,r_feature,f_feature],name="combined_model")

    loss_adv=-K.mean(f_src)
    loss_con=K.mean(K.abs(img_inp-gen_img))
    loss_lat=K.mean(K.square(r_feature-f_feature))
    loss=loss_adv+40*loss_con+loss_lat

    model.add_loss(loss)
    model.compile(optimizer=Adam(lr=1e-3))
    return model

def calc_error(li):
    img,gen_img,r_feature,f_feature=li
    rec=np.mean(np.abs(img-gen_img),axis=(1,2,3))
    feat=np.mean(np.square(r_feature-f_feature),axis=1)
    return np.array([rec,feat])

def test(model,norm_img,abnorm_img):
    norms=model.predict(norm_img,batch_size=10)
    norms.insert(0,norm_img)
    norm_error=calc_error(norms)

    abnorms = model.predict(abnorm_img, batch_size=10)
    abnorms.insert(0,abnorm_img)
    abnorm_error=calc_error(abnorms)
    return norm_error,abnorm_error

def train(generator,discriminator,model,img,epochs,save_span=100,version=0):
    save_weights=save_func([generator,discriminator],"Skip_ganomaly/models")
    pic_path="Skip_ganomaly/pics"
    batch_size=10

    params = {
        "rotation_range": 5,
        "width_shift_range": 0.2,
        "horizontal_flip": True,
        "height_shift_range": 0.2,
        "shear_range": 5,
        "zoom_range": 0.1,
        "channel_shift_range": 5,
        "brightness_range": [0.7, 1.0],
        "rescale": 1. / 127.5
    }
    datagen = image.ImageDataGenerator(**params)
    split=int(img.shape[0]*0.9)
    train_iter = datagen.flow(x=img[:split], batch_size=batch_size)
    img=img/127.5-1
    test_img=img[split:]
    abnorm_img=load("abnormal_face.pkl")
    abnorm_img=abnorm_img/127.5-1
    one=np.ones((batch_size,1))
    zero=np.zeros((batch_size,1))
    for epoch in range(epochs):
        print("{0}/{1}".format(epoch+1,epochs))
        r_img=train_iter.next()-1
        f_img=generator.predict(r_img)

        r_loss=discriminator.train_on_batch(r_img, one)
        f_loss=discriminator.train_on_batch(f_img, zero)

        gen_loss=model.train_on_batch(r_img)

        print("r_loss:",r_loss)
        print("f_loss:",f_loss)
        print("gen_loss:",gen_loss)
        if (epoch+1)%save_span==0:
            version+=1
            save_weights(version)
            idx = np.random.randint(0, img.shape[0], 100)
            r_img = img[idx]
            f_img = generator.predict(r_img,batch_size=batch_size)
            save_pic(r_img,pic_path,"{}_r.png".format(version))
            save_pic(f_img,pic_path,"{}_f.png".format(version))
            norm,abnorm=test(model,test_img,abnorm_img)
            print("----norm----")
            print("max:", np.max(norm, axis=1))
            print("mean:", np.mean(norm, axis=1))
            print("min:", np.min(norm, axis=1))
            print("----abnorm----")
            print("max:", np.max(abnorm, axis=1))
            print("mean:", np.mean(abnorm, axis=1))
            print("min:", np.min(abnorm, axis=1))



def main(isTrain=True,version=0):
    img=load("normal_face.pkl")
    generator=build_generator(img.shape[1:])
    generator.summary()
    discriminator_feature,discriminator=build_discriminator(img.shape[1:])
    discriminator.summary()
    model=build_compiled_model(generator,discriminator_feature)
    if version!=0:
        load_weights([generator,discriminator],"Skip_ganomaly/models",version)
    if isTrain:
        train(generator,discriminator,model,img,epochs=5000,version=version)
    return model


if __name__ == '__main__':
    main()