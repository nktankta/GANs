from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from ResBlock import ResBlock

model_input = layers.Input((32,32,3))
x = layers.Conv2D(64,kernel_size=1)(model_input)
filter=16
for i in [3,3,3,3]:
    for j in range(i-1):
        x = ResBlock(filter,activation="relu")(x)
    filter*=2
    if filter!=512:
        x = ResBlock(filter,strides=2,activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(10,activation="softmax")(x)

model=Model(model_input,x)
model.compile(loss="categorical_crossentropy",metrics=["acc"])
model.summary()

(x_train, y_train), (x_test,y_test)=cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

params={
    "rotation_range":20,
    "width_shift_range":0.2,
    "horizontal_flip":True,
    "height_shift_range":0.2,
    "shear_range":5,
    "zoom_range":0.3,
    "channel_shift_range":5,
    "brightness_range":[0.5,1.0],
    "rescale":1./255
}
datagen=image.ImageDataGenerator(**params)
testgen=image.ImageDataGenerator(rescale=1./255)
train_iter=datagen.flow(x=x_train,y=y_train,batch_size=128)
test_iter=testgen.flow(x=x_test,y=y_test,batch_size=128)

hist=model.fit_generator(train_iter,steps_per_epoch=200,epochs=100,validation_data=test_iter)

print(hist)