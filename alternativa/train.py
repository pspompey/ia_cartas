import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import applications
import matplotlib.pyplot as plt

# vgg=applications.vgg16.VGG16()

# cnn=Sequential()
# for capa in vgg.layers:
#     cnn.add(capa)

# cnn.pop()

# for layer in cnn.layers:
#     layer.trainable=False
# cnn.add(Dense(6,activation='softmax'))
# cnn.summary()


# def modelo():
#     vgg=applications.vgg16.VGG16()
#     cnn=Sequential()
#     for capa in vgg.layers:
#         cnn.add(capa)
#     cnn.layers.pop()
#     for layer in cnn.layers:
#         layer.trainable=False
#     cnn.add(Dense(6,activation='softmax'))
    
#     return cnn

K.clear_session()

data_entrenamiento = './tmp/train'
data_validacion = './tmp/test'

"""
Parameters
"""
epocas=200
longitud, altura = 100, 100
batch_size = 32
validation_steps = 300
clases = 48
lr = 0.0004


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

##CREAR LA RED VGG16

cnn = Sequential()

cnn.add(Flatten())
cnn.add(Dense(256, activation='sigmoid'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

# cnn=modelo()

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(learning_rate=lr),
            metrics=['accuracy'])

history1 = cnn.fit(
    entrenamiento_generador,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

# history2 = cnn.fit(
#     entrenamiento_generador,
#     steps_per_epoch=5,
#     epochs=epocas,
#     validation_data=validacion_generador,
#     validation_steps=150)

# history3 = cnn.fit(
#     entrenamiento_generador,
#     steps_per_epoch=20,
#     epochs=epocas,
#     validation_data=validacion_generador,
#     validation_steps=600)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
cnn.summary()


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = cnn.evaluate(validacion_generador, batch_size=batch_size)
print("test loss, test acc:", results)

# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['history1'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['history1'], loc='upper left')
plt.show()