# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:44:38 2020

@author: Usuario
"""
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import matplotlib.pyplot as plt
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

#Cargamos las imágenes
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
zip_dir_base = os.path.dirname(zip_dir)

# Estructura de los datos:
# cats_and_dogs_filtered
# |__ train
#     |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ...]
#     |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
# |__ validation
#     |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ...]
#     |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
#Asignamos los directiorios de los perros y gatos de entrenamiento y validación
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

#Analizamos la cantidad de perros y gatos de entrenamiento y validación
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

#Cantidad de imagenes y la forma de las mismas:
BATCH_SIZE = 100  # Número de ejemplos a procesar
IMG_SHAPE  = 150  # Las imagenes serán de 150x150 píxeles

#Generadores de imagenes de entrenamiento y validación
train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data1

#Flow_from_directory generará las imagenes tomando paquetes de 150, para entrenamiento al azar y le hará el reshape automático
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')

#Vemos algunas imágenes:
sample_training_images, _ = next(train_data_gen) 
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])  # Plot images 0-4


#Modelo:
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2,activation="softmax")
])
#Recordar que el primer número hace referencia a cuántas salidas convolucionadas habrá
#Es decir la cantidad de kernels o filtros aplicados.
#Las maxpooling reducen el tamaño de las imagenes que ya tienen kernel aplicado
#Podemos utilizar en vez de softmax, utilizar sigmoid pero cambia el tipo de salida a una binaria 
#por lo que el output tiene solo una neurona por lo que cambia de 2  a 1.

#Compilamos el modelo
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy", 
              #loss='binary_crossentropy',
#Utilizamos binary para clasificación con salida sigmoide
              metrics=['accuracy'])
#Vemos las capas del modelo
model.summary()

#Entrenamos el modelo
EPOCHS = 100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
#Como las imagenes vienen de imagegenerator utilizamos fit_generator en vez de fit

#Visualizamos el resultado
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

#El modelo tiene overfitting!! por el numero de epocas


