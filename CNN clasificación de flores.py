# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:13:33 2020

Clasificación de flores

@author: José Manuel Ruvalcaba Rascón
"""
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
import math 
import glob
import shutil
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
#Estructura:
# flower_photos
# |__ diasy
# |__ dandelion
# |__ roses
# |__ sunflowers
# |__ tulips

#Creamos nuevas carpetas que divida el 80 como de entrenamiento y 20 de test
for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  num_train = int(round(len(images)*0.8))
  train, val = images[:num_train], images[num_train:]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))
    
direc_ent = os.path.join(base_dir, 'train')
direc_val = os.path.join(base_dir, 'val')

#Crear Batch y Img_shape
BATCH_SIZE=100
IMG_SHAPE=150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
# #Vuelta horizontal
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

datos_gen_ent = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=direc_ent,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE))
#Unicamente agregamos un horizontal_flip=True a la imagen generada con el imagedatagenerator
#Toma un batch de las imagenes a entrenar les aplica el giro horizontal



#Rotando la imagen
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

datos_gen_ent = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=direc_ent,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE, IMG_SHAPE))
#Nuevamente generamos un batch aleatorio de imagenes a las cuales les aplicamos
#una rotación de 45 grados



#Aplicando Zoom
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

datos_gen_ent = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=direc_ent,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE, IMG_SHAPE))
# De nuevo tomamos la imagen de manera aleatoria generando un zoom de 0.5 a la imagen

#Generamos las imágenes de entrenamiento
gen_img_ent= ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.15,
      height_shift_range=0.15,
      zoom_range=0.5,
      horizontal_flip=True,
      fill_mode='nearest')

datos_gen_ent = gen_img_ent.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=direc_ent,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='sparse')

#Para la validación:
gen_img_val=ImageDataGenerator(rescale=1./255)

datos_gen_val=gen_img_val.flow_from_directory(batch_size=BATCH_SIZE,
                                              directory=direc_val,
                                              shuffle=True,
                                              target_size=(IMG_SHAPE,IMG_SHAPE),
                                              class_mode="sparse")

#Creamos la CNN:
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5,activation="softmax")
    ])

#Compilamos el modelo:
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


#Entrenamiento:
epochs=100
history=model.fit_generator(datos_gen_ent,
                    steps_per_epoch=int(math.ceil(len(datos_gen_ent)/BATCH_SIZE)),
                    epochs=epochs,
                    validation_data=datos_gen_val,
                    validation_steps=int(math.ceil(len(datos_gen_val)/BATCH_SIZE)),
                    )

#Resultados
acc = history.history['acc']
val_acc = history.history['val_acc']
#Diccionarios con esos valores
#['accuracy', 'loss', 'val_accuracy', 'val_loss']
#Para la versión utilizada es ['acc', 'loss', 'val_acc', 'val_loss']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
plt.show()

#Ejercicio:
"""
Crear un nuevo CNN con diferentes parámetros y aplicar diferentes
modificaciones a la imagen como el aumento o disminución del brillo
"""