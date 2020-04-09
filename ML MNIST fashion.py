# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:46:33 2020

@author: Usuario
"""
import tensorflow as tf
tf.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)
# import logging 
# logger=tf.get_logger()
# logger.setLevel(logging.ERROR)
import tqdm
import tqdm.auto
tqdm.tqdm=tqdm.auto.tqdm

#Cargamos los artículos MNIST
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
#Utilizamos train_dataset para entrenar y test para testear

#Nombramos los 10 artículos que hay
class_names=["Playera","Pantalón","Suéter",
              "Vestido","Abrigo","Sandálias/Tacon",
              "Camisa","Tenis","Bolsa","Botas"]
#Vemos cuántos datos tenemos
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Número de ejemplos de entrenamiento: {}".format(num_train_examples))
print("Número de ejemplos de Test:{}".format(num_test_examples))

#Preproceso de los datos
def normalize(images,labels):
    images=tf.cast(images,tf.float32) 
    images/=255 #El valor de los píxeles está entre 0 y 255
    return images, labels #Por lo que los normalizamos para que este entre 0 y 1
#La función map aplica la función normalize
train_dataset=train_dataset.map(normalize)
test_dataset=test_dataset.map(normalize)


#Incluimos las imagenes a la memoria cache para 
#Entrenar más rápido
train_dataset=train_dataset.cache()
test_dataset=test_dataset.cache()

# #Cargamos una imagen para visualizarla
# for image, label in test_dataset.take(1):
#     break
# image=image.numpy().reshape((28,28)) #Tomamos una imagen y la convertimos en un array de numpy 28x28
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# #Mostramos 25 imagenes de entrenamiento y vemos que nombre tienen
# plt.figure(figsize=(10,10))
# i=0
# for (image,label) in test_dataset.take(25):
#     image=image.numpy().reshape((28,28))
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image,cmap=plt.cm.binary)
#     plt.xlabel(class_names[label])
#     i+=1
# plt.show()

#Construcción del modelo
modelo=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    # tf.keras.layers.Dense(512,activation=tf.nn.relu),
    # tf.keras.layers.Dense(45,activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
#Entrenamiento del modelo
BATCH_SIZE=32
train_dataset=train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset=test_dataset.cache().batch(BATCH_SIZE)

modelo.fit(train_dataset,epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
#Steps-per-epoch se elige con el total de los datos a utilizar entre los batches 
#con la finalidad de utilizar todos los datos de entrenamiento
#math.ceil elige el numero entero más próximo.

#Evaluamos la accuracy
test_loss,test_accuracy=modelo.evaluate(test_dataset,steps=math.ceil(num_test_examples/32))
print("Precisión en los datos de testeo:",test_accuracy)

# #Hacemos predicciones 
# for test_images,test_labels in test_dataset.take(1):
#     test_images=test_images.numpy()
#     test_labels=test_labels.numpy()
#     predictions=modelo.predict(test_images)
# predictions.shape
# predictions[0]
# np.argmax(predictions[0])
# test_labels[0]

# def plot_image(i, predictions_array, true_labels, images):
#   predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
  
#   plt.imshow(img[...,0], cmap=plt.cm.binary)

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'
  
#   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array[i], true_label[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1]) 
#   predicted_label = np.argmax(predictions_array)
  
#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)

# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)

# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
# img = test_images[0]
# print(img.shape)
# img = np.array([img])
# print(img.shape)
# img=np.array([img])
# print(img.shape)
# predictions_single = modelo.predict(img)
# print(predictions_single)
# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# np.argmax(predictions_single[0])