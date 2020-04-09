# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:20:05 2020

@author: Usuario
"""

import tensorflow as tf
tf.enable_eager_execution()
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow_hub as hub

import numpy as np
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)

(training_set, validation_set), dataset_info = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)
num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0
num_validation_examples = 0

for example in training_set:
  num_training_examples += 1

for example in validation_set:
  num_validation_examples += 1

print('Número de clases: {}'.format(num_classes))
print('Número de ejemplos: {}'.format(num_training_examples))
print("Número de ejemplos de validación: {} \n".format(num_validation_examples))

#Formato de las imagnes
IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

URL="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor=hub.KerasLayer(
    URL,
    input_shape=(IMAGE_RES,IMAGE_RES,3))
feature_extractor.trainable=False

model=tf.keras.models.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(5,activation="softmax")])

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(num_classes,activation="softmax")
])

model.summary()
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 6

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs_range=range(EPOCHS)
class_names=np.array(dataset_info.features["label"].names)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión Validación')
plt.legend(loc='lower right')
plt.title('Precisión del Entrenamiento y Validación')

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Error del Entrenamiento')
plt.plot(epochs_range, val_loss, label='Error de Validación')
plt.legend(loc='upper right')
plt.title('Error de Entrenamiento y Validación')
plt.show()

class_names = np.array(dataset_info.features['label'].names)

print(class_names)
#Predicciones:
image_batch, label_batch = next(iter(train_batches))


image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
#Guardamos el modelo:
export_path="C:/ModelosML/ModeloTransferFlores.h5"
model.save(export_path)
#Cargar modelo:
# reloaded=tf.keras.models.load_model(export_path,
                                    # custom_objects={"KerasLayer":hub.KerasLayer})
#Ejercicio: Hacer lo mismo pero con el modelo Inception V3

