# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:04:28 2020

@author: Usuario
"""
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import PIL.Image as Image
#Cargar el modelo 
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224
#Utilizamos  el modelo del url
#El modelo utiliza imgenes de 224 pixeles


model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])
#Definimos el modelo cargandolo en un modelo específico con la resolución 224 RGB

# import numpy as np
# import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper)/255.0
result = model.predict(grace_hopper[np.newaxis, ...])
#Se encuentra entre 1 y 1001 dado que hay 1000 clases
predicted_class = np.argmax(result[0], axis=-1)
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
#Se importó una imagen para realizar la predicción según la clase nombrada en el modelo
#Se intenta hacer una predicción de la imagen

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs', 
    with_info=True, 
    as_supervised=True, 
    split=['train[:80%]', 'train[80%:]'],
)
#Cargamos los modelos de perros y gatos diviendolos en 80 de entrenamiento
# y 20 de testeo

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
#Se verifica la cantidad de ejemplos y clases que hay en los entrenamientos

for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))
#Verifica las formas de las imagenes que hay en todos los ejemplos.

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label
#Se cambia la resolucióin de la imagen a ser 224x224 y se normaliza de 0 a 1
BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
#Para los ejemplos de entrenamiento y validación les aplica la función del formato y eligue la cantidad
#que tendran los batches, para los de entrenamiento los  ordena de forma aleatoria
#prefetch es para optimizar.

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()
#Toma ejemplos de entrenamientos
result_batch = model.predict(image_batch)
#Los intenta predecir
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()
result_batch = model.predict(image_batch)
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
#Hacemos distintas predicciones y se grafican con los labels asignados.


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))
feature_batch = feature_extractor(image_batch)
#Volvemos a llamar al modelo y lo asignamos a un extractor.
#Ahora no hemos asignado que el modelo se configure inmediatamente para modificar el output

print(feature_batch.shape)
#Verificamos que las imagenes son de 32 grupos y cómo se alteró por el output

feature_extractor.trainable = False
#Aquí especificamos que no se pueda entrenar el esqueleto del modelo, es decir lo congelamos

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(2,activation="softmax")
])
#Definimos un nuevo modelo que únicamente agregue la salida como la clasificadora de 
#perros y gatos, habiendo freezeado el modelo únicamente se entrenará la última capa


model.summary()
#Vemos la estructura
 
#Compilamos
model.compile(
  optimizer='adam',
  loss="sparse_categorical_crossentropy",
  metrics=['accuracy'])

#Entrenamos al modelo:
#Utilizamos pocas epocas dado que el modelo ya ha sido previamente entrenado
#y necesitamos checar sólo el output
EPOCHS = 6
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

#Graficamos el error y la validación, etc.
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
plt.show()

#Hacemos prediccions
class_names = np.array(info.features['label'].names) #definimos los nombres de las clases a cambiarlos por 0 y 1 como perro o gato
predicted_batch = model.predict(image_batch) #Tomamos un batch para aplicar la predicción
predicted_batch = tf.squeeze(predicted_batch).numpy() 
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)
plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
