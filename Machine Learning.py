# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:26:29 2020

@author: Usuario
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf
import numpy as np
#import logging
#logger = tf.get_logger()
#logger.setLevel(logging.ERROR) #Indica a TensorFlow que carge sólo mensajes de error
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
 #Input_shape dice cómo es la forma de entrada en este caso el array
 #Units indica la cantidad de Neuronas que tendrá esa capa, la capa se inicializa como Dense dado que en todas las
 #neuronas habrá entradas, en este caso solo es 1 neurona y por tanto 1 entrada.
l0=tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([
  l0
])
 #Se crea el modelo Secuencial, que toma una lista de las capas a utilizar.
 #Input_shape dice cómo es la forma de entrada en este caso el array
 #Units indica la cantidad de Neuronas que tendrá esa capa, la capa se inicializa como Dense dado que en todas las
 #neuronas habrá entradas, en este caso solo es 1 neurona y por tanto 1 entrada.
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
 #Se inicia el compilado del modelo, el cual recibe cómo se medirá el error 
 #Con la función loss, en este caso con el mean squared error
 #Se optimiza para reducir a lo máximo el error utilizando el optimizador Adam
 #Adam recibe como 0.1 el learning rate que es para mejorar la velocidad y 
 #precisión de aprendizaje.

"""Entrenamiento del modelo"""
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
#Iniciamos que el modelo realice la estimación con el model.fit
#Tomando como entrada las coordenadas de los puntos en x y y, en este caso
#los grados celsius respecto a los farenheit.
#El ciclo de aprendizaje se realiza 500 veces o 500 epocas (epochs)
#Verbose indica cuánta salida produce el método
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
#Se grafica el objeto de history el cual representa el error del modelo
#A medida que avanza se ralentiza el proceso
while True:
    entrada=float(input("Ingrese la entrada a estimar: "))
    print("La estimación del modelo es:")
    print(model.predict([entrada]))#El modelo realiza una predicción para el valor de 100
    # print("These are the layer variables: {}".format(l0.get_weights())) 
    y=input("Desea salir? [y]/[n]")
    if y=="y" or y=="Y": break

##Los pesos que se obtienen corresponden a 1.8 y 32 como y=mx+b
##Probamos un modelo diferente que tenga una capa con 4 neuronas y luego reciba un solo dato
##La siguiente utilizará 4 neuronas más y las dará a la neurona de salida
#l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
#l1 = tf.keras.layers.Dense(units=4)
#l2 = tf.keras.layers.Dense(units=1)
#model = tf.keras.Sequential([l0, l1, l2]) #El modelo son las capas en orden
#model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)) 
#model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
#print("Finished training the model")
#print(model.predict([100.0]))
#print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
#print("These are the l0 variables: {}".format(l0.get_weights()))
#print("These are the l1 variables: {}".format(l1.get_weights()))
#print("These are the l2 variables: {}".format(l2.get_weights()))