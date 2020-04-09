# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:50:33 2020

@author: José Manuel Ruvalcaba Rascón
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_series(time,series,format="-",start=0,end=None,label=None):
    plt.plot(time[start:end],series[start:end],format,label=label)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
def trend(time,slope=0):
    return slope*time
def seasonal_pattern(season_time):
    """Patron arbitrario"""
    return np.where(season_time<0.4,
                    np.cos(season_time*2*np.pi),
                    1/np.exp(3*season_time))
def seasonality(time,period,amplitude=1,phase=0):
    """Repite el mismo patrón cada periodo"""
    season_time=((time+phase)%period)/period
    return amplitude*seasonal_pattern(season_time)
def white_noise(time,noise_level=1,seed=None):
    rnd=np.random.RandomState(seed)
    return rnd.randn(len(time))*noise_level

#Tiempo y estacionalidad:
tiempo=np.arange(4*365+1)
slope=0.05
baseline=10
amplitude=40
series=baseline+trend(tiempo,slope)+seasonality(tiempo,period=365,amplitude=amplitude)
noise_level=5
noise=white_noise(tiempo,noise_level,seed=42)
series+=noise
plt.figure(figsize=(10,6))
plot_series(tiempo,series)
plt.show()



#Dividimos el tiempo:
dividir_tiempo=1000 #Se dividirá en el día 1000
tiempo_entrenamiento=tiempo[:dividir_tiempo]#Tomamos los 999 días antes del día elegido
x_entrenamiento=series[:dividir_tiempo]#Tomamos las x de entrenamiento que es la serie en los 999 dias
tiempo_validacion=tiempo[dividir_tiempo:] #Tomamos los días restantes
x_validacion=series[dividir_tiempo:]#Tomamos las x restantes

#Naive Forecast:
naive_forecast=series[dividir_tiempo-1:-1] #Tomamos todos los días de validación exceptuando el día 1000 para que no se junte
plt.figure(figsize=(10,6))
plot_series(tiempo_validacion,x_validacion,label="Series")
plot_series(tiempo_validacion,naive_forecast,label="Forecast")
plt.show()

#Hacemos un zoom a partir del dia 1000 al dia 1150 para observar el desfase
plt.figure(figsize=(10,6))
plot_series(tiempo_validacion,x_validacion,start=0,end=150,label="Series")
plot_series(tiempo_validacion,naive_forecast,start=1,end=151,label="Forecast")
errors=naive_forecast-x_validacion
error_abs=np.abs(errors)
mae=error_abs.mean()
print(mae)
plt.show()

# #Tipos de errores:
# errors=forecasts-actual
# mse=np.square(errors).mean()
# mae=np.abs(errors).mean()
# mape=np.abs(errors/x_valid).mean()
#Algunas ocasiones puede ser mejor utilizar otro tipo de forecast
#Tal como es el Moving Average y la técnica de Diferenciación