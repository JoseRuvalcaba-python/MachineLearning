# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:25:37 2020

@author: Usuario
"""
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.logging.set_verbosity(tf.logging.ERROR)
keras=tf.keras
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

#Calculamo el mean squared error
mse=keras.metrics.mean_absolute_error(x_validacion,naive_forecast).numpy()

#Moving Average:
# def moving_average_forecast(series,window_size):
#     """Forecast el promedio de unos últimos valores.
#     Si el tamaño de ventana es 1 entonces es equivalente 
#     al naive forecast"""
#     forecast=[]
#     for time in range(len(series)-window_size):
#         forecast.append(series[time:time+window_size].mean())
#     return np.array(forecast)

def moving_average_forecast(series,window_size):
    """Forecast el promedio de los últimos valores.
    Si window_size=1 entonces es equivalente al naive_forecast
    Esta implementación es más rápida que el anterior."""
    mov=np.cumsum(series)
    mov[window_size:]=mov[window_size:]-mov[:-window_size]
    return mov[window_size-1:-1]/window_size
moving_avg=moving_average_forecast(series, 30)[dividir_tiempo-30:]
plt.figure(figsize=(10,6))
plot_series(tiempo_validacion,x_validacion,label="Serie")
plot_series(tiempo_validacion, moving_avg,label="Moving Average (30 días")
plt.show()
error=keras.losses.MAE(x_validacion,moving_avg)
print(error)
#Aquí utilizamos la diferenciación, tomamos el valor de un año y luego lo tomamos unos años luego
#Y sacamos la diferencia entre estos
diff_series=(series[365:]-series[:-365])
diff_time=tiempo[365:]
plt.figure(figsize=(10,6))
plot_series(diff_time,diff_series,label="Serie(t)-Serie(t-365)")
plt.show()

#Checamos para el tiempo de validacion
plt.figure(figsize=(10,6))
plot_series(tiempo_validacion, diff_series[dividir_tiempo-365:],label="Serie(t)-Serie(t-365)")
plt.show()

#Ahora que tenemos la diferencias utilizamos el moving average
diff_moving_avg=moving_average_forecast(diff_series,50)[dividir_tiempo-365-50:]
plot_series(tiempo_validacion,diff_series[dividir_tiempo-365:],label="Serie(t)-Serie(t-365")
plot_series(tiempo_validacion,diff_moving_avg,label="Moving Average de la Diferenciación")
plt.show()

#Regresamos a la estacionalidad y tendencia
diff_moving_avg_plus_past=series[dividir_tiempo-365:-365]+diff_moving_avg
plt.figure(figsize=(10,6))
plot_series(tiempo_validacion,x_validacion,label="Serie")
plot_series(tiempo_validacion,diff_moving_avg_plus_past,label="Forecast")
plt.show()
print(keras.losses.MAE(x_validacion,diff_moving_avg_plus_past))

#Volvemos a aplicar el moving average para remover el ruido:
diff_moving_avg_plus_smooth_past=moving_average_forecast(series[dividir_tiempo - 370:-359], 11) + diff_moving_avg
plt.figure(figsize=(10, 6))
plot_series(tiempo_validacion, x_validacion, label="Series")
plot_series(tiempo_validacion, diff_moving_avg_plus_smooth_past, label="Forecasts")
plt.show()
print(keras.losses.MAE(x_validacion,diff_moving_avg_plus_smooth_past))



