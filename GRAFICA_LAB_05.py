import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import nidaqmx
from scipy.signal import butter, filtfilt, windows
from scipy.fftpack import fft, fftfreq
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind, norm

# Cargamos los datos del archivo capturados y guardados anteriormente
señal_EMG = np.loadtxt("datos_señaljoseph.txt", skiprows = 1)
tiempo = señal_EMG[:, 0]
voltaje = señal_EMG[:, 1]
fs = 100
duracion2=len(voltaje)/fs
print("duracion de la señal", duracion2)
fs = 1000 # Frecuencia de muestreo (Hz)
lowcout = 20 # Frecuencia mínima de corte (Hz)
highcut = 1000 # Frecuencia máxima de corte (Hz)

# 1. Graficar la señal original
plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,120)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(115,120)
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(tiempo, voltaje, label="Señal ECG", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (v)")
plt.title("Señal ECG")
plt.xlim(0,5)
plt.legend()
plt.show()

