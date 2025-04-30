# Laboratorio-5-Heart-Rate-Variation

- Introducción:
  
La variabilidad de la frecuencia cardíaca (HRV, por sus siglas en inglés) es un indicador fundamental del estado del sistema nervioso autónomo, ya que permite evaluar el equilibrio entre la actividad simpática y parasimpática que regula el ritmo cardíaco. En esta práctica de laboratorio, se abordará el análisis de la HRV mediante la aplicación de la transformada wavelet, una herramienta eficaz para observar la evolución temporal y frecuencial de señales biológicas. Esta técnica permite detectar cambios sutiles en la actividad cardíaca que no son evidentes mediante métodos tradicionales del dominio del tiempo. El conocimiento y análisis de la HRV no solo es relevante en contextos clínicos, sino también en campos como el deporte, la psicología y el monitoreo de estrés.

- Metodología, desarrollo y análisis:

Primero se descargaron los correspondientes programas para la adquisicion de datos DAQ NI USB, en el cual instalamos una aplicacion y ampliacion en Python, luego se conectan los electrodos al amplificador y al sistema DAQ. Luego realizamos la correspondiente investigación de acuerdo a la guia de laboratorio con respecto a la actividad simpática y parasimpática del sistema nervioso autónomo, transformada wavelet y Variabilidad de la frecuencia cardiaca (HRV) medida como fluctuaciones en el intervalo R-R, y las frecuencias de interés en este análisis.

![image](https://github.com/user-attachments/assets/ab3df84d-3256-4978-8a06-895365832d51)
Fig. 1 Diagrama de flujo plan de acción.

- Montaje:

Se conectaron los electrodos al sensor y este a su vez al dispositivo NU, el cual de igual manera proporcionaba la energia suficiente para su correcto funcionamiento. Cabe recalcar que se utilizo un código parecido al de la captura de la señal de la actividad múscular desde un sensor de EMG, con cambios en la frecuencia ya que para caputrar una señal de ECG es recomendado usar desde 250 Hz a 1000 Hz, en nuestro caso utilizamos el máximo de 1000 Hz ya que nuestro objetivo es lograr identificar y análizar el comportamiento de los picos R.

![image](https://github.com/user-attachments/assets/b26ffaf4-68fe-436c-9667-3464a90e272d)
Fig. 2 Imagen ilustrativa montaje.

Ahora, ya elegido el sujeto de prueba realizamos la captura de la señal electrocardiográfica durante 5 minutos, al principio en reposo y luego exaltado, para capturar y análizar.
Utilizamos el siguiente código para capturar la señal en tiempo real y guardar los datos en un archivo .txt para realizar su correspondiente filtro y análisis.

```python 
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
```
En este apartado llamamos e importamos las diferentes librerias con las que vamos trabaja, en este caso nidaqx para interactuar con el hardware de adquisició NI DAQ, por otro
lado importamos para definir el tipo de adquisición (finita, continua, etc) y por último para manejar operaciones numéricas, en este caso como crear el eje de tiempo.

```python
# Parámetros de adquisición 
sample_rate = 1000        # Frecuencia de muestreo en Hz
duration_minutes = 5  # Duración de la adquisición en minutos
duration_seconds = duration_minutes * 60 # Duración en segundos
num_samples = int(sample_rate * duration_seconds) # Número total de muestras
```
Se definen la frecuencia de muestreo (1000 Hz) y la duración de la adquisición (5 minutos). A partir de estos datos, se calcula el número total de muestras necesarias para capturar la señal sin pérdida de información.

```python
with nidaqmx.Task() as task:
    # Configuramos canal a utilizar
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    
    # Adquisición finita de muestras
    task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples
    
    )
```
Dentro de un bloque with, se crea una tarea DAQ para garantizar que se inicie y cierre correctamente. Se configura el canal analógico de entrada (Dev3/ai0) y se define una adquisición finita del número exacto de muestras con el temporizador interno.
```python
    task.start()
    task.wait_until_done(timeout=duration_seconds + 10)
    data = task.read(number_of_samples_per_channel = num_samples)
```
Una vez finalizada la captura, los datos se leen en un arreglo. Se crea un eje de tiempo paralelo usando numpy.linspace que representa cada instante de muestra, desde 0 hasta el final de la adquisición.

```python
# Crear un eje de tiempo para la gráfica
time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
with open("datos_señal1.txt", "w") as archivo_txt:
    archivo_txt.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        archivo_txt.write(f"{t:.6f}\t{v:.6f}\n")
```
Finalmente, los datos de tiempo y voltaje se guardan en un archivo .txt, separados por tabulaciones y con encabezado, listos para ser graficados en el siguiente apartado.

Ahora ya teniendo listo el código para el filtrado y la adquisición de datos, procedemos a utilizar el siguiente código para gráficar los datos filtrados obtenidos anteriormente, de tal manera podemos identificar de manera clara los diferentes intervalos y picos. Mediante este código gráficamos toda la señal y luego más a fondo al principio en reposo y luego en excitación. Con esto podemos analizar el comportamiento de los picos y sus diferencias. 

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import nidaqmx
from scipy.signal import butter, filtfilt, windows
from scipy.fftpack import fft, fftfreq
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind, norm
```
