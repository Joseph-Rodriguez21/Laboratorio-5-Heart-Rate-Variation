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

import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
