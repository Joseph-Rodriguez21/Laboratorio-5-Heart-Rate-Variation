import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np

# Parámetros de adquisición 
sample_rate = 1000        # Frecuencia de muestreo en Hz
duration_minutes = 2 # Duración de la adquisición en minutos
duration_seconds = duration_minutes * 60 # Duración en segundos
num_samples = int(sample_rate * duration_seconds) # Número total de muestras

with nidaqmx.Task() as task:
    # Configuramos canal a utilizar
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    
    # Adquisición finita de muestras
    task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples
    
    )
    
    task.start()
    task.wait_until_done(timeout=duration_seconds + 10)
    data = task.read(number_of_samples_per_channel = num_samples)
    
# Crear un eje de tiempo para la gráfica
time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
with open("datos_señal1.txt", "w") as archivo_txt:
    archivo_txt.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        archivo_txt.write(f"{t:.6f}\t{v:.6f}\n")
        
        