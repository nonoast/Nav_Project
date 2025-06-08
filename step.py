import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

# Load data from CSV file
csv_file = "data/trajet_vert.csv"
df = pd.read_csv(csv_file, sep=';', skiprows=1)

# Extract sensor data
time = df.iloc[:, 0].values  # time in milliseconds
# Accelerometer data (m/s²)
ax = df.iloc[:, 1].values
ay = df.iloc[:, 2].values
az = df.iloc[:, 3].values

mag = np.sqrt(ax**2 + ay**2 + az**2)

# Function to design a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

sampling_rate = 100  # Hz (sampling frequency)
cutoff_freq = 10     # Hz (highest frequency of interest)
b, a = butter_lowpass(cutoff_freq, sampling_rate)

# Function to find local minima in a signal without inverting it
def find_minima(signal, distance=20, prominence=0.2, lower_threshold=None, upper_threshold=None):
    """
    Trouve les minimums locaux dans un signal positif sans inverser le signal.
    
    Args:
        signal: Le signal à analyser
        distance: Distance minimale entre deux minimums consécutifs
        prominence: Prominence minimale des minimums
        lower_threshold: Valeur minimale pour considérer un minimum (None = pas de limite)
        upper_threshold: Valeur maximale pour considérer un minimum (None = pas de limite)
    
    Returns:
        Array des indices des minimums détectés
    """
    # Pour un tableau signal, un indice i est un minimum local si:
    #   signal[i-1] > signal[i] < signal[i+1]
    # On crée un tableau qui sera True aux positions des minimums locaux
    minimums = np.zeros(len(signal), dtype=bool)
    
    # Ne pas traiter le premier et le dernier point
    for i in range(1, len(signal)-1):
        if signal[i-1] > signal[i] and signal[i] < signal[i+1]:
            # Vérifier si le minimum est entre les seuils
            if (lower_threshold is None or signal[i] >= lower_threshold) and \
               (upper_threshold is None or signal[i] <= upper_threshold):
                minimums[i] = True
    
    # Convertir en indices
    minimum_indices = np.where(minimums)[0]
    
    # Filtrer par prominence si demandé
    if prominence > 0 and len(minimum_indices) > 0:
        # Calculer la prominence pour chaque minimum détecté
        prominences = []
        for idx in minimum_indices:
            # Chercher le max à gauche
            left_max = signal[idx]
            for j in range(idx, max(0, idx-distance), -1):
                if signal[j] > left_max:
                    left_max = signal[j]
            
            # Chercher le max à droite
            right_max = signal[idx]
            for j in range(idx, min(len(signal), idx+distance)):
                if signal[j] > right_max:
                    right_max = signal[j]
            
            # La prominence est la différence entre le min de ces deux max et le minimum local
            prom = min(left_max, right_max) - signal[idx]
            prominences.append(prom)
        
        # Filtrer par prominence
        minimum_indices = minimum_indices[np.array(prominences) >= prominence]
    
    # Filtrer par distance
    if distance > 0 and len(minimum_indices) > 1:
        # Garder seulement les minimums qui sont à au moins 'distance' échantillons d'écart
        kept_indices = [minimum_indices[0]]
        for i in range(1, len(minimum_indices)):
            if minimum_indices[i] - kept_indices[-1] >= distance:
                kept_indices.append(minimum_indices[i])
        minimum_indices = np.array(kept_indices)
    
    return minimum_indices

# Apply the low-pass filter to the magnitude of acceleration
filtered_mag = filtfilt(b, a, mag)

# Define thresholds for step detection
lower_threshold = 0.88 * 9.81  # Example: 0.03g in m/s^2
upper_threshold = 1.15* 9.81   # Example: 1.5g in m/s^2
epsilon = 5

peaks_high, _ = find_peaks(filtered_mag, height=(upper_threshold, upper_threshold+epsilon), distance=20)
peaks_low= find_minima(filtered_mag, distance=20, prominence=0.3, lower_threshold=lower_threshold-3, upper_threshold=lower_threshold)

step_count = 1/2*(len(peaks_high) + len(peaks_low))

mean_step_length = 0.8
print(f"Nombre de minimums détectés low: {len(peaks_low)}")
print(f"Nombre de minimums détectés high: {len(peaks_high)}")

print(f"{mean_step_length*step_count} m")



# Plotting the results
plt.figure(figsize=(15,5))
plt.plot(filtered_mag, label='Filtered Magnitude')
plt.plot(peaks_high, filtered_mag[peaks_high], 'rx', label='Detected Steps')
plt.plot(peaks_low, filtered_mag[peaks_low], 'rx', label='Detected Steps')

# Plot threshold lines
plt.axhline(y=lower_threshold, color='g', linestyle='--', label=f'Lower Threshold ({lower_threshold:.2f} m/s²)')
plt.axhline(y=upper_threshold, color='b', linestyle='--', label=f'Upper Threshold ({upper_threshold:.2f} m/s²)')
plt.axhline(y=upper_threshold+epsilon, color='r', linestyle='--', label=f'Upper Threshold + epsilon({upper_threshold+epsilon:.2f} m/s²)')
plt.axhline(y=lower_threshold-3, color='y', linestyle='--', label=f'Lower Threshold - epsilon({lower_threshold-epsilon:.2f} m/s²)')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.show()
