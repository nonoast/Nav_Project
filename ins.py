import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Add NumPy import for mathematical functions
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
# Load data from CSV file
csv_file = "data/trajet_rouge.csv"
df = pd.read_csv(csv_file, sep=';', skiprows=1)

# Extract sensor data
time = df.iloc[:, 0].values  # time in milliseconds
# Accelerometer data (m/s²)
ax = df.iloc[:, 1].values
ay = df.iloc[:, 2].values
az = df.iloc[:, 3].values
# Gyroscope data (rad/s)
gx = df.iloc[:, 4].values
gy = df.iloc[:, 5].values
gz = df.iloc[:, 6].values
# Magnetometer data (μT)
mx = df.iloc[:, 7].values
my = df.iloc[:, 8].values
mz = df.iloc[:, 9].values


# Fonction de filtrage Butterworth passe-bas
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    # Conception du filtre
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Appliquer le filtre
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Calculer la fréquence d'échantillonnage approximative
fs = 1000 / np.mean(np.diff(time))  # Hz
cutoff = 2.0  # Fréquence de coupure en Hz

# Filtrage des données du gyroscope
gz_filtered = butter_lowpass_filter(gz, cutoff, fs)


# Initialisation
vx, vy = [0], [0]
x, y = [0], [0]
theta = [0]  # Initial heading angle (in radians)

# Intégration simple (Euler)
for i in range(1, len(time)):
    dt = (time[i] - time[i - 1]) / 10000.0  # Convert milliseconds to seconds
    
    # Update orientation (integrate gyroscope data)
    # Using gz for heading in 2D plane (assuming device is roughly horizontal)
    theta.append(theta[-1] + gz[i] * dt)
    
    # Calculate accelerations in global frame
    # Rotate accelerometer readings based on current heading
    ax_global = ax[i] * np.cos(theta[-1]) - ay[i] * np.sin(theta[-1])
    ay_global = ax[i] * np.sin(theta[-1]) + ay[i] * np.cos(theta[-1])
    
    # Vitesse (in global frame)
    vx.append(vx[-1] + ax_global * dt)
    vy.append(vy[-1] + ay_global * dt)

    # Position
    x.append(x[-1] + vx[-1] * dt)
    y.append(y[-1] + vy[-1] * dt)


# Affichage de la trajectoire
plt.figure(figsize=(6, 6))
plt.plot(x, y, marker='o')
plt.title("Trajectoire estimée avec orientation (INS)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.show()

# Affichage de l'orientation (optionnel)
plt.figure(figsize=(10, 4))
plt.plot(np.array(time)/1000, np.degrees(theta))  # Convert to seconds and degrees
plt.title("Évolution de l'orientation")
plt.xlabel("Temps (s)")
plt.ylabel("Angle (degrés)")
plt.grid(True)
plt.show()