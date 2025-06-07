import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import math

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
gz = df.iloc[:, 6].values  # Rotation autour de l'axe Z (positif = sens trigonométrique)

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

# Filtrage des données
gz_filtered = butter_lowpass_filter(gz, cutoff, fs)
ax_filtered = butter_lowpass_filter(ax, cutoff, fs)
ay_filtered = butter_lowpass_filter(ay, cutoff, fs)

# Définir un seuil pour détecter les rotations significatives
rotation_threshold = 1  # rad/s (à ajuster selon vos données)
min_rotation_time = 100  # ms (durée minimale pour considérer une rotation)

# Détecter les périodes de rotation significative
rotation_periods = []
current_rotation = None
rotation_type = None

for i in range(len(time)):
    if abs(gz_filtered[i]) > rotation_threshold:
        if current_rotation is None:
            # Début d'une nouvelle rotation
            current_rotation = time[i]
            rotation_type = 'trigonométrique' if gz_filtered[i] > 0 else 'horaire'
    else:
        if current_rotation is not None:
            # Fin d'une rotation
            if (time[i] - current_rotation) >= min_rotation_time:
                rotation_periods.append((current_rotation, time[i], rotation_type))
            current_rotation = None
            rotation_type = None

# Afficher la convention de signe pour la rotation avec les périodes détectées
plt.figure(figsize=(12, 6))
plt.plot(np.array(time)/10000, gz_filtered, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=rotation_threshold, color='g', linestyle='--', alpha=0.5)
plt.axhline(y=-rotation_threshold, color='r', linestyle='--', alpha=0.5)
plt.fill_between(np.array(time)/10000, 0, gz_filtered, where=(gz_filtered > 0), color='green', alpha=0.3, label='Sens trigonométrique')
plt.fill_between(np.array(time)/10000, 0, gz_filtered, where=(gz_filtered < 0), color='red', alpha=0.3, label='Sens horaire')

# Marquer les périodes de rotation significative
for start, end, rot_type in rotation_periods:
    color = 'green' if rot_type == 'trigonométrique' else 'red'
    plt.axvspan(start/10000, end/10000, color=color, alpha=0.2)

plt.title("Détection de rotation avec seuil (±{:.2f} rad/s)".format(rotation_threshold))
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse angulaire (rad/s)")
plt.legend(['Vitesse angulaire', 'Zéro', 'Seuil positif', 'Seuil négatif'])
plt.grid(True)
plt.show()

# Origine du trajet (à personnaliser selon vos besoins)
origin_x, origin_y = 0, 0  # Coordonnées de départ

# Initialisation
vx, vy = [0], [0]
x, y = [origin_x], [origin_y]
theta = [0]  # Angle initial (en radians)
significant_rotations = [False]  # Indicateur de rotation significative

# Intégration avec gyroscope uniquement et détection des rotations significatives
for i in range(1, len(time)):
    dt = (time[i] - time[i - 1]) / 1000.0  # Convert milliseconds to seconds
    
    # Appliquer le seuil pour la détection de rotation significative
    if abs(gz_filtered[i]) > rotation_threshold:
        # Rotation significative détectée
        rotation_increment = gz_filtered[i] * dt
        significant_rotations.append(True)
    else:
        # Petite rotation ignorée (considérée comme du bruit)
        rotation_increment = 0
        significant_rotations.append(False)
    
    # Mise à jour de l'angle en tenant compte du seuil
    theta.append(theta[-1] + rotation_increment)
    
    # Calcul des accélérations dans le repère global
    ax_global = ax_filtered[i] * np.cos(theta[-1]) - ay_filtered[i] * np.sin(theta[-1])
    ay_global = ax_filtered[i] * np.sin(theta[-1]) + ay_filtered[i] * np.cos(theta[-1])
    
    # Intégration des vitesses
    vx.append(vx[-1] + ax_global * dt)
    vy.append(vy[-1] + ay_global * dt)
    
    # Intégration des positions
    x.append(x[-1] + vx[-1] * dt)
    y.append(y[-1] + vy[-1] * dt)

# Affichage de la trajectoire avec indication de l'orientation
plt.figure(figsize=(10, 10))
plt.plot(x, y, '-', linewidth=2, alpha=0.7)

# Marquer le départ et l'arrivée
plt.plot(x[0], y[0], 'go', markersize=10, label='Départ')
plt.plot(x[-1], y[-1], 'ro', markersize=10, label='Arrivée')

# Ajouter des flèches pour montrer l'orientation à intervalles réguliers
arrow_indices = np.linspace(0, len(x)-1, 20).astype(int)
for idx in arrow_indices:
    # Calculer la direction basée sur l'angle
    dx = np.cos(theta[idx])
    dy = np.sin(theta[idx])
    plt.arrow(x[idx], y[idx], dx*0.5, dy*0.5, head_width=0.2, head_length=0.3, 
              fc='blue', ec='blue', alpha=0.5)
    
    # Marquer les points où une rotation significative a été détectée
    if idx > 0 and significant_rotations[idx]:
        plt.plot(x[idx], y[idx], 'o', color='magenta', markersize=5)

plt.title("Trajectoire avec rotation significative détectée (seuil = {:.2f} rad/s)".format(rotation_threshold))
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.legend(['Trajectoire', 'Départ', 'Arrivée'])
plt.show()

# Affichage de l'évolution de l'orientation
plt.figure(figsize=(12, 6))
plt.plot(np.array(time)/10000, np.degrees(theta), 'b-', linewidth=2)

# Marquer les périodes de rotation
rotation_times = np.array([time[i]/10000 for i in range(len(time)) if significant_rotations[i]])
rotation_angles = np.array([np.degrees(theta[i]) for i in range(len(theta)) if i < len(significant_rotations) and significant_rotations[i]])

plt.plot(rotation_times, rotation_angles, 'ro', markersize=2, alpha=0.5)

plt.title("Évolution de l'orientation avec rotations significatives")
plt.xlabel("Temps (s)")
plt.ylabel("Angle (degrés)")
plt.grid(True)
plt.legend(['Angle', 'Rotation détectée'])
plt.show()