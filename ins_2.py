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
rotation_threshold = 0.8  # rad/s (à ajuster selon vos données)
min_rotation_time = 100   # ms (durée minimale pour considérer une rotation)

# Détecter les périodes de rotation significative
rotation_indices = []
current_rotation = None

for i in range(len(time)):
    if abs(gz_filtered[i]) > rotation_threshold:
        if current_rotation is None:
            # Début d'une nouvelle rotation
            current_rotation = i
    else:
        if current_rotation is not None:
            # Fin d'une rotation
            if (time[i] - time[current_rotation]) >= min_rotation_time:
                # On mémorise l'indice où la rotation se termine
                rotation_indices.append((current_rotation, i))
            current_rotation = None

# Afficher la détection des rotations
plt.figure(figsize=(12, 6))
plt.plot(np.array(time)/1000, gz_filtered, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=rotation_threshold, color='g', linestyle='--', alpha=0.5)
plt.axhline(y=-rotation_threshold, color='r', linestyle='--', alpha=0.5)

# Marquer les périodes de rotation significative
for start_idx, end_idx in rotation_indices:
    plt.axvspan(time[start_idx]/1000, time[end_idx]/1000, color='yellow', alpha=0.3)

plt.title("Détection de rotation avec seuil (±{:.2f} rad/s)".format(rotation_threshold))
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse angulaire (rad/s)")
plt.grid(True)
plt.show()

# Origine du trajet
origin_x, origin_y = 0, 0

# Traitement du parcours par segments entre les rotations
x_positions = [origin_x]
y_positions = [origin_y]
directions = [0]  # En radians (0 = direction Est)

# Vitesse moyenne constante par segment (à ajuster selon votre scénario)
speed = 1.0  # unités/seconde

# Ajouter le premier point de départ
segment_indices = [0]

# Ajouter tous les indices de fin de rotation pour diviser le parcours
for _, end_idx in rotation_indices:
    segment_indices.append(end_idx)

# Ajouter le dernier indice pour le dernier segment
segment_indices.append(len(time)-1)
segment_indices = sorted(set(segment_indices))  # Éliminer les doublons et trier

# Pour chaque segment
for i in range(1, len(segment_indices)):
    start_idx = segment_indices[i-1]
    end_idx = segment_indices[i]
    
    # Calculer la durée du segment en secondes
    duration = (time[end_idx] - time[start_idx]) / 1000.0
    
    # Calculer la rotation pendant ce segment (moyenne des vitesses angulaires)
    mean_rotation_rate = np.mean(gz_filtered[start_idx:end_idx+1])
    total_rotation = mean_rotation_rate * duration
    
    # Mettre à jour la direction après la rotation
    new_direction = directions[-1] + total_rotation
    
    # Calculer le déplacement dans cette direction
    distance = speed * duration
    dx = distance * np.cos(new_direction)
    dy = distance * np.sin(new_direction)
    
    # Ajouter le nouveau point
    x_positions.append(x_positions[-1] + dx)
    y_positions.append(y_positions[-1] + dy)
    directions.append(new_direction)

# Afficher le parcours
plt.figure(figsize=(10, 10))
plt.plot(x_positions, y_positions, 'b-o', linewidth=2)
plt.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Départ')
plt.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, label='Arrivée')

# Ajouter des flèches pour montrer la direction
for i in range(len(x_positions)):
    dx = np.cos(directions[i])
    dy = np.sin(directions[i])
    plt.arrow(x_positions[i], y_positions[i], dx*0.5, dy*0.5, 
              head_width=0.2, head_length=0.3, fc='blue', ec='blue', alpha=0.7)

# Marquer les points de rotation détectés
for i in range(1, len(segment_indices)-1):
    idx = i
    plt.plot(x_positions[idx], y_positions[idx], 'mo', markersize=8)

plt.title("Trajectoire par segments avec rotations détectées")
plt.xlabel("x (unités)")
plt.ylabel("y (unités)")
plt.grid(True)
plt.axis('equal')
plt.legend(['Trajectoire', 'Départ', 'Arrivée'])
plt.show()