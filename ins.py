import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import math

# Load data from CSV file
csv_file = "data/trajet_vert.csv"
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
                # On garde aussi l'indice de début et de fin, ainsi que la valeur moyenne de rotation
                start_idx = np.where(time == current_rotation)[0][0]
                end_idx = i
                # Calculer la valeur moyenne de la rotation pendant cet intervalle
                mean_rotation = np.mean(gz_filtered[start_idx:end_idx+1])
                # Calculer l'angle total de rotation
                rotation_duration = (time[end_idx] - time[start_idx]) / 1000.0  # en secondes
                total_angle = mean_rotation * rotation_duration  # en radians
                
                rotation_periods.append({
                    'start_time': current_rotation,
                    'end_time': time[i],
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'type': rotation_type,
                    'mean_value': mean_rotation,
                    'total_angle': total_angle,
                    'duration': rotation_duration
                })
            current_rotation = None
            rotation_type = None

# Afficher la convention de signe pour la rotation avec les périodes détectées
plt.figure(figsize=(12, 6))
plt.plot(np.array(time)/1000, gz_filtered, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=rotation_threshold, color='g', linestyle='--', alpha=0.5)
plt.axhline(y=-rotation_threshold, color='r', linestyle='--', alpha=0.5)
plt.fill_between(np.array(time)/1000, 0, gz_filtered, where=(gz_filtered > 0), color='green', alpha=0.3, label='Sens trigonométrique')
plt.fill_between(np.array(time)/1000, 0, gz_filtered, where=(gz_filtered < 0), color='red', alpha=0.3, label='Sens horaire')

# Marquer les périodes de rotation significative
for rot in rotation_periods:
    color = 'green' if rot['type'] == 'trigonométrique' else 'red'
    plt.axvspan(rot['start_time']/1000, rot['end_time']/1000, color=color, alpha=0.2)
    
    # Ajouter une annotation pour l'angle total de rotation
    mid_time = (rot['start_time'] + rot['end_time']) / 2000.0  # Moyenne des temps en secondes
    plt.text(mid_time, gz_filtered[rot['start_idx']] * 1.1, 
             f"{np.degrees(rot['total_angle']):.1f}°", 
             horizontalalignment='center', fontsize=8)

plt.title("Détection de rotation avec seuil (±{:.2f} rad/s)".format(rotation_threshold))
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse angulaire (rad/s)")
plt.legend(['Vitesse angulaire', 'Zéro', 'Seuil positif', 'Seuil négatif'])
plt.grid(True)
plt.show()

print("Rotations significatives détectées:")
for i, rot in enumerate(rotation_periods):
    print(f"Rotation {i+1}:")
    print(f"  Type: {rot['type']}")
    print(f"  Début: {rot['start_time']/1000:.2f} s (indice {rot['start_idx']})")
    print(f"  Fin: {rot['end_time']/1000:.2f} s (indice {rot['end_idx']})")
    print(f"  Durée: {rot['duration']:.2f} s")
    print(f"  Vitesse angulaire moyenne: {rot['mean_value']:.2f} rad/s")
    print(f"  Angle total: {np.degrees(rot['total_angle']):.2f}°")
    print()

# Origine du trajet (à personnaliser selon vos besoins)
origin_x, origin_y = 0, 0  # Coordonnées de départ

# Initialisation
vx, vy = [0], [0]
x, y = [origin_x], [origin_y]
theta = [0]  # Angle initial (en radians)
significant_rotations = [False]  # Indicateur de rotation significative
rotation_info = []  # Pour stocker les informations sur chaque rotation significative

# Intégration avec gyroscope uniquement et détection des rotations significatives
current_rotation_start = None
current_rotation_values = []

for i in range(1, len(time)):
    dt = (time[i] - time[i - 1]) / 1000.0  # Convert milliseconds to seconds
    
    # Appliquer le seuil pour la détection de rotation significative
    if abs(gz_filtered[i]) > rotation_threshold:
        # Rotation significative détectée
        rotation_increment = gz_filtered[i] * dt
        significant_rotations.append(True)
        
        # Si c'est le début d'une nouvelle rotation
        if current_rotation_start is None:
            current_rotation_start = i
            current_rotation_values = [gz_filtered[i]]
        else:
            # Continuer à accumuler les valeurs de rotation
            current_rotation_values.append(gz_filtered[i])
            
    else:
        # Petite rotation ignorée (considérée comme du bruit)
        rotation_increment = 0
        significant_rotations.append(False)
        
        # Si une rotation vient de se terminer
        if current_rotation_start is not None:
            # Calculer le temps de rotation
            rotation_time = (time[i] - time[current_rotation_start]) / 1000.0  # en secondes
            
            # Vérifier si la rotation est suffisamment longue
            if rotation_time >= min_rotation_time / 1000.0:
                # Calculer la valeur moyenne et l'angle total
                mean_rotation = np.mean(current_rotation_values)
                total_angle = mean_rotation * rotation_time
                
                rotation_info.append({
                    'start_idx': current_rotation_start,
                    'end_idx': i-1,
                    'start_time': time[current_rotation_start],
                    'end_time': time[i-1],
                    'mean_value': mean_rotation,
                    'total_angle': total_angle,
                    'direction': 'trigonométrique' if mean_rotation > 0 else 'horaire'
                })
            
            # Réinitialiser pour la prochaine rotation
            current_rotation_start = None
            current_rotation_values = []
    
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

# Affichage des informations sur les rotations détectées pendant l'intégration
print("\nRotations détectées pendant l'intégration de la trajectoire:")
for i, rot in enumerate(rotation_info):
    print(f"Rotation {i+1}:")
    print(f"  Direction: {rot['direction']}")
    print(f"  Début: {rot['start_time']/1000:.2f} s (indice {rot['start_idx']})")
    print(f"  Fin: {rot['end_time']/1000:.2f} s (indice {rot['end_idx']})")
    print(f"  Durée: {(rot['end_time'] - rot['start_time'])/1000:.2f} s")
    print(f"  Vitesse angulaire moyenne: {rot['mean_value']:.2f} rad/s")
    print(f"  Angle total: {np.degrees(rot['total_angle']):.2f}°")
    print()

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

# Marquer les points de rotation significative avec leur angle
for rot in rotation_info:
    midpoint_idx = (rot['start_idx'] + rot['end_idx']) // 2
    if midpoint_idx < len(x):  # Vérifier que l'indice est valide
        plt.plot(x[midpoint_idx], y[midpoint_idx], 'o', color='magenta', markersize=8)
        plt.annotate(f"{np.degrees(rot['total_angle']):.1f}°", 
                    (x[midpoint_idx], y[midpoint_idx]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)

plt.title("Trajectoire avec rotations significatives détectées")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.legend(['Trajectoire', 'Départ', 'Arrivée'])
plt.show()

# Affichage de l'évolution de l'orientation
plt.figure(figsize=(12, 6))
plt.plot(np.array(time)/1000, np.degrees(theta), 'b-', linewidth=2)

# Marquer les rotations significatives
for rot in rotation_info:
    start_time = time[rot['start_idx']]/1000
    end_time = time[rot['end_idx']]/1000
    start_angle = np.degrees(theta[rot['start_idx']])
    end_angle = np.degrees(theta[rot['end_idx']])
    
    # Surligner la période de rotation
    color = 'green' if rot['direction'] == 'trigonométrique' else 'red'
    plt.axvspan(start_time, end_time, color=color, alpha=0.2)
    
    # Annoter l'angle de rotation
    plt.annotate(f"{np.degrees(rot['total_angle']):.1f}°", 
                 xy=((start_time + end_time)/2, (start_angle + end_angle)/2),
                 xytext=(0, 15),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

plt.title("Évolution de l'orientation avec rotations significatives")
plt.xlabel("Temps (s)")
plt.ylabel("Angle (degrés)")
plt.grid(True)
plt.show()

# Exporter les informations de rotation au format CSV si nécessaire
rotation_df = pd.DataFrame(rotation_info)
print(rotation_df)
# rotation_df.to_csv('rotations_detectees.csv', index=False)