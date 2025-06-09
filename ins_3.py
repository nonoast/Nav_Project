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

# --------------------------------------------------------------
# NOUVELLE APPROCHE: Construction de la trajectoire par segments
# --------------------------------------------------------------

# Paramètres
origin_x, origin_y = 0, 0  # Coordonnées de départ
step_distance = 0.5  # Distance parcourue entre chaque échantillon (mètres)
current_heading = 0  # Direction initiale (en radians)

# Trier les rotations par indice de début pour faciliter le traitement
rotations_sorted = sorted(rotation_periods, key=lambda x: x['start_idx'])

# Construire la liste des segments
segments = []
last_index = 0

for rot in rotations_sorted:
    # Ajouter un segment avant la rotation (déplacement rectiligne)
    if rot['start_idx'] > last_index:
        segments.append({
            'type': 'deplacement',
            'start_idx': last_index,
            'end_idx': rot['start_idx'],
            'heading': current_heading
        })
    
    # Ajouter le segment de rotation
    segments.append({
        'type': 'rotation',
        'start_idx': rot['start_idx'],
        'end_idx': rot['end_idx'],
        'total_angle': rot['total_angle'],
        'rot_type': rot['type']
    })
    
    # Mettre à jour la direction courante après la rotation
    current_heading += rot['total_angle']
    # Normaliser l'angle entre -π et π
    current_heading = ((current_heading + np.pi) % (2 * np.pi)) - np.pi
    
    last_index = rot['end_idx']

# Ajouter le dernier segment si nécessaire
if last_index < len(time) - 1:
    segments.append({
        'type': 'deplacement',
        'start_idx': last_index,
        'end_idx': len(time) - 1,
        'heading': current_heading
    })

# Construire la trajectoire à partir des segments
trajectory_x = [origin_x]
trajectory_y = [origin_y]
segment_colors = ['blue']  # Couleur pour chaque segment
headings = [0]  # Liste pour stocker les orientations

current_x, current_y = origin_x, origin_y
current_heading = 0

for i, segment in enumerate(segments):
    if segment['type'] == 'deplacement':
        # Calculer la durée du segment en secondes
        segment_duration = (time[segment['end_idx']] - time[segment['start_idx']]) / 1000.0
        # Distance parcourue pendant ce segment (proportionnelle à la durée)
        distance = step_distance * segment_duration
        # Calculer le déplacement selon la direction actuelle
        dx = distance * np.cos(segment['heading'])
        dy = distance * np.sin(segment['heading'])
        # Nouveau point après déplacement
        current_x += dx
        current_y += dy
        # Ajouter le point à la trajectoire
        trajectory_x.append(current_x)
        trajectory_y.append(current_y)
        segment_colors.append('blue')  # Segment de déplacement en bleu
        headings.append(segment['heading'])
        
    elif segment['type'] == 'rotation':
        # Pour une rotation, on reste au même endroit mais on change de direction
        # On ajoute quand même un point pour marquer la rotation
        trajectory_x.append(current_x)
        trajectory_y.append(current_y)
        # La couleur dépend du type de rotation
        color = 'green' if segment['rot_type'] == 'trigonométrique' else 'red'
        segment_colors.append(color)
        # Mettre à jour l'orientation
        current_heading += segment['total_angle']
        # Normaliser l'angle entre -π et π
        current_heading = ((current_heading + np.pi) % (2 * np.pi)) - np.pi
        headings.append(current_heading)

# Afficher la trajectoire par segments
plt.figure(figsize=(10, 10))

# Tracer les segments avec des couleurs différentes
for i in range(len(trajectory_x)-1):
    plt.plot([trajectory_x[i], trajectory_x[i+1]], [trajectory_y[i], trajectory_y[i+1]], 
             color=segment_colors[i+1], linewidth=2, alpha=0.8)
    # Ajouter une flèche de direction
    if i % 2 == 0:  # Une flèche tous les 2 segments pour éviter l'encombrement
        plt.arrow(trajectory_x[i], trajectory_y[i], 
                  0.2 * np.cos(headings[i]), 0.2 * np.sin(headings[i]), 
                  head_width=0.1, head_length=0.15, fc='blue', ec='blue', alpha=0.7)

# Marquer les points de rotation
rotation_points_x = []
rotation_points_y = []
rotation_labels = []

for i, segment in enumerate(segments):
    if segment['type'] == 'rotation':
        idx = i + 1  # +1 car les indices de trajectory_x sont décalés (on a ajouté l'origine)
        if idx < len(trajectory_x):
            rotation_points_x.append(trajectory_x[idx])
            rotation_points_y.append(trajectory_y[idx])
            angle_degrees = np.degrees(segments[i]['total_angle'])
            rotation_labels.append(f"{angle_degrees:.1f}°")

plt.scatter(rotation_points_x, rotation_points_y, color='magenta', s=100, zorder=5)
for i, txt in enumerate(rotation_labels):
    plt.annotate(txt, (rotation_points_x[i], rotation_points_y[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Marquer le départ et l'arrivée
plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Départ')
plt.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=10, label='Arrivée')

plt.title("Trajectoire reconstruite par segments avec rotations détectées")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# Animation du parcours
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(min(trajectory_x)-1, max(trajectory_x)+1)
ax.set_ylim(min(trajectory_y)-1, max(trajectory_y)+1)
ax.grid(True)
ax.set_title("Animation du parcours")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# Tracer la trajectoire complète en arrière-plan (en gris clair)
ax.plot(trajectory_x, trajectory_y, 'lightgray', linewidth=1, alpha=0.5)

# Ligne qui s'allongera progressivement
line, = ax.plot([], [], 'b-', linewidth=2)
# Point qui se déplacera
point, = ax.plot([], [], 'ro', markersize=8)

# Marqueur de départ et d'arrivée
ax.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Départ')
ax.plot(trajectory_x[-1], trajectory_y[-1], 'mo', markersize=10, label='Arrivée')
ax.legend()

# Liste pour stocker les objets flèche afin de pouvoir les supprimer
arrows = []

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(frame):
    # Protection contre les index hors limite
    if frame >= len(trajectory_x):
        return line, point
        
    # Mettre à jour la ligne (trajectoire jusqu'au point actuel)
    line.set_data(trajectory_x[:frame+1], trajectory_y[:frame+1])
    
    # Mettre à jour le point (position actuelle)
    point.set_data([trajectory_x[frame]], [trajectory_y[frame]])  # Envelopper dans une liste
    
    # Dessiner une nouvelle flèche pour l'orientation actuelle
    if frame < len(headings):
        # Supprimer toutes les flèches précédentes
        for arrow in arrows:
            if arrow in ax.get_children():  # Vérifier si la flèche existe encore
                arrow.remove()
        arrows.clear()
        
        # Créer une nouvelle flèche
        arr = ax.arrow(trajectory_x[frame], trajectory_y[frame],
                      0.2 * np.cos(headings[frame]), 0.2 * np.sin(headings[frame]),
                      head_width=0.1, head_length=0.2, fc='green', ec='green')
        arrows.append(arr)
    
    # Afficher les informations sur le segment actuel
    # Trouver dans quel segment nous sommes
    segment_info = ""
    for i, segment in enumerate(segments):
        if segment['start_idx'] <= frame <= segment['end_idx']:
            if segment['type'] == 'deplacement':
                segment_info = f"Déplacement (heading: {np.degrees(segment['heading']):.1f}°)"
            else:
                segment_info = f"Rotation {segment['rot_type']} ({np.degrees(segment['total_angle']):.1f}°)"
            break
    
    # Mettre à jour le titre avec les informations sur le segment
    ax.set_title(f"Animation du parcours - Frame {frame}/{len(trajectory_x)-1}\n{segment_info}")
    
    return line, point

# Créer l'animation avec une vitesse adaptée à la longueur de la trajectoire
# Utiliser un intervalle plus long si la trajectoire est longue
interval = max(50, min(200, 20000 // len(trajectory_x)))  # Entre 50 et 200 ms

ani = FuncAnimation(fig, update, frames=len(trajectory_x),
                   init_func=init, blit=False, interval=interval, 
                   repeat=True, repeat_delay=1000)  # Répéter avec délai

plt.show()