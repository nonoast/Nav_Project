import pandas as pd
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
# Initialisation
vx, vy = [0], [0]
x, y = [0], [0]

# Intégration simple (Euler)
for i in range(1, len(time)):
    dt = (time[i] - time[i - 1]) / 1000.0  # Convert milliseconds to seconds

    # Vitesse
    vx.append(vx[-1] + ax[i] * dt)
    vy.append(vy[-1] + ay[i] * dt)

    # Position
    x.append(x[-1] + vx[-1] * dt)
    y.append(y[-1] + vy[-1] * dt)

# Affichage de la trajectoire
plt.figure(figsize=(6, 6))
plt.plot(x, y, marker='o')
plt.title("Trajectoire estimée (INS)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis('equal')
plt.show()