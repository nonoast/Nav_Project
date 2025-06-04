import pandas as pd
import matplotlib.pyplot as plt

csv_file = "forward.csv"

df = pd.read_csv(csv_file, skiprows=1)


time = df.iloc[:, 0].values  # temps en secondes
ax = df.iloc[:, 1].values    # accélération en X
ay = df.iloc[:, 2].values    # accélération en Yacceleration

# Initialisation
vx, vy = [0], [0]
x, y = [0], [0]

# Intégration simple (Euler)
for i in range(1, len(time)):
    dt = time[i] - time[i - 1]

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