import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

df = pd.read_csv('sensor_L.csv', skiprows=1, header=None, names=['time', 'x', 'y', 'z'])

df['accel_mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

sampling_rate = 50  # Hz (adjust based on your data)
cutoff_freq = 5     # Hz

b, a = butter_lowpass(cutoff_freq, sampling_rate)
filtered_mag = filtfilt(b, a, df['accel_mag'])

# Assume 'filtered_mag' is your filtered acceleration magnitude array
lower_threshold = 0.03 * 9.81  # Example: 0.03g in m/s^2
upper_threshold = 1.5 * 9.81   # Example: 1.5g in m/s^2

peaks, _ = find_peaks(filtered_mag, height=(lower_threshold, upper_threshold), distance=20)
step_count = len(peaks)

plt.figure(figsize=(15,5))
plt.plot(filtered_mag, label='Filtered Magnitude')
plt.plot(peaks, filtered_mag[peaks], 'rx', label='Detected Steps')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.show()


print(f"Estimated steps: {step_count}")
