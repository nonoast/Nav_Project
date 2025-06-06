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

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

sampling_rate = 100  # Hz (sampling frequency)
cutoff_freq = 10     # Hz (highest frequency of interest)

b, a = butter_lowpass(cutoff_freq, sampling_rate)
filtered_mag = filtfilt(b, a, mag)
print(filtered_mag[:10])  # Print the first 10 values
# Assume 'filtered_mag' is your filtered acceleration magnitude array
lower_threshold = 0.8 * 9.81  # Example: 0.03g in m/s^2
upper_threshold = 1.15* 9.81   # Example: 1.5g in m/s^2
epsilon = 5 # Small value to avoid division by zero
peaks_high, _ = find_peaks(filtered_mag, height=(upper_threshold, upper_threshold+epsilon), distance=20)
peaks_low, _ = find_peaks(filtered_mag, height=(lower_threshold-epsilon, lower_threshold), distance=20)
step_count = len(peaks_high)
mean_step_length = 0.8

print(f"{mean_step_length*step_count} m")

# Plotting the results
plt.figure(figsize=(15,5))
plt.plot(filtered_mag, label='Filtered Magnitude')
plt.plot(peaks_high, filtered_mag[peaks_high], 'rx', label='Detected Steps')
plt.plot(peaks_low, filtered_mag[peaks_low], 'rx', label='Detected Steps')

# Plot threshold lines
plt.axhline(y=lower_threshold, color='g', linestyle='--', label=f'Lower Threshold ({lower_threshold:.2f} m/s²)')
plt.axhline(y=upper_threshold, color='b', linestyle='--', label=f'Upper Threshold ({upper_threshold:.2f} m/s²)')
plt.axhline(y=upper_threshold+epsilon, color='b', linestyle='--', label=f'Upper Threshold + epsilon({upper_threshold+epsilon:.2f} m/s²)')
plt.axhline(y=lower_threshold-epsilon, color='b', linestyle='--', label=f'Upper Threshold - epsilon({lower_threshold-epsilon:.2f} m/s²)')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.show()


print(f"Estimated steps: {step_count}")
