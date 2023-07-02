import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Task A: Load the file path.mat and plot the impulse and frequency responses of the echo path.

# Load the path.mat file
data_path = sio.loadmat('path.mat')
impulse_response = data_path['path'].flatten()

# Plot the impulse response
plt.figure()
plt.plot(impulse_response)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Impulse Response of the Echo Path')
plt.grid(True)
plt.show()

# Calculate and plot the frequency response using freqz
w, h = freqz(impulse_response, worN=len(impulse_response))
frequency = w / (2 * np.pi)  # Convert angular frequency to Hz

plt.figure()
plt.plot(frequency, np.abs(h))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Response of the Echo Path')
plt.grid(True)
plt.show()

