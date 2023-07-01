import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the path.mat file
data = sio.loadmat('path.mat')
impulse_response = data['path'].flatten()

# Plot the impulse response
plt.figure()
plt.stem(impulse_response)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Impulse Response of the Echo Path')
plt.grid(True)
plt.show()

# Calculate and plot the frequency response
frequency_response = np.fft.fft(impulse_response)
frequency = np.fft.fftfreq(len(impulse_response))
plt.figure()
plt.plot(frequency, np.abs(frequency_response))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Response of the Echo Path')
plt.grid(True)
plt.show()
