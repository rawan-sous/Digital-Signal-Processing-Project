import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
mat_data = sio.loadmat('css.mat')

# Extract the data from the 'css' variable
css_data = mat_data['css']

# Reshape the data to a 1D array
css_data = css_data.reshape(-1)

# Calculate the length of the signal
n_samples = len(css_data)

# Generate the time axis for plotting
time = np.arange(n_samples)

# Plot the samples
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, css_data)
plt.title('CSS Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Calculate the power spectrum using FFT
fft_data = np.fft.fft(css_data)
psd = np.abs(fft_data) ** 2

# Generate the frequency axis for plotting
freq = np.fft.fftfreq(n_samples)

# Plot the power spectrum density (PSD)
plt.subplot(2, 1, 2)
plt.plot(freq, psd)
plt.title('Power Spectrum Density (PSD)')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.xlim(0, 0.5)  # Plot up to the Nyquist frequency
plt.ylim(0, np.max(psd))
plt.tight_layout()

# Show the plots
plt.show()