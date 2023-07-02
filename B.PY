import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the css.mat file
data_css = sio.loadmat('css.mat')
css_data = data_css['css'].flatten()

# Calculate the length of the signal
n_samples = len(css_data)

# Generate the time axis for plotting
time = np.arange(n_samples)
# Calculate the power spectrum using FFT
fft_data = np.fft.fft(css_data)
psd = np.abs(fft_data) ** 2

# Generate the frequency axis for plotting
freq = np.fft.fftfreq(n_samples)

# Plot the power spectrum density (PSD)
plt.subplot(3, 1, 2)
plt.plot(freq, psd)
plt.title('Power Spectrum Density (PSD)')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.xlim(0, 0.5)  # Plot up to the Nyquist frequency
plt.ylim(0, np.max(psd))
plt.tight_layout()

# Plot the CSS signal
plt.figure(figsize=(12, 4))
plt.plot(time, css_data)
plt.title('far-end-signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Define the impulse response of the echo path
data_path = sio.loadmat('path.mat')
impulse_response = data_path['path'].flatten()

# Convolve the CSS data with the impulse response to obtain the echo signal
echo_signal = np.convolve(css_data, impulse_response, mode='same')

# Calculate the error signal
error_signal = css_data - echo_signal

# Plot the echo signal
plt.figure(figsize=(12, 4))
plt.plot(time, echo_signal)
plt.title('Echo Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot the error signal
plt.figure(figsize=(12, 4))
plt.plot(time, error_signal)
plt.title('Error Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
