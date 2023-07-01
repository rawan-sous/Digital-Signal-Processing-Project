import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Task A: Load the file path.mat and plot the impulse and frequency responses of the echo path.

# Load the path.mat file
data_path = sio.loadmat('path.mat')
impulse_response = data_path['path'].flatten()

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

# Task C: Concatenate five blocks of CSS data, feed them into the echo path, and plot the resulting echo signal.

# Load the css.mat file
data_css = sio.loadmat('css.mat')
css_data = data_css['css'].flatten()

# Concatenate five blocks of CSS data
num_blocks = 5
concatenated_css_data = np.tile(css_data, num_blocks)

# Feed the concatenated CSS data into the echo path
echo_signal = np.convolve(concatenated_css_data, impulse_response)[:len(concatenated_css_data)]

# Plot the resulting echo signal
plt.figure()
plt.plot(echo_signal)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Resulting Echo Signal')
plt.grid(True)
plt.show()

# Estimate the input and output powers
input_power = np.sum(concatenated_css_data ** 2)
output_power = np.sum(echo_signal ** 2)

# Calculate the echo-return-loss (ERL) in dB
erl = 10 * np.log10(output_power / input_power)

print(f"Input Power: {input_power} (dB)")
print(f"Output Power: {output_power} (dB)")
print(f"Echo-Return-Loss (ERL): {erl} dB")
