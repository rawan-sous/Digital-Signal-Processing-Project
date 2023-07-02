import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the css.mat file
data_css = sio.loadmat('css.mat')
css_data = data_css['css'].flatten()

# Concatenate five blocks of CSS data
num_blocks = 5
concatenated_css_data = np.tile(css_data, num_blocks)

# Define the impulse response of the echo path
impulse_response = np.array([2, 10, 1])

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

# Estimate the input power
input_power = 10 * np.log10(np.sum(css_data ** 2))

# Estimate the output power
output_power = 10 * np.log10(np.sum(echo_signal ** 2) / len(echo_signal))

# Calculate the echo-return-loss (ERL) in dB
erl = input_power - output_power

print(f"Input Power: {input_power} dB")
print(f"Output Power: {output_power} dB")
print(f"Echo-Return-Loss (ERL): {erl} dB")
