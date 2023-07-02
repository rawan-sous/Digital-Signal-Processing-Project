import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Task E: Plot the amplitude and phase response for the estimated FIR channel at the end of the iterations. Compare it with the given FIR system (Path).

# Load the path.mat file
data_path = sio.loadmat('path.mat')
impulse_response = data_path['path'].flatten()

# Define NLMS algorithm parameters
filter_length = 128
step_size = 1e-6
mu = 0.25

# Generate far-end signal and echo signal
far_end_signal = np.random.randn(len(impulse_response) * 10)  # Random signal for demonstration
echo_signal = np.convolve(far_end_signal, impulse_response)[:len(far_end_signal)]

# Initialize FIR filter coefficients
estimated_channel = np.zeros(filter_length)

# Apply NLMS algorithm for adaptive filtering
for i in range(filter_length, len(echo_signal)):
    # Extract input and desired signals for current iteration
    x = far_end_signal[i-filter_length:i]
    d = echo_signal[i]
    
    # Calculate error signal
    y = np.dot(estimated_channel, x)
    e = d - y
    
    # Update filter coefficients using NLMS algorithm
    norm_x = np.linalg.norm(x)
    estimated_channel += 2 * mu / (norm_x ** 2 + np.finfo(float).eps) * e * x

# Calculate frequency response of the estimated FIR channel
frequency_response_estimated = np.fft.fft(estimated_channel)
frequency = np.fft.fftfreq(filter_length)

# Calculate frequency response of the original FIR system (Path)
frequency_response_original = np.fft.fft(impulse_response)

# Plot the amplitude response
plt.figure()
plt.plot(frequency, np.abs(frequency_response_estimated), label='Estimated Channel')
plt.plot(frequency, np.abs(frequency_response_original), label='Original Path')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Response Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Plot the phase response
plt.figure()
plt.plot(frequency, np.angle(frequency_response_estimated), label='Estimated Channel')
plt.plot(frequency, np.angle(frequency_response_original), label='Original Path')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase')
plt.title('Phase Response Comparison')
plt.legend()
plt.grid(True)
plt.show()

