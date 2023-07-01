import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Parameters
n_blocks = 10  # Number of blocks
n_taps = 128  # Number of filter taps
alpha = 6e-10  # Step size parameter
mu = 0.25  # Adaptation factor

# Load the CSS data
mat_data = sio.loadmat('css.mat')
css_data = mat_data['css'].reshape(-1)

# Create far-end and echo signals
far_end = css_data[:n_blocks * n_taps]
echo = np.convolve(far_end, css_data)[:n_blocks * n_taps]

# Initialize adaptive filter weights
weights = np.zeros(n_taps)

# Initialize error signal and estimated echo
error_signal = np.zeros(n_blocks * n_taps)
estimated_echo = np.zeros(n_blocks * n_taps)

# Run the adaptive filter
for block in range(n_blocks):
    # Extract current block of far-end signal
    block_start = block * n_taps
    block_end = block_start + n_taps
    x = far_end[block_start:block_end]
    
    # Filter input signal
    y = np.dot(weights, x)
    
    # Calculate error signal
    d = echo[block_start:block_end]
    e = d - y
    
    # Update filter weights using NLMS algorithm
    norm_x = np.linalg.norm(x)
    weights += (mu / (alpha + norm_x**2)) * x * np.conj(e)
    
    # Store error signal and estimated echo
    error_signal[block_start:block_end] = e
    estimated_echo[block_start:block_end] = y

# Plot the signals
sample_index = np.arange(n_blocks * n_taps)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sample_index, far_end)
plt.title('Far-End Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(sample_index, echo)
plt.title('Echo Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(sample_index, error_signal)
plt.title('Error Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Plot the echo path and its estimate
plt.figure(figsize=(12, 4))
plt.plot(range(n_taps), css_data[:n_taps], label='Actual Echo Path')
plt.plot(range(n_taps), weights, label='Estimated Echo Path')
plt.title('Echo Path and Its Estimate')
plt.xlabel('Tap Index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()