# Fourier Series and Signal Reconstruction

This project demonstrates the use of Fourier Series and Fourier Transform to reconstruct periodic signals based on their harmonic components. The signals are generated and reconstructed for different harmonic counts (1, 5, 10, and 20), and the results are compared to the original signal.

## Features

- **Signal Generation**: Generates periodic signals (`x1(t)` and `x2(t)`) with a defined period `T=4` seconds.
- **Fourier Series Reconstruction**: Reconstructs the signals using Fourier Series with varying harmonic counts (1, 5, 10, and 20).
- **Fourier Transform**: Performs a Fourier Transform to analyze the frequency components of the signals.
- **Signal Comparison**: Compares the original and reconstructed signals, highlighting the effect of increasing harmonics on the accuracy of the reconstruction.
- **Signal Visualization**: Visualizes both the original and reconstructed signals using plots for better understanding.

## File Structure

- `fourier_series.py`: Main Python script for signal generation, Fourier series reconstruction, and visualization.
- `signal_plot.png`: Output image file that contains the plots comparing the original and reconstructed signals.
- `README.md`: This file containing project documentation.

## Functions

### 1. **`generate_x1`**
   - **Description**: Generates the first periodic signal (`x1(t)`) with a slope of 1 in the specified period.
   - **Output**: Returns the values of the signal `x1(t)` for plotting.

### 2. **`generate_x2`**
   - **Description**: Generates the second periodic signal (`x2(t)`) with constant value in the specified period.
   - **Output**: Returns the values of the signal `x2(t)` for plotting.

### 3. **`fourier_series_reconstruction`**
   - **Description**: Reconstructs a signal using the Fourier series for a specified number of harmonics.
   - **Input**: `x(t)` (Signal data), `harmonics_count` (Number of harmonics to include).
   - **Output**: Returns the reconstructed signal after Fourier series approximation.

### 4. **`plot_signal`**
   - **Description**: Plots the original and reconstructed signals for comparison.
   - **Input**: `t` (Time array), `original_signal` (Original signal data), `reconstructed_signal` (Reconstructed signal data).
   - **Output**: Displays a plot comparing the original and reconstructed signals.

### 5. **`compute_fourier_transform`**
   - **Description**: Computes the Fourier Transform of a signal.
   - **Input**: `signal` (Signal data).
   - **Output**: Returns the frequency-domain representation of the signal.

## Requirements

- **Python 3.x**
- **NumPy**
- **Matplotlib**
