import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat  # Replace this with a .sif reader if available

# Placeholder for SIFreader function
def SIFreader(filename):
    # This function should read `.sif` files. Replace this with the actual implementation.
    return {
        "imageData": np.random.random((2000, 300)),  # Example simulated data
        "axisWavelength": np.linspace(400, 700, 2000),  # Simulated wavelength axis
        "stackCycleTime": 0.1,
        "kineticLength": 300
    }

# Placeholder for FindPeaks function
def FindPeaks(mode, threshold, min_wave, max_wave, wavelength, spectra):
    # Replace this with the actual implementation.
    # Returns mwave, maxint, err, width
    return (np.array([[500]]), np.array([[1.0]]), 0.01, 0.1, None)

# Preallocation
class Data:
    def __init__(self):
        self.rawdata = None
        self.name = None
        self.wave = None
        self.t = None
        self.numb = None
        self.timeaxis = None
        self.data = None
        self.background = None
        self.intmax = [None, None]
        self.wavemax = [None, None]
        self.fit = None

class Spectra:
    def __init__(self):
        self.spectra = None
        self.mwave = None
        self.maxint = None
        self.err = None
        self.width = None

# Get list of files
files = [f for f in os.listdir('.') if f.endswith('.sif')]

# Initialize data
data = [Data() for _ in range(len(files))]
spectra = [Spectra() for _ in range(len(files))]

# Process each file
for i, file in enumerate(files):
    raw = SIFreader(file)
    
    data[i].rawdata = raw["imageData"]
    data[i].name = file
    data[i].wave = raw["axisWavelength"]
    data[i].t = raw["stackCycleTime"]
    data[i].numb = raw["kineticLength"]
    data[i].timeaxis = np.arange(1, raw["kineticLength"] + 1) * raw["stackCycleTime"]
    
    # Remove background
    background_mean = np.mean(data[i].rawdata[1000:1500, :], axis=0)
    data[i].data = data[i].rawdata - abs(background_mean)
    data[i].background = abs(np.mean(data[i].data[2:100, 2]))
    spectra[i].spectra = data[i].data

# Find maxima in 2 spectra
for i in range(len(files)):
    for j in range(2):
        column_index = 100  # Example column index
        data[i].intmax[j] = np.max(data[i].data[:, column_index])
        index = np.where(data[i].data[:, column_index] == data[i].intmax[j])[0]
        if len(index) > 0:
            data[i].wavemax[j] = data[i].wave[index[0]]

# Check if maxima are the same
fit = []
min_vals = []
wavemid = []
wavelengths = []

for i in range(len(files)):
    data[i].fit = round(data[i].wavemax[0]) == round(data[i].wavemax[1])
    fit.append(data[i].fit)
    min_vals.append(5 * abs(data[i].background))
    wavemid.append(data[i].wavemax[0])
    wavelengths.append(data[i].wave)

# Perform peak fitting
for i in range(len(files)):
    if data[i].fit:
        print(f"Processing file {i + 1}: {files[i]}")
        mwave, maxint, err, width, _ = FindPeaks(
            2,
            min_vals[i],
            wavemid[i] - 0.6,
            wavemid[i] + 0.6,
            wavelengths[i],
            spectra[i].spectra
        )
        spectra[i].mwave = mwave
        spectra[i].maxint = maxint
        spectra[i].err = err
        spectra[i].width = width

# Visualization
for i in range(len(files)):
    plt.figure(i)
    try:
        # Plot 1: Peak Wavelength
        plt.subplot(3, 1, 1)
        plt.plot(spectra[i].mwave[:, 0])
        plt.title(files[i])
        plt.xlabel('Frame #')
        plt.ylabel('Peak Wavelength')

        # Plot 2: Intensity
        plt.subplot(3, 1, 2)
        plt.plot(spectra[i].maxint[:, 0])
        plt.xlabel('Frame #')
        plt.ylabel('Intensity')

        # Plot 3: Width
        plt.subplot(3, 1, 3)
        plt.plot(spectra[i].width[:, 0] * 1000)
        plt.xlabel('Frame #')
        plt.ylabel('Width (pm)')

        # Save figure
        base_name, _ = os.path.splitext(files[i])
        plt.savefig(f"{base_name}.png", dpi=600)
    except Exception as e:
        print(f"Error visualizing file {i + 1}: {e}")

plt.show()
