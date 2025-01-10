import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Voigt profile function
def sumvoigt1(params, x):
    a, x0, sig, nu = params
    f1 = nu * np.sqrt(np.log(2) / np.pi) * np.exp(-4 * np.log(2) * ((x - x0) / abs(sig))**2) / abs(sig)
    f2 = (1 - nu) / (np.pi * abs(sig) * (1 + 4 * ((x - x0) / abs(sig))**2))
    f3 = nu * np.sqrt(np.log(2) / np.pi) / abs(sig) + (1 - nu) / (np.pi * abs(sig))
    return a * (f1 + f2) / f3

# Main function to find lasing peaks
def full_find_lasing_peaks_voigt(max_no_peaks, min_peak_prominence, lower_lambda, upper_lambda, wavelength, data):
    # Combine wavelength and data into a single array
    timeseries = np.column_stack((wavelength, data))

    # Filter wavelengths outside the region of interest
    condition = (timeseries[:, 0] > lower_lambda) & (timeseries[:, 0] < upper_lambda)
    timeseries = timeseries[condition]

    lambda_vals = timeseries[:, 0]  # Wavelengths
    spectra = timeseries[:, 1:]    # Spectra data
    n_data = spectra.shape[1]      # Number of datasets

    # Outputs
    lasing_spectra = np.zeros((n_data, max_no_peaks))
    lasing_intensity = np.zeros((n_data, max_no_peaks))
    lasing_error = np.zeros((n_data, max_no_peaks))
    lasing_width = np.zeros((n_data, max_no_peaks))
    lasing_width_error = np.zeros((n_data, max_no_peaks))

    for m in range(n_data):  # Iterate through datasets
        spectrum = spectra[:, m]

        # Check for lasing
        if np.max(spectrum) < 5:
            print(f"No lasing detected in dataset number: {m + 1}")
            continue

        # Find peaks
        locs, properties = find_peaks(spectrum, distance=1.0, prominence=min_peak_prominence)
        pks = spectrum[locs]
        w = properties['widths']
        n = len(pks)

        if n == 0:
            continue

        # Fit peaks with Voigt profiles
        for i in range(n):
            # Define fitting window
            if n > 1:
                a = np.diff(np.concatenate(([locs[0]], locs, [locs[-1]]))) / 6
                lower_limit = max(lambda_vals[0], lambda_vals[locs[i]] - a[i])
                upper_limit = min(lambda_vals[-1], lambda_vals[locs[i]] + a[i + 1])
            else:
                lower_limit = lower_lambda
                upper_limit = upper_lambda

            mask = (lambda_vals > lower_limit) & (lambda_vals < upper_limit)
            X = lambda_vals[mask]
            Y = spectrum[mask]

            # Initial guess
            p0 = [pks[i], lambda_vals[locs[i]], w[i] / 2.355, 0.5]

            # Bounds
            lb = [pks[i] * 0.95, lower_limit, w[i] / 2.355 / 5, 0]
            ub = [pks[i] * 1.05, upper_limit, w[i] / 2.355 * 5, 1]

            try:
                popt, pcov = curve_fit(
                    lambda x, a, x0, sig, nu: sumvoigt1([a, x0, sig, nu], x),
                    X, Y, p0=p0, bounds=(lb, ub)
                )

                # Calculate confidence intervals
                perr = np.sqrt(np.diag(pcov))

                lasing_intensity[m, i] = popt[0]
                lasing_spectra[m, i] = popt[1]
                lasing_width[m, i] = popt[2]
                lasing_error[m, i] = perr[1]
                lasing_width_error[m, i] = perr[2]
            except RuntimeError:
                print(f"Fit failed for dataset {m + 1}, peak {i + 1}")

    return lasing_spectra, lasing_intensity, lasing_error, lasing_width, lasing_width_error
