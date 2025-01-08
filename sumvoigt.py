import numpy as np

def sumvoigt1(parveci, x):
    """
    Fitting base functions: Voigt profile.
    Parameters:
        parveci: [a, x0, sig, nu]
            a    : Scale
            x0   : Center
            sig  : Width (FWHM)
            nu   : Shape (0=Lorentzian, 1=Gaussian)
        x: ndarray
            Input data points.
    Returns:
        svx: ndarray
            Voigt profile values at input x.
    """
    # Parameters
    a = parveci[0]
    x0 = parveci[1]
    sig = parveci[2]
    nu = parveci[3]

    # Gaussian and Lorentzian components
    f1 = nu * np.sqrt(np.log(2) / np.pi) * np.exp(-4 * np.log(2) * ((x - x0) / abs(sig))**2) / abs(sig)
    f2 = (1 - nu) / (np.pi * abs(sig) * (1 + 4 * ((x - x0) / abs(sig))**2))
    f3 = nu * np.sqrt(np.log(2) / np.pi) / abs(sig) + (1 - nu) / (np.pi * abs(sig))

    # Voigt profile
    svx = a * (f1 + f2) / f3

    return svx
