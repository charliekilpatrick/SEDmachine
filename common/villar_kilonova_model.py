import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d, interp1d

# From https://arxiv.org/pdf/1710.11576.pdf
# eqn. 1
def rprocess_lumin(time, mass):

    # Assume time in days
    lum = 4.0e18 * 1.989e33 * mass * (0.5 - 1.0/np.pi * np.arctan((time*86400-1.3)/0.11))**1.3

    return(lum)

# eqn. 2
def rprocess_efficiency(time, mass, velocity):

    a, b, d = get_abd(mass, velocity)

    eta = 0.36 * (np.exp(-a * time) + np.log(1.0 + 2.0 * b * time**d) / (2.0 * b * time**d))
    mask = time==0.0
    eta[mask]=0.0

    return(eta)

def rprocess_luminosity(time, mass, velocity, kappa):

    lscale = mass * 1.989e33 * 4.0e18 * 0.36
    tscale = (time * 86400.0 - 1.3)/0.11

    a, b, d = get_abd(mass, velocity)

    lum1 = lscale * (0.5 - 1.0/np.pi * np.arctan(tscale))**1.3
    lum2 = np.exp(-a * time) + np.log1p(2.0 * b * time**d)/(2.0 * b * time**d)

    luminosity = lum1 * lum2

    return(luminosity)

# eqn. 3
def rprocess_luminosity1(time, mass, velocity, kappa):

    # Assume kappa cm2 g-1, velocity natural units, mass solar masses
    # yields td in days
    td = 6.6517 * np.sqrt(kappa * mass/velocity)

    # Need to integrate lumin * efficiency * exp(-t**2 / td**2) * t/td dt
    lumin = rprocess_lumin(time, mass)
    eta = rprocess_efficiency(time, mass, velocity)
    term1 = np.exp(-time**2 / td**2)
    integral = [0.0]
    total = 0.0
    for i,t in enumerate(time):
        if i==0: continue
        dt = t-time[i-1]
        total += lumin[i] * eta[i] * term1[i] * t/td * dt
        integral.append(total)

    lbol = term1 * np.array(integral)
    mask = lbol==0.0
    lbol[mask]=np.min(lbol[lbol>0.0])

    return(lbol)

def tfloor_method(kappa):

    mkappa = np.array([np.log10(0.5), np.log10(3.0), np.log10(10.0)])
    mtfloor = np.array([np.log10(674.), np.log10(1307.), np.log10(3745.)])

    f = interp1d(mkappa, mtfloor)

    tfloor = 10**f(np.log10(kappa))

    return(tfloor)

# eqn. 4
def temperature(time, mass, velocity, kappa, tfloor=None):

    if tfloor is not None:
        tfl = tfloor
    else:
        tfl = tfloor_method(kappa)

    lbol = rprocess_luminosity(time, mass, velocity, kappa)
    tphot = 1.2026e-7 * (lbol/(velocity * time)**2)**0.25

    mask = tphot < tfl
    tphot[mask] = tfl

    return(tphot)

# From Table 1 of https://arxiv.org/pdf/1605.07218.pdf
# Assume solar masses and fractional speed of light for mass and velocity
def get_abd(mass, velocity):

    barnes_v = np.asarray([0.1, 0.2, 0.3, 0.4])
    barnes_M = np.asarray([1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1])
    barnes_a = np.asarray([[2.01, 4.52, 8.16, 16.3], [0.81, 1.9, 3.2, 5.0],
                              [0.56, 1.31, 2.19, 3.0], [.27, .55, .95, 2.0],
                              [0.20, 0.39, 0.65, 0.9]])
    barnes_b = np.asarray([[0.28, 0.62, 1.19, 2.4], [0.19, 0.28, 0.45, 0.65],
                              [0.17, 0.21, 0.31, 0.45], [0.10, 0.13, 0.15, 0.17],
                              [0.06, 0.11, 0.12, 0.12]])
    barnes_d = np.asarray([[1.12, 1.39, 1.52, 1.65], [0.86, 1.21, 1.39, 1.5],
                              [0.74, 1.13, 1.32, 1.4], [0.6, 0.9, 1.13, 1.25],
                              [0.63, 0.79, 1.04, 1.5]])

    therm_func_a = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_a, bounds_error=False, fill_value=None)
    therm_func_b = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_b, bounds_error=False, fill_value=None)
    therm_func_d = RegularGridInterpolator(
            (barnes_M, barnes_v), barnes_d, bounds_error=False, fill_value=None)

    ma = therm_func_a([mass, velocity])[0]
    mb = therm_func_b([mass, velocity])[0]
    md = therm_func_d([mass, velocity])[0]

    return(ma, mb, md)
