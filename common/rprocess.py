import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d, interp1d
import scipy.integrate

# See: https://github.com/guillochon/MOSFiT/blob/master/mosfit/modules/engines/rprocess.py
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

def rprocess_luminosity1(time, mass, velocity, kappa):
    """
    See: https://arxiv.org/pdf/1710.11576.pdf
    This is an adaptation of eqns 1 and 2 from above, also in rprocess mosfit 
    module.

    time = time from merger in days
    mass = ejecta mass of merger in Msun
    velocity = ejecta velocity of merger in c
    kappa = ejecta opacity of merger in cm2 g-1

    Returns the initial rprocess luminosity in erg s-1 that is integrated below
    with the diffusion in rprocess_luminosity function.
    """

    tscale = (time * 86400.0 - 1.3)/0.11

    a, b, d = get_abd(mass, velocity)

    lum1 = mass * 1.989e33 * 4.0e18 * (0.5 - 1.0/np.pi * np.arctan(tscale))**1.3
    lum2 = 0.36 * (np.exp(-a * time) + np.log1p(2.0 * b * time**d)/(2.0 * b * time**d))

    luminosity = lum1 * lum2

    return(luminosity)

# Integrate yy along xx
def int_func(yy, xx):

    integral = [0.0]
    total = 0.0
    for i,x in enumerate(xx):
        if i==0: continue
        dx = x-xx[i-1]
        total += yy[i] * dx
        integral.append(total)

    integral = np.array(integral)
    return(integral)

# eqn. 3
def rprocess_luminosity(time, mass, velocity, kappa):
    """
    See: https://arxiv.org/pdf/1710.11576.pdf
    This is an adaptation of eqns 1 and 2 from above, also in rprocess mosfit 
    module.

    time = time from merger in days
    mass = ejecta mass of merger in Msun
    velocity = ejecta velocity of merger in c
    kappa = ejecta opacity of merger in cm2 g-1

    Returns the bolometric luminosity from an analytic light curve with these 
    parameters in erg s-1.
    """ 

    # Need to integrate from the initial light curve, so start at a small 
    # initial time.  Default is ~0.864 seconds.
    t_min = -5.0
    if np.log10(np.min(time))<t_min:
        t_min = np.log10(np.min(time))

    # Time array that we will integrate in units of days
    int_t = np.logspace(t_min, np.log10(np.max(time)), 10000)

    # Diffusion timescale ($t_{d}$ in Villar+2017)
    td = 6.55923919 * np.sqrt(kappa * mass/velocity)
    td2 = td**2

    # This is basically L_in * eta in Villar+2017
    luminosity = rprocess_luminosity1(int_t, mass, velocity, kappa)
    
    # Multiply by the time array and scaled diffusion time to derive our 
    # argument of integration.
    int_arg = luminosity * int_t * np.exp(-int_t**2 / td2)

    int_l = int_func(int_arg, int_t)
    int_l *= 2.0 * np.exp(-int_t**2 / td2) / td2

    linterp = interp1d(int_t, int_l)
    lbol = linterp(time)

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

    ma = therm_func_a([mass, velocity])[0]
    mb = therm_func_b([mass, velocity])[0]
    md = therm_func_d([mass, velocity])[0]

    return(ma, mb, md)
