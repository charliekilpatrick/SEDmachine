#!/usr/bin/env python3

"""
By C. D. Kilpatrick 2019-05-08
"""

import sys
import os
import astropy
import h5py
import bisect
import warnings
import scipy
import glob
import re
import copy
import json
import numpy as np
import astropy.constants
import astropy.units as u
from astropy.io import fits, ascii
from astropy.table import Table
import scipy.integrate
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt

import pysynphot as S
S.setref(area = 25.0 * 10000)

A_TO_CM = u.angstrom.to(u.centimeter)
A_TO_MU = u.angstrom.to(u.micron)
MPC_TO_CM = u.megaparsec.to(u.centimeter)
C_CGS = astropy.constants.c.cgs.value
DAY_TO_S = u.day.to(u.second)

# Convert normalized pysynphot.blackbody to solar luminosity flux units
BB_SCALE = 1.4125447e+59
L_SUN = 3.826e33

# Internal dependency
from common import grb
from common import rprocess


def is_number(x):

    try:
        float(x)
        return(True)
    except:
        return(False)

class sedmachine(object):
    def __init__(self, distance=1.0e-5):

        # Initialize times, frequency, luminosity
        self.times   = None
        self.nu      = None
        self.Lnu_all = None
        self.meta    = {}

        self.distance = distance

        # time parameters
        self.start_time = 0.001
        self.end_time = 1000.0
        self.ntimes = 15000
        self.time_scale = 'log'

        # wave set
        S.setref(waveset=(10.0,1000000.0,15000))
        test = S.BlackBody(5000)
        self.waves = test.wave

        # Bandpass
        self.bandpass = {}
        self.bandpass_names = []

        self.model_params = {}

        # Photometry tables for different phases, object types, models, etc.
        self.phottables = {}

        if 'PANDEIA' in os.environ.keys():
            self.pandeia = os.environ['PANDEIA']
        else:
            print('WARNING: set PANDEIA environmental variable to '+\
                'pandeia path if you want to use JWST filters.')
            self.pandeia = None

        if 'PYSYN_CDBS' not in os.environ.keys():
            raise Exception(f'ERROR: install pysynphot CDBS'+\
                'see: https://pysynphot.readthedocs.io/en/latest/index.html')

        # Generic options that might need to be changed
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        self.options = {
            'magsystem': 'abmag',
            'time':self.time_scale,
            'dirs': {
                'filter': os.path.join(parent_dir, 'data/filters'),
                'models': os.path.join(parent_dir, 'data/models'),
                'figures': os.path.join(parent_dir, 'output/figures'),
                'tables': os.path.join(parent_dir, 'output/tables'),
            }
        }

        # For loading GRB models
        grb_model = os.path.join(self.options['dirs']['models'],'Table.h5')
        self.grb_generator = grb.FluxGeneratorClass(grb_model, True, 'tau')

        # Make dirs
        for directory in self.options['dirs'].keys():
            if not os.path.exists(self.options['dirs'][directory]):
                os.makedirs(self.options['dirs'][directory])

        # These are the default filters that we want to model
        self.filters = {
                'hst': {
                    'wfc3,uvis1': ['f200lp','f300x','f350lp','f475x','f600lp',
                                 'f850lp','f218w','f225w','f275w','f336w',
                                 'f390w','f438w','f475w','f555w','f606w',
                                 'f625w','f775w','f814w'],
                    'wfc3,ir': ['f105w','f110w','f125w','f140w','f160w'],
                    'acs': {
                        'wfc': ['f555w','f775w','f625w','f550m','f850lp',
                                'f606w','f475w','f814w','f435w'],
                        'sbc': ['f115lp','f125lp','f140lp','f150lp','f165lp',
                                'f122m']
                    },
                    'wfpc2': []
                },
                'jwst': {
                    'nircam': ['f070w','f090w','f115w','f150w','f150w2','f200w',
                        'f277w','f322w2','f356w','f444w'],
                    'miri': ['f0560w','f0770w','f1000w'],
                },
                'swift': ['w2','m2','w1','U','B','V','W'],
                'spitzer': ['ch1','ch2','ch3','ch4'],
                'sdss': ['u','g','r','i','z'],
                'johnson': ['U','B','V','R','I'],
                'kpno': ['J', 'H', 'K'],
                'ps1': ['g','r','i','z','Y','w'],
                'clear': ['Clear'],
                'ukirt': ['J','H','K'],
                'atlas': ['c','o']
        }

    def add_options(self):

        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--model-file',default=None,type=str,
            help='Input model file for sedmachine.py')
        parser.add_argument('--models',default=None,nargs='*',type=str,
            help='Model types to generate [grb,grb_angle,grb170817a,kilonova].')
        parser.add_argument('--filters',type=str,default=None,
            help='Override default to output all filters with input value.')
        parser.add_argument('--wfc3',default=False,action='store_true',
            help='Add all WFC3 UVIS and IR bands to output.')
        parser.add_argument('--theta-obs',default=0.0,type=float,
            help='Observation angle for GRB jet we are observing.')
        parser.add_argument('--kappa',default=1.0,type=float,
            help='Opacity for Villar kilonova model.')
        parser.add_argument('--tfloor',default=None,type=float,
            help='Use this value as temperature floor (T_c) for Villar model.')
        parser.add_argument('--clobber',default=False,action='store_true',
            help='Clobber previous versions of model files.')
        parser.add_argument('--instruments',default=None,type=str,
            help='Comma-separated list of instruments to generate bandpass.')
        parser.add_argument('--armin',default=False,action='store_true',
            help='Armin likes the output in this specific format.  '+\
            'Flag to have it output this way.')
        parser.add_argument('--flat-ascii',default=False,action='store_true',
            help='Output as flat ascii file.')
        # Time parameters
        parser.add_argument('--start-time',default=0.001,type=float,
            help='Start time for model light curves.')
        parser.add_argument('--end-time',default=100.0,type=float,
            help='End time for model light curves.')
        parser.add_argument('--ntimes',default=400,type=int,
            help='Number of times in time array for model light curves.')
        parser.add_argument('--time-scale', default='linear',type=str,
            help='Scale to use for time axis.')
        parser.add_argument('--distance',default=1.0e-5,type=float,
            help='Distance for constructing the light curves.')

        args = parser.parse_args()

        return(args)

    def get_ab_to_vega(self, bandpass):

        flat = np.zeros(len(self.waves))+1.0
        test = S.ArraySpectrum(self.waves, flat)
        kwargs = {'force': 'taper', 'binset': self.waves}
        obs = S.Observation(test, bandpass, **kwargs)

        # AB to Vega conversion is Mag_Vega - Mag_AB
        ab_to_vega = obs.effstim('vegamag')-obs.effstim('abmag')

        return(ab_to_vega)

    # Calculates extinction in input bandpass given Av and Rv
    def calculate_extinction(self, Av, Rv, bandpass):

        test = self.extinction_law(self.waves, Av, Rv)
        flat = np.zeros(len(self.waves))+1.0

        test1 = S.ArraySpectrum(self.waves, flat)
        test2 = S.ArraySpectrum(self.waves, flat*test.flux)

        kwargs = {'force': 'taper', 'binset': self.waves}

        obs1 = S.Observation(test1, bandpass, **kwargs)
        obs2 = S.Observation(test2, bandpass, **kwargs)

        # This is extinction in bandpass
        a_lambda = obs2.effstim('abmag')-obs1.effstim('abmag')

        return(a_lambda)

    def extinction_law(self, wave, Av, Rv, deredden=False):

        # Inverse wavelength dependent quantities
        x=1./(wave*1.0e-4)
        y=x-1.82

        # Variables proportional to extinction
        a=np.zeros(len(wave))
        b=np.zeros(len(wave))
        Fa=np.zeros(len(wave))
        Fb=np.zeros(len(wave))

        # Masks to apply piecewise Cardelli et al. function
        mask1 = np.where(x < 1.1)
        mask2 = np.where((x > 1.1) & (x < 3.3))
        mask3 = np.where(x > 3.3)
        mask4 = np.where(x > 5.9)

        x=1./(wave[mask1]*1.0e-4)
        y=x-1.82

        a[mask1]=0.574 * x**1.61
        b[mask1]=-0.527 * x**1.61

        x=1./(wave[mask2]*1.0e-4)
        y=x-1.82

        a[mask2]=1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 +\
            0.72085 * y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b[mask2]=1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 -\
            5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7

        x=1./(wave[mask4]*1.0e-4)
        y=x-1.82

        Fa[mask4]=-0.04473 * (x-5.9)**2 - 0.009779 * (x-5.9)**3
        Fb[mask4]=0.2130 * (x-5.9)**2 + 0.1207 * (x-5.9)**3

        x=1./(wave[mask3]*1.0e-4)
        y=x-1.82

        a[mask3]=1.752 - 0.316 * x - 0.104/((x-4.67)**2 + 0.341) + Fa[mask3]
        b[mask3]=-3.090 + 1.825 * x + 1.206/((x-4.62)**2 + 0.263) + Fb[mask3]

        Alam = Av*(a+b/Rv)
        elam = 10**(-0.4 * Alam)
        if deredden: elam = 1./elam

        sp = S.ArraySpectrum(wave, elam, fluxunits='count')

        return(sp)

    def create_blackbody(self, lum, temp):
        if np.isnan(temp): temp=0.0
        if np.isnan(lum): lum=0.0

        bb = S.BlackBody(temp)

        # Normalize blackbody
        bb.convert('flam')
        scale = 1.0/scipy.integrate.simps(bb.flux, bb.wave)
        bb = bb * scale * lum * L_SUN
        bb.convert('fnu')

        return(bb)

    def generate_villar_model(self, parameters={}):
        P = {
            'mass': 0.023,
            'kappa': 0.5,
            'velocity': 0.256,
        }

        P.update(parameters)

        kappa = P['kappa']
        mass = P['mass']
        velo = P['velocity']
        if 'tfloor' in P.keys():
            tfloor = P['tfloor']
        else:
            tfloor = None

        luminosity = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor)

        return(luminosity, temperature)

    def load_villar_2comp_model(self, parameters={}):
        """
        Generate analytic Villar et al. (2017) spectral models and transform
        into L_nu and wavelength (cgs) for input distance, and save in arrays.
        Uses a 2 component model.
        """

        P = {
            'mass1': 0.023,
            'kappa1': 0.5,
            'velocity1': 0.256,
            'mass2': 0.050,
            'kappa2': 3.65,
            'velocity2': 0.149,
        }

        P.update(parameters)

        self.model_params.update(P)

        # Get temperatures and luminosities for given model parameters
        kappa = P['kappa1']
        mass = P['mass1']
        velo = P['velocity1']
        if 'tfloor1' in P.keys():
            tfloor1 = P['tfloor1']
        else:
            tfloor1 = None

        luminosity1 = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature1 = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor1)

        kappa = P['kappa2']
        mass = P['mass2']
        velo = P['velocity2']
        if 'tfloor2' in P.keys():
            tfloor2 = P['tfloor2']
        else:
            tfloor2 = None

        luminosity2 = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature2 = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor2)

        # Now create spectral luminosity using blackbodies
        self.nu      = None
        self.Lnu_all = None

        for i in np.arange(len(self.times)):
            lum1 = luminosity1[i]/(3.839e33)
            temp1 = temperature1[i]

            lum2 = luminosity2[i]/(3.839e33)
            temp2 = temperature2[i]

            bb1 = self.create_blackbody(lum1, temp1)
            bb2 = self.create_blackbody(lum2, temp2)

            if self.nu is None:
                self.nu      = C_CGS / (bb1.wave * A_TO_CM)
            if self.Lnu_all is None:
                self.Lnu_all = np.zeros((len(self.times), len(self.nu)))

            self.Lnu_all[i,:] = bb1.flux + bb2.flux

    def load_villar_3comp_model(self, parameters={}):
        """
        Generate analytic Villar et al. (2017) spectral models and transform
        into L_nu and wavelength (cgs) for input distance, and save in arrays.
        Uses a 2 component model.
        """

        # Generic, 0.05 Msun purple kilonova model with velocity = 0.15c
        P = {
            'mass1': 0.020,
            'kappa1': 0.5,
            'velocity1': 0.266,
            'mass2': 0.047,
            'kappa2': 3.0,
            'velocity2': 0.152,
            'mass3': 0.011,
            'kappa3': 10.0,
            'velocity3': 0.137,
        }

        P.update(parameters)

        self.model_params.update(P)

        # Get temperatures and luminosities for given model parameters
        kappa = P['kappa1']
        mass = P['mass1']
        velo = P['velocity1']
        if 'tfloor1' in P.keys():
            tfloor1 = P['tfloor1']
        else:
            tfloor1 = None

        luminosity1 = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature1 = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor1)

        kappa = P['kappa2']
        mass = P['mass2']
        velo = P['velocity2']
        if 'tfloor2' in P.keys():
            tfloor2 = P['tfloor2']
        else:
            tfloor2 = None

        luminosity2 = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature2 = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor2)

        kappa = P['kappa3']
        mass = P['mass3']
        velo = P['velocity3']
        if 'tfloor3' in P.keys():
            tfloor3 = P['tfloor3']
        else:
            tfloor3 = None

        luminosity3 = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature3 = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor3)

        # Now create spectral luminosity using blackbodies
        self.nu      = None
        self.Lnu_all = None

        for i in np.arange(len(self.times)):
            lum1 = luminosity1[i]/(3.839e33)
            temp1 = temperature1[i]

            lum2 = luminosity2[i]/(3.839e33)
            temp2 = temperature2[i]

            lum3 = luminosity3[i]/(3.839e33)
            temp3 = temperature3[i]

            bb1 = self.create_blackbody(lum1, temp1)
            bb2 = self.create_blackbody(lum2, temp2)
            bb3 = self.create_blackbody(lum3, temp3)

            if self.nu is None:
                self.nu      = C_CGS / (bb1.wave * A_TO_CM)
            if self.Lnu_all is None:
                self.Lnu_all = np.zeros((len(self.times), len(self.nu)))

            self.Lnu_all[i,:] = bb1.flux + bb2.flux + bb3.flux

    def load_villar_model(self, parameters={}):
        """
        Generate analytic Villar et al. (2017) spectral models and transform
        into L_nu and wavelength (cgs) for input distance, and save in arrays
        """

        # Generic, 0.05 Msun purple kilonova model with velocity = 0.15c
        P = {
            'mass': 0.05,
            'velocity': 0.15,
            'kappa': 3.0,
        }

        P.update(parameters)

        self.model_params.update(P)

        # Get temperatures and luminosities for given model parameters
        kappa = P['kappa']
        mass = P['mass']
        velo = P['velocity']
        if 'tfloor' in P.keys():
            tfloor = P['tfloor']
        else:
            tfloor = None

        luminosity = rprocess.rprocess_luminosity(self.times, 
            mass, velo, kappa)
        temperature = rprocess.temperature(self.times,
            mass, velo, kappa, tfloor=tfloor)

        # Now create spectral luminosity using blackbodies
        self.nu      = None
        self.Lnu_all = None

        for i in np.arange(len(self.times)):
            lum = luminosity[i]/(3.839e33)
            temp = temperature[i]

            bb = self.create_blackbody(lum, temp)

            if self.nu is None:
                self.nu      = C_CGS / (bb.wave * A_TO_CM)
            if self.Lnu_all is None:
                self.Lnu_all = np.zeros((len(self.times), len(self.nu)))

            self.Lnu_all[i,:] = bb.flux

    def load_kasen_model(self, model):
        """
        Read in Kasen et al. (2017) spectral models, transform to wavelength
        in angstrom and flam (cgs) for input distance, and save in arrays
        """

        # Read Kasen et al. 2017 spectral model and return the base arrays
        fin  = h5py.File(os.path.join('models',model), 'r')

        # frequency in Hz
        nu    = np.array(fin['nu'],dtype='d')
        # array of time in seconds
        times = np.array(fin['time'])
        # covert time to days
        times *= u.second.to(u.day)

        # specific luminosity (ergs/s/Hz)
        # this is a 2D array, Lnu[times][nu]
        Lnu_all   = np.array(fin['Lnu'],dtype='d')

        self.times   = times
        self.nu      = nu
        self.Lnu_all = Lnu_all

    def load_double_kasen_model(self, model1, model2):
        """
        Same as above, except combining two Kasen models instead of one.  These
        are the methods used in Kilpatrick et al. 2017.
        """

        # Read in both models
        f1 = os.path.join(self.options['dirs']['models'],model1)
        f2 = os.path.join(self.options['dirs']['models'],model2)
        fin1 = h5py.File(f1, 'r')
        fin2 = h5py.File(f2, 'r')

        # Convert frequency and time info
        nu1 = np.array(fin1['nu'], dtype='d')
        nu2 = np.array(fin2['nu'], dtype='d')

        times1 = np.array(fin1['time'])
        times2 = np.array(fin2['time'])

        times1 *= u.second.to(u.day)
        times2 *= u.second.to(u.day)

        # specific luminosity (ergs/s/Hz)
        # this is a 2D array, Lnu[times][nu]
        Lnu1_all   = np.array(fin1['Lnu'],dtype='d')
        Lnu2_all   = np.array(fin2['Lnu'],dtype='d')

        # Now we need to add these models together by flux.  Use model1 to
        # define frequencies and times and then get indices for model2
        self.times   = times1
        self.nu      = nu1
        self.Lnu_all = Lnu1_all

        # Add on flux from model2
        for i,t in enumerate(self.times):
            idx1 = bisect.bisect(times2, t)
            for j,f in enumerate(self.nu):
                idx2 = bisect.bisect(nu2, f)

                self.Lnu_all[i, j] += Lnu2_all[idx1, idx2]

    def create_time_array(self):

        # Generate a spectrum for the default waveset and input phase
        Ntimes = self.ntimes
        MAX_TIME = self.end_time
        MIN_TIME = self.start_time

        if self.options['time']=='linear':
            self.times = (MAX_TIME - MIN_TIME)/Ntimes * np.arange(0, Ntimes+1)+\
                MIN_TIME
        elif self.options['time']=='log':
            self.times   = 10**((np.log10(MAX_TIME)-np.log10(MIN_TIME))/Ntimes*\
                np.arange(0, Ntimes + 1) + np.log10(MIN_TIME))

    def load_sed(self, model, phase=0, waveunit=u.angstrom, fluxunit=u.cgs):
        """
        Load a single epoch of a SED.  Default phase is 0 (this can be ignored
        if phase doesn't matter).  Model is assumed to be a 2 column file with a
        wavelength column and flux column.
        """

        # Assume the SED is simply a table with some type of wavelength and some
        # type of flux.  The user will need to identify the units on wavelength
        # and the units on flux in the options, but default is to assume they
        # are angstroms and erg/s/cm2/angstrom.

        table = ascii.read(model, names=['wave', 'flux'])
        wavelength = [w * waveunit.to(u.angstrom) for w in table['wave']]
        flux = [f * fluxunit.to(u.cgs) for f in table['flux']]

        if self.times is None:
            self.times = [phase]
            self.wave = [wavelength]
            self.flux = [flux]
        else:
            self.times.append(phase)
            self.wave.append(wavelength)
            self.flux.append(flux)

    def load_grb(self, parameters={}):
        # Default parameters
        # See Fig. 13 in https://arxiv.org/pdf/2104.02070.pdf
        P = {
            'E': 10**0.40,
            'Eta0': 8.02,
            'GammaB': 12.0,
            'dL': 0.012188,
            'epsb': 10**-5.17,
            'epse': 0.1,
            'n': 1.0e-2,
            'p': 2.15,
            'theta_obs':0.44,
            'xiN': 1.0,
            'z': 0.0,
        }

        # Update parameters if there are any provided
        P.update(parameters)

        if 'E_kiso' in P.keys() and 'GammaB' in P.keys():
            theta0 = 1./P['GammaB']
            E = 0.5 * P['E_kiso'] * (1 - np.cos(theta0/2))
            P['E'] = E

        self.model_params.update(P)
        if self.distance!=1.0e-5:
            self.model_params['dL']=self.distance * 0.0003085678

        self.nu      = C_CGS / (self.waves * A_TO_CM)
        self.Lnu_all = np.zeros((len(self.times), len(self.nu)))

        # Construct dummy arrays to hold these data in 1D
        all_nu = np.array(list(self.nu) * len(self.times), dtype=np.longdouble)
        all_times = np.array(sum([[t * DAY_TO_S] * len(self.nu)
            for t in self.times], []), dtype=np.longdouble)
        # Output fluxes are in units of mJy
        all_flux = self.grb_generator.GetSpectral(all_times, all_nu, P)

        all_flux = np.array(all_flux)

        # Mask out bad flux values
        all_flux[np.isnan(all_flux)] = 0.0
        # Converting mJy to erg/s/Hz given input distance dL in 1.0e28 cm units
        all_flux = all_flux * 1.0e-26 * 4 * np.pi * (P['dL'] * 1.0e28)**2

        # Reshape the flux array into a 2D array for time and nu
        self.Lnu_all = np.reshape(all_flux, (len(self.times), len(self.nu)))

    # Get all filters from Pandeia directory (see pandeia variable above)
    def get_jwst_filters(self, filtnam):

        tel, inst, filt = filtnam.split(',')
        filt = filt.lower()
        inst = inst.lower()

        globstr=os.path.join(self.pandeia, 'jwst', inst,
            'filters', f'*{filt}_trans*')
        filts = glob.glob(globstr)

        if len(filts)!=1:
            return(None)

        f = filts[0]
            
        base = os.path.split(f)[1]
        name = inst.lower()+'_'+base.split('_')[1]
        hdu = fits.open(f)
        wave = np.array([float(el[0])*1.0e4 for el in hdu[1].data])
            
        tran = np.array([float(el[1]) for el in hdu[1].data])
            
        # Only consider wavelengths < 30 microns
        mask = wave < 30.0*1.0e4
        wave = wave[mask]
        tran = tran[mask]
            
        bp = S.ArrayBandpass(wave, tran, name=name, waveunits='angstrom')
            
        return(bp)
        
    def load_passbands(self, passbands=None, instruments=None, wfc3=False):
        """
        Load passbands either from pysynphot or check for an equivalent file in
        the default filter directory.
        """

        # Load default filters first
        filters = self.filters
        message = 'Loading {source} bandpasses...'
        if 'hst' in filters.keys() and 'wfc3,uvis1' in filters['hst'].keys():
            print(message.format(source='wfc3,uvis'))
            for bp in filters['hst']['wfc3,uvis1']:
                name = 'wfc3,uvis1,'+bp
                add_name = name.replace(',','_')
                if not wfc3:
                    if passbands:
                        if bp not in passbands:
                            continue
                    if instruments:
                        if 'wfc3_uvis' not in instruments:
                            continue
                bpmodel = S.ObsBandpass(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'hst' in filters.keys() and 'wfc3,ir' in filters['hst'].keys():
            print(message.format(source='wfc3,ir'))
            for bp in filters['hst']['wfc3,ir']:
                name = 'wfc3,ir,'+bp
                add_name = name.replace(',','_')
                if not wfc3:
                    if passbands:
                        if bp not in passbands:
                            continue
                    if instruments:
                        if 'wfc3_ir' not in instruments:
                            continue
                bpmodel = S.ObsBandpass(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'hst' in filters.keys() and 'acs,wfc' in filters['hst'].keys():
            print(message.format(source='acs,wfc'))
            for bp in filters['hst']['acs']['wfc']:
                name = 'acs,wfc1,'+bp
                add_name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'acs_wfc' not in instruments:
                        continue
                bpmodel = S.ObsBandpass(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'hst' in filters.keys() and 'acs,sbc' in filters['hst'].keys():
            print(message.format(source='acs,sbc'))
            for bp in filters['hst']['acs']['sbc']:
                name = 'acs,sbc,'+bp
                add_name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'acs_sbc' not in instruments:
                        continue
                bpmodel = S.ObsBandpass(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'jwst' in filters.keys() and 'nircam' in filters['jwst'].keys():
            print(message.format(source='jwst,nircam'))
            for bp in filters['jwst']['nircam']:
                name = 'jwst,nircam,'+bp
                add_name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'jwst_nircam' not in instruments:
                        continue
                bpmodel = self.get_jwst_filters(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'jwst' in filters.keys() and 'miri' in filters['jwst'].keys():
            print(message.format(source='jwst,miri'))
            for bp in filters['jwst']['miri']:
                name = 'jwst,miri,'+bp
                add_name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'jwst_miri' not in instruments:
                        continue
                bpmodel = self.get_jwst_filters(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'swift' in filters.keys():
            print(message.format(source='swift,uvot'))
            for bp in filters['swift']:
                file = self.options['dirs']['filter']+'/SWIFT.'+bp.upper()+'.dat'
                name = 'swift,uvot,'+bp
                name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'swift_uvot' not in instruments:
                        continue
                wave,transmission = np.loadtxt(file, unpack=True)
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=name, waveunits='Angstrom')
                self.bandpass[name] = bpmodel
                self.bandpass_names.append(name)
        if 'swope' in filters.keys():
            print(message.format(source='swope'))
            for bp in filters['swope']:
                file = self.options['dirs']['filter']+'/SWOPE.'+bp+'.dat'
                name = 'swope,'+bp
                name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'swift_uvot' not in instruments:
                        continue
                wave,transmission = np.loadtxt(file, unpack=True)
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=name, waveunits='Angstrom')
                self.bandpass[name] = bpmodel
                self.bandpass_names.append(name)
        if 'johnson' in filters.keys():
            print(message.format(source='johnson'))
            for bp in filters['johnson']:
                name = 'johnson,'+bp
                add_name = name.replace(',','_')
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'johnson' not in instruments:
                        continue
                bpmodel = S.ObsBandpass(name)
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'sdss' in filters.keys():
            print(message.format(source='sdss'))
            for bp in filters['sdss']:
                name = 'sdss,'+bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'sdss' not in instruments:
                        continue
                bpmodel = S.ObsBandpass(name)
                add_name = name.replace(',','_')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'kpno' in filters.keys():
            print(message.format(source='kpno'))
            for bp in filters['kpno']:
                name = 'kpno,'+bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'kpno' not in instruments:
                        continue
                bpmodel = S.ObsBandpass(name)
                add_name = name.replace(',','_')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'ps1' in filters.keys():
            print(message.format(source='PS1'))
            for bp in filters['ps1']:
                name = 'PS1,'+bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'ps1' not in instruments:
                        continue
                file = os.path.join(self.options['dirs']['filter'],
                    f'PS1.GPC1.{bp}.dat')
                wave,transmission = np.loadtxt(file, unpack=True)
                add_name = name.replace(',','_')
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=add_name, waveunits='Angstrom')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'atlas' in filters.keys():
            print(message.format(source='ATLAS'))
            for bp in filters['atlas']:
                name = 'ATLAS,'+bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'ps1' not in instruments:
                        continue
                file = os.path.join(self.options['dirs']['filter'],
                    f'ATLAS.{bp}.dat')
                wave,transmission = np.loadtxt(file, unpack=True)
                add_name = name.replace(',','_')
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=add_name, waveunits='Angstrom')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'ukirt' in filters.keys():
            print(message.format(source='UKIRT'))
            for bp in filters['ukirt']:
                name = 'ukirt,'+bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'ukirt' not in instruments:
                        continue
                file = os.path.join(self.options['dirs']['filter'],
                    f'UKIRT.{bp}.dat')
                wave,transmission = np.loadtxt(file, unpack=True)
                add_name = name.replace(',','_')
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=add_name, waveunits='Angstrom')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)
        if 'clear' in filters.keys():
            print(message.format(source='Clear'))
            for bp in filters['clear']:
                name = bp
                if passbands:
                    if bp not in passbands:
                        continue
                if instruments:
                    if 'clear' not in instruments:
                        continue
                file = os.path.join(self.options['dirs']['filter'],
                    f'clear.dat')
                wave,transmission = np.loadtxt(file, unpack=True)
                add_name = name.replace(',','_')
                bpmodel = S.ArrayBandpass(wave, transmission,
                    name=add_name, waveunits='Angstrom')
                self.bandpass[add_name] = bpmodel
                self.bandpass_names.append(add_name)

    def find_models(self, model_types=[], params={}):
        """
        Get a list of models and metadata from models/ directory
        """

        model_list = []
        if model_types is None:
            return(model_list)


        if 'kilonova' in model_types:
            model_dir = self.options['dirs']['models']

            # Grab all h5 models and resolve their info
            models = glob.glob(model_dir + '/*.h5')

            for model in models:
                # Get model metadata
                data = {}
                if 'knova_d1_n10_' in model and 'fd' not in model and 'vs' not in model:
                    groups = re.findall(
                        'knova_d1_n10_m(.+)_vk(.+)_Xlan(.+)\.h5', model)[0]
                    data['type'] = 'Kasen'
                    data['mass'] = float(groups[0])
                    data['velocity'] = float(groups[1])
                    data['Xlan'] = str(groups[2])
                    data['model']=model
                    data.update(params)
                    model_list.append(data)

        elif 'grb170817a' in model_types:
            data={'E': 0.15869069395227384,
                'Eta0': 7.973477192135503,
                'GammaB': 11.000923300022666,
                'dL': 0.012188,
                'epsb': 0.013323706571267526,
                'epse': 0.04072783842837688,
                'n': 0.0009871221028954489,
                'p': 2.1333493591554804,
                'theta_obs': 0.4769798916899842,
                'xiN': 1.0,
                'z': 0.0,
                'type':'grb'}
            data.update(params)
            model_list=[data]

        elif 'grb' in model_types:

            N_E = 35
            N_n = 35

            # In units of log10(E/1e50 erg)
            MAX_E = 3.0
            MIN_E = -2.0

            # In units of log10(n * cm3)
            MAX_n = 1.0
            MIN_n = -6.0

            for i in np.arange(0, N_E+1):
                for j in np.arange(0, N_n+1):

                    data = {
                        'E': 0.1*0.00043396498*10**0.40,
                        'Eta0': 8.02,
                        'GammaB': 12.0,
                        'dL': 0.012188,
                        'epsb': 10**-5.17,
                        'epse': 0.1,
                        'n': 1.0e-2,
                        'p': 2.15,
                        'theta_obs':0.44,
                        'xiN': 1.0,
                        'z': 0.0,
                    }

                    data['type'] = 'grb'
                    data['E_kiso'] = 10**(MIN_E + (MAX_E - MIN_E) * float(i)/N_E)
                    data['n'] = 10**(MIN_n + (MAX_n - MIN_n) * float(j)/N_n)
                    data['theta_obs'] = 0.0
                    data.update(params)

                    model_list.append(data)

        elif 'villar_gw170817' in model_types:

            model = {
                'type': 'villar_2comp',
                'mass1': 0.023,
                'kappa1': 0.5,
                'velocity1': 0.256,
                'tfloor1': 3983,
                'mass2': 0.050,
                'kappa2': 3.65,
                'velocity2': 0.149,
                'tfloor2': 1151,
            }

            model_list.append(model)

        elif 'villar_gw170817_3comp' in model_types:

            model = {
                'type': 'villar_3comp',
                'mass1': 0.017,
                'kappa1': 0.5,
                'velocity1': 0.282,
                'tfloor1': 741,
                'mass2': 0.038,
                'kappa2': 3.0,
                'velocity2': 0.119,
                'tfloor2': 1164,
                'mass3': 0.013,
                'kappa3': 10.0,
                'velocity3': 0.133,
                'tfloor3': 4145,
            }

            model_list.append(model)

        elif 'villar_gw170817_3comp_mod' in model_types:

            model_file = os.path.join(self.options['dirs']['models'],
                'villar_gw170817_3comp_mod.dat')

            if os.path.exists(model_file):
                with open(model_file,'r') as f:
                    model = json.load(f)
            else:
                model = {
                    'type': 'villar_3comp',
                    'mass1': 0.0156,
                    'kappa1': 0.5,
                    'velocity1': 0.241,
                    'tfloor1': 852,
                    'mass2': 0.0441,
                    'kappa2': 3.0,
                    'velocity2': 0.125,
                    'tfloor2': 1666,
                    'mass3': 0.0083,
                    'kappa3': 10.0,
                    'velocity3': 0.13338,
                    'tfloor3': 3837,
                    'name': 'villar_gw170817_3comp_mcmc'
                }

            model_list.append(model)
            
        elif 'villar' in model_types:

            N_M = 30
            N_v = 30

            # In units of log10(E/1e50 erg)
            MAX_M = -3.0
            MIN_M = -0.30102999566

            # In units of log10(n * cm3)
            MAX_v = 0.03
            MIN_v = 0.50

            for i in np.arange(0, N_M+1):
                for j in np.arange(0, N_v+1):

                    data = {
                        'kappa': 1.0,
                    }

                    data['type'] = 'villar'
                    data['mass'] = 10**(MIN_M + (MAX_M - MIN_M) * float(i)/N_M)
                    data['velocity'] = MIN_v + (MAX_v - MIN_v) * float(j)/N_v
                    data.update(params)

                    model_list.append(data)

        elif 'grb_angle' in model_types:

            N_theta = 30

            # In units of cos_theta
            MAX_sin_theta = 0.99
            MIN_sin_theta = 0.0

            for i in np.arange(0, N_theta+1):

                sin_theta = (MAX_sin_theta-MIN_sin_theta)/N_theta*i

                data = {}
                data['type'] = 'grb'
                data['E'] = 12.0
                data['n'] = 0.004
                data['theta_obs'] = np.arcsin(sin_theta)
                data.update(params)

                model_list.append(data)


        return(model_list)

    def get_model(self, phase):
        """
        Get the flam spectrum for some specific phase
        """
        idx = np.argmin(np.abs(self.times-phase))
        Lnu = self.Lnu_all[idx,:]

        # We want things in Flambda (ergs/s/Angstrom)
        lam  = C_CGS / self.nu / A_TO_CM
        Llam = Lnu * self.nu**2.0 / C_CGS * A_TO_CM
        return(lam, Llam)

    def get_norm_model(self, phase, distance):
        """
        Get the flam spectrum for some specific phase and distance
        """

        lam, flam = self.get_model(phase)
        lamz = lam
        fnorm = flam / (4 * np.pi * (distance * MPC_TO_CM)**2.)
        return(lamz, fnorm)

    def get_mag(self, passband, phase, distance):
        """
        Get magnitude for input passband at phase and distance
        """
        lamz, fnorm = self.get_norm_model(phase, distance)
        spec = S.ArraySpectrum(wave=lamz, flux=fnorm,
            waveunits='angstrom', fluxunits='flam')

        obs = S.Observation(spec, passband, force='taper', binset=self.waves)

        try:
            mag = obs.effstim(self.options['magsystem'])
        except ValueError as e:
            mag = np.nan
        return(mag)

    def empty_table(self, passband_names):
        """
        Make an empty phottable with time and all passbands so we can add to it
        row by row.
        """
        names = ['time'] + [n.replace(',','_') for n in passband_names]
        value = [[0.]] + [[0.]] * len(passband_names)
        table = Table(value, names=names)[:0].copy()
        return(table)

    def load_model_file(self, model_file):

        if not model_file:
            return([])
        elif not os.path.exists(model_file):
            return([])

        try:
            table = Table.read(model_file, format='ascii.commented_header')
        except:
            print(f'WARNING: could not parse model_file {model_file}')
            return([])

        if not all([v in table.keys() for v in ['type','name','models']]):
            print(f'WARNING: {model_file} needs type,name,models columns')
            return([])

        all_models = []
        for row in table:

            model = {}
            model['type']=row['type']
            model['name']=row['name']
            typ=row['type']

            if model['type']=='Kasen':
                model['model']=row['models']
            elif model['type']=='Kasen_double':
                m1, m2 = row['models'].split(',')
                model['model1']=m1
                model['model2']=m2
            elif model['type']=='grb':
                data = json.loads(row['models'])
                model.update(data)
            else:
                print(f'WARNING: could not parse model type {typ}')

            all_models.append(model)

        return(all_models)

    def load_model(self, model):
        typ=model['type']
        if typ=='villar_2comp':
            self.load_villar_2comp_model(parameters=model)
        elif typ=='villar_3comp':
            self.load_villar_3comp_model(parameters=model)
        elif typ=='villar':
            self.load_villar_model(parameters=model)
        elif typ=='Kasen':
            self.load_kasen_model(model['model'])
        elif typ=='grb':
            self.load_grb(parameters=model)
        elif typ=='Kasen_double':
            self.load_double_kasen_model(model['model1'], model['model2'])
        else:
            raise Exception(f'ERROR: unrecognized model type {typ}')

    def get_model_name(self, model):
        if model['type']=='Kasen':
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = model['model'].replace('h5','')

        elif model['type']=='grb':
            E_fmt = '%7.4f' % model['E_kiso']
            n_fmt = '%7.7f' % model['n']
            theta_fmt = '%7.4f' % model['theta_obs']
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = '{type}_{theta_obs}_{E}_{n}'.format(
                    type=model['type'], E=E_fmt.strip(),
                    n=n_fmt.strip(), theta_obs=theta_fmt.strip())

        elif model['type']=='villar':
            M_fmt = '%.4f'%model['mass']
            v_fmt = '%.4f'%model['velocity']
            kappa_fmt = '%.4f'%model['kappa']

            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = '{type}_{mass}_{velocity}_{kappa}'.format(
                    type=model['type'], mass=M_fmt.strip(),
                    velocity=v_fmt.strip(), kappa=kappa_fmt.strip())

        elif model['type']=='villar_2comp':
            M1_fmt = '%.4f'%model['mass1']
            v1_fmt = '%.4f'%model['velocity1']
            kappa1_fmt = '%.4f'%model['kappa1']

            M2_fmt = '%.4f'%model['mass2']
            v2_fmt = '%.4f'%model['velocity2']
            kappa2_fmt = '%.4f'%model['kappa2']

            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = '{type}_{mass1}_{velocity1}_{kappa1}_'+\
                    '{mass2}_{velocity2}_{kappa2}'
                model_name = model_name.format(
                    type=model['type'], mass1=M1_fmt.strip(),
                    velocity1=v1_fmt.strip(), kappa1=kappa1_fmt.strip(), 
                    mass2=M2_fmt.strip(),
                    velocity2=v2_fmt.strip(), kappa2=kappa2_fmt.strip())

        elif model['type']=='villar_3comp':
            M1_fmt = '%.4f'%model['mass1']
            v1_fmt = '%.4f'%model['velocity1']
            kappa1_fmt = '%.4f'%model['kappa1']

            M2_fmt = '%.4f'%model['mass2']
            v2_fmt = '%.4f'%model['velocity2']
            kappa2_fmt = '%.4f'%model['kappa2']

            M3_fmt = '%.4f'%model['mass3']
            v3_fmt = '%.4f'%model['velocity3']
            kappa3_fmt = '%.4f'%model['kappa3']

            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = '{type}_{mass1}_{velocity1}_{kappa1}_'+\
                    '{mass2}_{velocity2}_{kappa2}'+\
                    '{mass3}_{velocity3}_{kappa3}'
                model_name = model_name.format(
                    type=model['type'], mass1=M1_fmt.strip(),
                    velocity1=v1_fmt.strip(), kappa1=kappa1_fmt.strip(), 
                    mass2=M2_fmt.strip(),
                    velocity2=v2_fmt.strip(), kappa2=kappa2_fmt.strip(),
                    mass3=M3_fmt.strip(),
                    velocity3=v3_fmt.strip(), kappa3=kappa3_fmt.strip())

        elif model['type']=='Kasen_double':
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = 'combined_Kasen_model'

        return(model_name)


def main():
    # Make an object with distance
    kn = sedmachine(distance=1.0e-5)

    # load models
    args = kn.add_options()

    # Time parameters
    kn.start_time = args.start_time
    kn.end_time = args.end_time
    kn.ntimes = args.ntimes
    kn.time_scale = args.time_scale
    kn.clobber = args.clobber
    kn.distance = args.distance

    kn.create_time_array()

    # Load filters]
    if args.filters:
        passbands = args.filters.split(',')
    else:
        passbands = None

    if args.instruments:
        instruments = args.instruments.split(',')
    else:
        instruments = None

    kn.load_passbands(passbands=passbands,instruments=instruments,
        wfc3=args.wfc3)

    # Load models from model file - see example
    params={'theta_obs': args.theta_obs * np.pi/180.0, 'kappa': args.kappa,
        'tfloor': args.tfloor,}
    all_models = []
    all_models += kn.find_models(model_types=args.models, params=params)
    all_models += kn.load_model_file(args.model_file)

    nmodels = str(len(all_models)).zfill(4)
    for num, model in enumerate(all_models):

        kn.model_params.update(model)

        param_str = ''
        for key in sorted(list(kn.model_params.keys())):
            if model['type']=='grb' and key not in ['E_kiso','theta_obs','n']:
                continue
            if is_number(kn.model_params[key]):
                val = '%.4e'%float(kn.model_params[key])
            else:
                val = str(kn.model_params[key])
            param_str += str(key)+'='+val+' '

        model_num = str(int(num)+1).zfill(4)

        # Get model name and output file name
        model_name = kn.get_model_name(kn.model_params)
        outfile_name = os.path.join(kn.options['dirs']['tables'], 
            kn.model_params['type'], model_name + '.dat')
        if not os.path.exists(os.path.dirname(outfile_name)):
            os.makedirs(os.path.dirname(outfile_name))

        if os.path.exists(outfile_name) and not args.clobber:
            print(f'Model {model_num}/{nmodels}: {outfile_name} exists...')
            continue
        else:
            print(f'Model {model_num}/{nmodels}: {param_str}')

        kn.load_model(kn.model_params)

        meta = ['{key} = {value}'.format(key=key, value=kn.model_params[key])
            for key in kn.model_params.keys()]

        # Each model will have a phottable organized as phase (rows) vs.
        # mag in passband (cols).   Here we'll initialize the phottable as
        # an empty table with time (observer frame) and all of the passbands.
        kn.phottables[model_name] = kn.empty_table(kn.bandpass_names)
        kn.phottables[model_name].meta['comment'] = meta

        # Iterate over times in the model
        for i, phase in enumerate(kn.times):
            # Calculate the observer frame phase
            time = float('%.6f'%phase)

            # Generate an empty row of photometry starting with current time
            row = [time] + [0.] * len(kn.bandpass_names)
            kn.phottables[model_name].add_row(row)

            # Get a reference for adding the photometry to table later
            row_num = kn.phottables[model_name]['time'] == time

            for bp in kn.bandpass_names:
                # get synthetic mag for this passband
                passband = kn.bandpass[bp]
                mag = float('%2.3f'%kn.get_mag(passband, phase, kn.distance))

                # Now add to table
                kn.phottables[model_name][bp][row_num] = mag

        # Now output phottable
        formats = {'time': '{:<16}'}
        for bp in kn.bandpass_names:
            formats[bp] = '{:<24}'

        kn.phottables[model_name].write(outfile_name, overwrite=True,
            format='ascii.ecsv', formats=formats)

        if args.armin:
            newtable = copy.copy(kn.phottables[model_name])
            for key in newtable.keys():
                if '_' in key:
                    newkey = key.split('_')[-1]
                    newtable.rename_column(key, newkey)

            outdir, armin_file = os.path.split(outfile_name)
            armin_dir = os.path.join(outdir, 'armin')
            if not os.path.exists(armin_dir): os.makedirs(armin_dir)
            armin_file = os.path.join(armin_dir, 'armin_'+armin_file)

            newtable.write(armin_file, overwrite=True, 
                format='ascii.commented_header', formats=formats)

        if args.flat_ascii:
            newtable = copy.copy(kn.phottables[model_name])

            outdir, ascii_file = os.path.split(outfile_name)
            ascii_dir = os.path.join(outdir, 'ascii')
            if not os.path.exists(ascii_dir): os.makedirs(ascii_dir)
            ascii_file = os.path.join(ascii_dir, 'ascii_'+ascii_file)

            newtable.write(ascii_file, overwrite=True, 
                format='ascii.commented_header', formats=formats)

if __name__=='__main__':
    sys.exit(main())
