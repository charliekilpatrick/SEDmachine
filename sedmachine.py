#!/usr/bin/env python3

"""
By C. D. Kilpatrick 2019-05-08
"""

# Python 2/3 compatibility
from __future__ import print_function

import sys,os,astropy,h5py,bisect,warnings,scipy
import numpy as np
import astropy.constants
import astropy.units as u
from astropy.table import Table
warnings.filterwarnings('ignore')

import pysynphot as S
S.setref(area = 25.0 * 10000)

A_TO_CM = u.angstrom.to(u.centimeter)
A_TO_MU = u.angstrom.to(u.micron)
MPC_TO_CM = u.megaparsec.to(u.centimeter)
C_CGS = astropy.constants.c.cgs.value

# Set up IDL and options
#idl = pidyl.IDL(os.environ['IDL_DIR']+'/bin/idl')
#idl.pro('set_plot','ps')

# Global plotting options for IDL
#idl_opt = {
#    'font': '!6',
#    'thick': 4,
#    'xthick': 4,
#    'ythick': 4,
#    'xstyle': True,
#    'ystyle': True
#}

class sedmachine(object):
    def __init__(self):

        # Initialize times, frequency, luminosity
        self.times   = None
        self.nu      = None
        self.Lnu_all = None

        # wave set
        self.waves = np.array(10.0 + 0.5 * np.arange(60000))

        # Bandpass
        self.bandpass = {}
        self.bandpass_names = []

        # Photometry tables for different phases, object types, models, etc.
        self.phottables = {}

        # Generic options that might need to be changed
        self.options = {
            'magsystem': 'abmag'
            'dirs': {
                'filter': 'Filters',
                'figures': 'Figures',
                'tables': 'Tables',
                'models': 'Models'
            }
        }

        # These are the default filters that we want to model
        self.filters = {
                'hst': {
                    'wfc3': {
                        'uvis': ['f200lp','f300x','f350lp','f475x','f600lp',
                                 'f850lp','f218w','f225w','f275w','f336w',
                                 'f390w','f438w','f475w','f555w','f606w',
                                 'f625w','f775w','f814w'],
                        'ir': ['f105w','f110w','f125w','f140w','f160w']
                    },
                    'acs': {
                        'wfc': ['f555w','f775w','f625w','f550m','f850lp',
                                'f606w','f475w','f814w','f435w','f330w','f250w',
                                'f220w'],
                        'sbc': ['f115lp','f125lp','f140lp','f150lp','f165lp',
                                'f122m']
                    },
                    'wfpc2': []
                },
                'swift': {
                    'uvot': ['uvw2','uvm2','uvw1','B','V','W']
                },
                'spitzer': {
                    'irac': []
                },
                'sdss': ['u','g','r','i','z','y'],
                'johnson': ['U','B','V','R','I'],
                'ukirt': ['Y', 'J', 'H', 'K']
        }

    def load_kasen_model(self, model, distance=40.):
        """
        Read in Kasen et al. 2017 spectral models, transform to wavelength
        in angstrom and flam (cgs) for input distance, and save in arrays
        """

        # Read Kasen et al. 2017 spectral model and return the base arrays
        name = self.options['dirs']['model'] + '/' + model
        fin  = h5py.File(name,'r')

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

    def load_blackbody(self, temperature, luminosity, lumunit=u.cgs):
        """
        Load a blackbody SED into self.wave and self.flux for a given
        temperature and luminosity.  Uses pysynphot built in blackbody function.
        """



    def load_passbands(self, passbands={}):
        """
        Load passbands either from pysynphot or check for an equivalent file in
        the default filter directory.
        """

        # Load default filters first
        filters = self.filters
        message = 'Loading {source} bandpasses...'
        print(message.format(source='wfc3,uvis'))
        for bp in filters['hst']['wfc3']['uvis']:
            name = 'wfc3,uvis1,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='wfc3,ir'))
        for bp in filters['hst']['wfc3']['ir']:
            name = 'wfc3,ir,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='acs,wfc'))
        for bp in filters['hst']['acs']['wfc']:
            name = 'acs,wfc,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='acs,sbc'))
        for bp in filters['hst']['acs']['sbc']:
            name = 'acs,sbc,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='swift,uvot'))
        for bp in filters['swift']['uvot']:
            file = self.options['dirs']['filters']+'/SWIFT.'+bp.upper()+'.dat'
            name = 'swift,uvot,'+bp
            wave,transmission = np.loadtxt(file, unpack=True)
            bpmodel = S.ArrayBandpass(wave, transmission,
                name=name, waveunits='Angstrom')
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='johnson'))
        for bp in johnson_bandpasses:
            name = 'johnson,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='sdss'))
        for bp in sdss_bandpasses:
            name = 'sdss,'+bp
            bpmodel = S.ObsBandpass(name)
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)

        # Now if there are other passbands, load them individually
        # Must be formatted as dict with {name: filename}
        for bp in passbands.keys():
            print(message.format(source=bp))
            file = passbands[bp]
            wave,transmission = np.loadtxt(file, unpack=True)
            bpmodel = S.ArrayBandpass(wave, transmission,
                name=bp, waveunits='Angstrom')
            self.bandpass[bp] = bpmodel
            self.bandpass_names.append(bp)

    def get_model(self, phase):
        """
        Get the flam spectrum for some specific phase
        """
        it  = bisect.bisect(self._times, phase)
        it -= 1 # Kasen array indexing is off by 1
        Lnu = self._Lnu_all[it,:]

        const = self.options['constants']

        # We want things in Flambda (ergs/s/Angstrom)
        lam  = const['c'] / self._nu * const['ang->cm']
        Llam = Lnu * self._nu**2.0 / const['c'] / const['ang->cm']
        return(lam, Llam)

    def get_norm_model(self, phase, distance):
        """
        Get the flam spectrum for some specific phase and distance
        """
        const = self.options['constants']

        dist = c.Distance(distance*u.megaparsec)
        z = dist.z
        lam, flam = self.get_model(phase)
        lamz = lam * (1. + z)
        fnorm = flam / (4 * np.pi * (distance * MPC_TO_CM)**2.)
        return(lamz, fnorm)

    def get_mag(self, passband, phase, distance):
        """
        Get magnitude for input passband at phase and distance
        """
        lam, flam   = self.get_model(phase)
        lum = -scipy.integrate.simps(flam,lam)
        lamz, fnorm = self.get_norm_model(phase, dmpc)
        spec = S.ArraySpectrum(wave=lamz, flux=fnorm,
            waveunits='angstrom', fluxunits='flam')

        obs  = S.Observation(spec, passband,
            force='taper', binset=self.waves)

        try:
            mag = obs.effstim[self.options['magsystem']]
        except ValueError as e:
            mag = np.nan
        return(mag)

    def empty_table(self, passband_names):
        """
        Make an empty phottable with time and all passbands so we can add to it
        row by row.
        """
        names = ['time'] + passband_names
        value = [0.] + [0.] * len(passband_names)
        table = Table([value], names=names)[:0].copy()
        return(table)

def main():
    # Make a kilonova object
    kn = sedmachine()

    # Load generic passbands
    kn.load_passbands()

    # load the kilonova models
    for model in models:
        kn = kn.load_kilonova_model(model)

        # Each model will have a phottable organized as phase (rows) vs.
        # mag in passband (cols).   Here we'll initialize the phottable as
        # an empty table with time (observer frame) and all of the passbands.
        kn.phottables[model] = kn.empty_table(kn.passband_names)

        # Iterate over times in the model
        for i, phase in enumerate(kn.times):
            # Calculate the observer frame phase
            z = kn.distance.z
            time = phase * (1.+z)

            # Generate an empty row of photometry starting with current time
            row = [time] + [0.] * len(kn.passband_names)
            kn.phottables[model].add_row(row)

            # Get a reference for adding the photometry to table later
            row_num = kn.phottables[model]['time'] == time

            for bp in kn.passband_names:
                # get synthetic mag for this passband
                passband = kn.bandpass[bp]
                mag = kn.get_mag(passband, phase, distance)

                # Now add to table
                kn.phottable[model][bp][row_num] = mag

if __name__=='__main__':
    sys.exit(main())
