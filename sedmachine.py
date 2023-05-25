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

# Internal dependency
from common import grb

class sedmachine(object):
    def __init__(self, distance=1.0e-5):

        # Initialize times, frequency, luminosity
        self.times   = None
        self.nu      = None
        self.Lnu_all = None
        self.meta    = {}

        self.distance = distance

        # wave set
        n_wave = 80
        self.waves = np.array(500.0+0.5*n_wave*np.arange(int(80000/n_wave)))

        # Bandpass
        self.bandpass = {}
        self.bandpass_names = []

        self.grb_param = {}

        # Photometry tables for different phases, object types, models, etc.
        self.phottables = {}

        self.pandeia = '/Users/ckilpatrick/scripts/stsci/pandeia/'

        # Generic options that might need to be changed
        self.options = {
            'magsystem': 'abmag',
            'time':'log',
            'dirs': {
                'filter': 'data/filters',
                'models': 'data/models',
                'figures': 'output/figures',
                'tables': 'output/tables',
            }
        }

        # For loading GRB models
        grb_model = self.options['dirs']['models']+'/gw170817.h5'
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
                'swift': {
                    'uvot': ['uvw2','uvm2','uvw1','B','V','W']
                },
                'spitzer': {
                    'irac': []
                },
                'sdss': ['u','g','r','i','z'],
                'johnson': ['U','B','V','R','I'],
                'kpno': ['J', 'H', 'K']
        }

    def add_options(self):

        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('model_file',type=str,
            help='Input model file for sedmachine.py')
        parser.add_argument('--filters',type=str,default=None,
            help='Override default to output all filters with input value.')
        parser.add_argument('--armin',default=False,action='store_true',
            help='Armin likes the output in this specific format.  '+\
            'Flag to have it output this way.')

        args = parser.parse_args()

        return(args)

    def load_kasen_model(self, model):
        """
        Read in Kasen et al. 2017 spectral models, transform to wavelength
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
        # Default parameters/GW170817
        dL = 0.012188
        P = {
            'E': 0.15869069395227384,
            'Eta0': 7.973477192135503,
            'GammaB': 11.000923300022666,
            'dL': 0.012188,
            'epsb': 0.013323706571267526,
            'epse': 0.04072783842837688,
            'n': 0.0009871221028954489,
            'p': 2.1333493591554804,
            'theta_obs':0.0,
            'xiN': 1.0,
            'z': 0.00973
        }

        # Update parameters if there are any provided
        if parameters:
            for key in P.keys():
                if key in parameters.keys():
                    P[key] = parameters[key]

        self.grb_param=P

        # Generate a spectrum for the default waveset and input phase
        Ntimes = 500
        MAX_TIME = 1500.0
        MIN_TIME = 0.01

        if self.options['time']=='linear':
            self.times = (MAX_TIME - MIN_TIME)/Ntimes * np.arange(0, Ntimes+1)+\
                MIN_TIME
        elif self.options['time']=='log':
            self.times   = 10**((np.log10(MAX_TIME)-np.log10(MIN_TIME))/Ntimes*\
                np.arange(0, Ntimes + 1) + np.log10(MIN_TIME))

        self.nu      = C_CGS / (self.waves * A_TO_CM)
        self.Lnu_all = np.zeros((len(self.times), len(self.nu)))

        # Construct dummy arrays to hold these data in 1D
        all_nu = np.array(list(self.nu) * len(self.times))
        all_times = np.array(sum([[t * DAY_TO_S] * len(self.nu)
            for t in self.times], []))
        all_flux = self.grb_generator.GetSpectral(all_times, all_nu, P)

        # Sanitize flux array for nan (set to zero) and rescale to luminosity
        all_flux = np.array(all_flux)

        all_flux[np.isnan(all_flux)] = 0.0
        all_flux = all_flux * 1.0e-26 * 4 * np.pi * (dL * 1.0e28)**2

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
        
    def load_passbands(self, passbands=None):
        """
        Load passbands either from pysynphot or check for an equivalent file in
        the default filter directory.
        """

        # Load default filters first
        filters = self.filters
        message = 'Loading {source} bandpasses...'
        print(message.format(source='wfc3,uvis'))
        for bp in filters['hst']['wfc3,uvis1']:
            name = 'wfc3,uvis1,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='wfc3,ir'))
        for bp in filters['hst']['wfc3,ir']:
            name = 'wfc3,ir,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='acs,wfc'))
        for bp in filters['hst']['acs']['wfc']:
            name = 'acs,wfc1,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='acs,sbc'))
        for bp in filters['hst']['acs']['sbc']:
            name = 'acs,sbc,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='jwst,nircam'))
        for bp in filters['jwst']['nircam']:
            name = 'jwst,nircam,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = self.get_jwst_filters(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='jwst,miri'))
        for bp in filters['jwst']['miri']:
            name = 'jwst,miri,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = self.get_jwst_filters(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='swift,uvot'))
        for bp in filters['swift']['uvot']:
            file = self.options['dirs']['filter']+'/SWIFT.'+bp.upper()+'.dat'
            name = 'swift,uvot,'+bp
            name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            wave,transmission = np.loadtxt(file, unpack=True)
            bpmodel = S.ArrayBandpass(wave, transmission,
                name=name, waveunits='Angstrom')
            self.bandpass[name] = bpmodel
            self.bandpass_names.append(name)
        print(message.format(source='johnson'))
        for bp in filters['johnson']:
            name = 'johnson,'+bp
            add_name = name.replace(',','_')
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        print(message.format(source='sdss'))
        for bp in filters['sdss']:
            name = 'sdss,'+bp
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            add_name = name.replace(',','_')
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)
        for bp in filters['kpno']:
            name = 'kpno,'+bp
            if passbands:
                if name not in passbands.keys():
                    continue
            bpmodel = S.ObsBandpass(name)
            add_name = name.replace(',','_')
            self.bandpass[add_name] = bpmodel
            self.bandpass_names.append(add_name)

        print(message.format(source='kpno'))

        # Now if there are other passbands, load them individually
        # Must be formatted as dict with {name: filename}
        for bp in passbands.keys():
            if not passbands[bp]:
                continue
            print(message.format(source=bp))
            file = self.options['dirs']['filter'] + '/' + passbands[bp]
            wave,transmission = np.loadtxt(file, unpack=True)
            bpmodel = S.ArrayBandpass(wave, transmission,
                name=bp, waveunits='Angstrom')
            self.bandpass[bp] = bpmodel
            self.bandpass_names.append(bp.replace(',','_'))

    def find_models(self, model_type='', params={}):
        """
        Get a list of models and metadata from models/ directory
        """

        model_list = []
        if 'kilonova' in model_type:
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
                    data.update(params)
                    model_list.append([data, model])

        elif 'grb170817a' in model_type:
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
            model_list=[[data]]

        elif model_type=='grb':

            N_E = 30
            N_n = 30

            # In units of log10(E/1e50 erg)
            MAX_E = 2.0
            MIN_E = -2.0

            # In units of log10(n * cm3)
            MAX_n = 0.0
            MIN_n = -6.0

            for i in np.arange(0, N_E+1):
                for j in np.arange(0, N_n+1):

                    data = {}
                    data['type'] = 'grb'
                    data['E'] = 10**(MIN_E + (MAX_E - MIN_E) * float(i)/N_E)
                    data['n'] = 10**(MIN_n + (MAX_n - MIN_n) * float(j)/N_n)
                    data['theta_obs'] = 0.0#17.0 * np.pi/180.0
                    data.update(params)

                    model_list.append([data])

        elif 'grb_angle' in model_type:

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

                model_list.append([data])


        return(model_list)

    def get_model(self, phase):
        """
        Get the flam spectrum for some specific phase
        """
        it  = bisect.bisect(self.times, phase)
        it -= 1 # Kasen array indexing is off by 1
        Lnu = self.Lnu_all[it,:]

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

        table = Table.read(model_file, format='ascii.commented_header')

        all_models = []
        for row in table:

            model = {}
            model['type']=row['type']
            model['name']=row['name']

            if model['type']=='Kasen':
                model['model']=row['models']
            elif model['type']=='Kasen_double':
                m1, m2 = row['models'].split(',')
                model['model1']=m1
                model['model2']=m2

            all_models.append(model)

        return(all_models)


def main():
    # Make an object with distance
    kn = sedmachine(distance=1.0e-5)

    kn.load_passbands(passbands)

    # load models
    args = kn.add_options()

    # Load filters
    passbands = args.filters
    kn.load_passbands(passbands)

    # Load models from model file - see example
    all_models = kn.load_model_file(args.model_file)

    for num, model in enumerate(all_models):

        param_str = ' '.join([str(key)+'='+str(model[key])
            for key in model.keys()])
        print('Working on model',num,param_str)

        if model['type']=='Kasen':
            kn.load_kasen_model(model['model'])
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = model['model'].replace('h5','')

        elif model['type']=='grb':
            kn.load_grb(parameters=model[0])
            E_fmt = '%7.4f' % model[0]['E']
            n_fmt = '%7.7f' % model[0]['n']
            theta_fmt = '%7.4f' % model[0]['theta_obs']
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = '{type}_{theta_obs}_{E}_{n}'.format(
                    type=model[0]['type'], E=E_fmt.strip(),
                    n=n_fmt.strip(), theta_obs=theta_fmt.strip())

        elif model['type']=='Kasen_double':
            kn.load_double_kasen_model(model['model1'], model['model2'])
            if 'name' in model.keys():
                model_name = model['name']
            else:
                model_name = 'combined_Kasen_model'

        meta = ['{key} = {value}'.format(key=key, value=model[key])
            for key in model.keys()]

        # Each model will have a phottable organized as phase (rows) vs.
        # mag in passband (cols).   Here we'll initialize the phottable as
        # an empty table with time (observer frame) and all of the passbands.
        kn.phottables[model_name] = kn.empty_table(kn.bandpass_names)
        kn.phottables[model_name].meta['comment'] = meta

        # Iterate over times in the model
        for i, phase in enumerate(kn.times):
            # Calculate the observer frame phase
            time = float('%.4f'%phase)

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
        outfile_name = os.path.join(kn.options['dirs']['tables'], model['type'],
            model_name + '.dat')

        if not os.path.exists(os.path.dirname(outfile_name)):
            os.makedirs(os.path.dirname(outfile_name))

        formats = {'time': '{:<12}'}
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
            armin_file = os.path.join(armin_dir, 'armin_'+armin_file)
            newtable.write(armin_file, overwrite=True, 
                format='ascii.commented_header', formats=formats)


if __name__=='__main__':
    sys.exit(main())
