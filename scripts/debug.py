import sys, os, astropy, h5py, bisect, warnings, scipy, glob, re
import grb
import numpy as np
import astropy.constants
import astropy.units as u
from astropy.table import Table
import scipy.integrate
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt

import pysynphot as S
S.setref(area = 25.0 * 10000)
from sedmachine.sedmachine import *

def main():
    # Make an object with distance
    kn = sedmachine(distance=1.0e-5)

    # load models
    all_models = []
    for i in np.arange(10):
        all_models.extend(kn.find_models(model_type='grb170817a',
            params={
    'E': 0.15869069395227384,
    'Eta0': 7.973477192135503,
    'GammaB': 11.000923300022666,
    'dL': 0.012188,
    'epsb': 0.013323706571267526,
    'epse': 0.04072783842837688,
    'n': 0.0009871221028954489,
    'p': 2.1333493591554804,
    'theta_obs':i*5.0*np.pi/180.0,
    'xiN': 1.0,
    'z': 0.00973
    }))

    kn.waves = np.array([3000,4000,5000,6000,7000,8000,9000])
    bandpass_names = ['3000','4000','5000','6000','7000','8000','9000']

    for num, model in enumerate(all_models):

        kn.load_grb(parameters=model[0])
        E_fmt = '%7.4f' % model[0]['E']
        n_fmt = '%7.7f' % model[0]['n']
        theta_fmt = '%7.4f' % model[0]['theta_obs']
        model_name = '{type}_{theta_obs}_{E}_{n}'.format(
                type=model[0]['type'], E=E_fmt.strip(),
                n=n_fmt.strip(), theta_obs=theta_fmt.strip())

        print(model_name)

        kn.phottables[model_name] = kn.empty_table(bandpass_names)

        # Iterate over times in the model
        for i, phase in enumerate(kn.times):
            # Calculate the observer frame phase
            time = phase

            # Generate an empty row of photometry starting with current time
            row = [time] + [0.] * len(bandpass_names)
            kn.phottables[model_name].add_row(row)

            # Get a reference for adding the photometry to table later
            row_num = kn.phottables[model_name]['time'] == time

            kn.grb_param['dL']=0.012188

            for j,w in enumerate(kn.waves):
                # get synthetic mag for this passband

                nu = np.array([2.998e18/w])

                all_flux = kn.grb_generator.GetSpectral(np.array([phase*86400.0]), nu, kn.grb_param)

                # Now add to table
                kn.phottables[model_name][str(w)][row_num] = all_flux[0]

        formats = {'time': '%7.4f'}
        for bp in bandpass_names:
            formats[bp] = '%7.9f'

        outfile_name = kn.options['dirs']['tables'] + '/' + model_name + '.dat'

        kn.phottables[model_name].write(outfile_name, overwrite=True,
            format='ascii.ecsv', formats=formats)

if __name__=='__main__':
    sys.exit(main())
