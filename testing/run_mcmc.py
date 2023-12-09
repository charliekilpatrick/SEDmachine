import json
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from matplotlib import rc
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
import os
import sys
import emcee
import json
import pysynphot as S
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import sedmachine
from common import utilities
from common import constants

machine = sedmachine.sedmachine()
machine.filters = {
    'swope': ['u','B','V','g','r','i'],
    'swift': ['M2','W1','W2','U','B','V'],
    'sdss': ['u','g','r','i','z'],
    'johnson': ['U','B','V','R','I'],
    'ukirt': ['J','H','K'],
    'ps1': ['y'],
}


data_file = os.path.join(parent_dir,"data","lightcurves","GW170817","GW170817.json")
filts=['K','H','J','y','z','i','r','V','g','B','U','u','W1','M2','W2']
all_data = utilities.load_data(data_file, filts=filts)
machine.filters = {
                'swope': ['u','B','V','g','r','i'],
                'swift': ['M2','W1','W2','U','B','V'],
                'sdss': ['u','g','r','i','z'],
                'johnson': ['U','B','V','R','I'],
                'ukirt': ['J','H','K'],
                'ps1': ['y'],
                }

sed_filts = [f for f in all_data.keys()]
machine.load_passbands()
bandpass_data = machine.bandpass

blackbody_file = os.path.join(parent_dir, "data","models","blackbodies.pkl")
if os.path.exists(blackbody_file):
    with open(blackbody_file, 'rb') as f:
        all_blackbodies = pickle.load(f)
else:
    temperatures = np.logspace(1, 6, num=1000)
    blackbody_data = {}
    for key in bandpass_data.keys():
        bp = bandpass_data[key]
        print(key)

        bb_mags = []
        for t in temperatures:

            bb = machine.create_blackbody(1.0, t)
            scale = 1.0 /(4 * np.pi * (1.0e-5 * 3.086e+24)**2)
            bb = bb * scale
            
            obs = S.Observation(bb, bp)

            try:
                bb_mags.append(obs.effstim('abmag'))
            except ValueError:
                bb_mags.append(99.999)

        blackbody_data[key]=bb_mags

    all_blackbodies = {}
    for key in blackbody_data.keys():
        f = interp1d(temperatures, blackbody_data[key],
            bounds_error=False, fill_value='extrapolate')
        all_blackbodies[key]=f

    with open(blackbody_file, 'wb') as f:
        pickle.dump(all_blackbodies, f)

# convert to app map
mu = 5*np.log10(40. * 1e6)-5
ebv = 0.105
rv = 3.1
av = ebv * rv

# Get all of the bandpasses, times, magnitudes, and magnitude errors
bandpasses = []
times = []
mags = []
magerrs = []

print(bandpass_data.keys())
for key in all_data.keys():

    filt_name = str(key)
    bandpass = bandpass_data[filt_name]

    time = all_data[key]['time']
    mag = all_data[key]['mag']
    magerr = all_data[key]['magerr']

    a_lambda = machine.calculate_extinction(av, rv, bandpass)
    print(filt_name,a_lambda)

    for i in np.arange(len(time)):

        bandpasses.append(filt_name)
        times.append(time[i])
        mags.append(mag[i]-mu-a_lambda)
        magerrs.append(magerr[i])

bandpasses = np.array(bandpasses)
times = np.array(times)
mags = np.array(mags)
magerrs = np.array(magerrs)

machine.times = times

figsize=10.0
rc('font',**{'family':'serif','serif':['Times'], 'size':2.2*figsize})
rc('text', usetex=True, color='k')

def get_init_pos(guess, ndim, nwalkers, initialize=0.25):

    init_pos = [guess * np.random.uniform(1-initialize,1+initialize,ndim)
            for i in range(nwalkers)]

    return(init_pos)

def sample_params(params, prob, ndim, downsample=1.0):

        mask = np.isinf(np.abs(prob)) | np.isnan(prob)
        if all(mask):
            print('WARNING: all probabilities are bad.  Try wider param range')
            return(params[0])
        if len(params.shape)==1:
            params = params[~mask]
            prob = prob[~mask]
        else:
            params = params[~mask,:]
            prob = prob[~mask]

        #prob = -1.0 * prob
        print('Minimum probability is:',np.min(prob))
        print('Maximum probability is:',np.max(prob))
        prob = prob - np.max(prob) - 1.0
        prob = prob * -1.0
        prob = prob / np.min(prob)

        chi2_limit = utilities.get_chi2_limit(ndim)

        mask = prob < 1.0 + chi2_limit

        params_sample = params[mask]
        prob_sample = prob[mask]

        print('There are',len(params_sample),'samples')

        return(params_sample, prob_sample)

def calculate_param_best_fit(params, prob, ndim, name, show=True,
        sampled=False, return_uncertainty=False):

        n = 4
        out_fmt = '{0:<18}: {1:>12} + {2:>12} - {3:>12}'

        params_sample = params

        # Get 16-50-84 values for param_sample now that we've restricted sample
        # to chi2 bounds
        #best = params_sample[np.argmin(prob)]
        best = np.percentile(params_sample, 50)
        minval = np.percentile(params_sample, 16)
        maxval = np.percentile(params_sample, 84)

        mcmc = best
        log_mcmc = np.log10(mcmc)
        if np.isnan(log_mcmc):
            log_mcmc = 0.0
        digits = int(np.ceil(log_mcmc))
        decimal_place = -1 * (digits - n)
        if float(mcmc)==int(mcmc) and decimal_place < 1:
            mcmc=int(mcmc)

        minval = round(minval, decimal_place)
        maxval = round(maxval, decimal_place)
        maxval = maxval-mcmc
        minval = mcmc-minval

        if float(minval)==int(minval) and decimal_place < 1:
            minval=int(minval)
        if float(maxval)==int(maxval) and decimal_place < 1:
            maxval=int(maxval)

        if np.log10(mcmc)<-3:
            str_fmt = '%.3e'
            mcmc = str_fmt % mcmc
            maxval = str_fmt % maxval
            minval = str_fmt % minval
        else:
            str_fmt = '%7.4f'.format(int(decimal_place))
            mcmc = str_fmt % mcmc
            maxval = str_fmt % maxval
            minval = str_fmt % minval

        if show: print(out_fmt.format(name,mcmc, maxval, minval))

        if return_uncertainty:
            return(float(maxval), float(minval))
        else:
            return(best)

def log_likelihood(theta, bandpasses, times, mag, magerr):

    # Set limits on parameters
    if any([v < 0 for v in theta]):
        return(-np.inf)
    mej1, vej1, tfloor1, mej2, vej2, tfloor2, mej3, vej3, tfloor3, sigma = theta
    if mej1 > 0.3 or mej2 > 0.3 or mej3 > 0.3:
        return(-np.inf)

    if vej1 > 1.0 or vej2 > 1.0 or vej3 > 1.0:
        return(-np.inf)

    if tfloor1 > 5000.0 or tfloor2 > 5000.0 or tfloor3 > 5000.0:
        return(-np.inf)

    try:
        model_mags =  compute_model_mag(bandpasses, times, theta)
    except ValueError:
        return(-np.inf)

    mask = np.isnan(model_mags)
    if any(mask):
        return(-np.inf)

    chi2 = -0.5 * np.sum((mag-model_mags)**2/(magerr**2 + sigma**2))
    chi2 -= np.sum(np.log(2 * np.pi * magerr**2))
    chi2 -= len(magerr)/2.0 * np.log(2 * np.pi * sigma**2)

    return(chi2)

def compute_model_mag(bandpasses, times, theta):

    mej1, vej1, tfloor1, mej2, vej2, tfloor2, mej3, vej3, tfloor3, sigma = theta
    parameters = {
            'mass1': mej1,
            'kappa1': 0.5,
            'velocity1': vej1,
            'tfloor1': tfloor1,
            'mass2': mej2,
            'kappa2': 3.0,
            'velocity2': vej2,
            'tfloor2': tfloor2,
            'mass3': mej3,
            'kappa3': 10.0,
            'velocity3': vej3,
            'tfloor3': tfloor3,
    }

    lums=[]
    temps=[]
    for i in np.arange(3):
        i=i+1
        P={}
        for param in ['mass','kappa','velocity','tfloor']:
            P[param]=parameters[param+str(i)]

        lum, temp = machine.generate_villar_model(parameters=P)

        lums.append(lum)
        temps.append(temp)

    lums=np.array(lums)
    temps=np.array(temps)

    mags=[]
    for j in np.arange(len(times)):
        blackbody_fluxes = np.array([10**(-0.4*all_blackbodies[bandpasses[j]](temps[:,j]))])

        mags.append(-2.5*np.log10(np.sum(lums[:,j]/(3.826e33) * blackbody_fluxes)))
    
    mags=np.array(mags)

    return(mags)

def plot_chains(sample, param):

    fig, ax = plt.subplots()
    idx = np.arange(len(sample))
    ax.plot(idx, sample)

    plt.savefig(os.path.join(parent_dir, 'plots', param+'.png'))
    plt.clf()

nwalkers = 100
nsteps = 100

backfile = os.path.join(parent_dir,'data','backends','backfile_new5.pkl')
backend = emcee.backends.HDFBackend(backfile, name='run00')

params=['mass1','velocity1','tfloor1','mass2','velocity2','tfloor2','mass3',
        'velocity3','tfloor3','sigma']
ndim=len(params)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood,
            args=(bandpasses, times, mags, magerrs), backend=backend)

try:
    sample = np.array(backend.get_chain(flat=True))
    prob = np.array(backend.get_log_prob(flat=True))
except AttributeError:
    sample = []
    prob = []

if len(sample)>0:
    guess = sample[-1,:]
else:
    model = machine.find_models(model_types=['villar_gw170817_3comp_mod'])[0]
    guess = [model[p] for p in params if p in model.keys()]
    if len(guess)==9:
        guess += [0.242]

init_pos = get_init_pos(guess, ndim, nwalkers)

sampler.run_mcmc(init_pos, nsteps, progress=True)

sample = np.array(backend.get_chain(flat=True))
prob = np.array(backend.get_log_prob(flat=True))

burn = int(0.5 * len(prob))
sample = sample[burn:,:]
prob = prob[burn:]

sample, prob = sample_params(sample, prob, ndim)
outdata={'kappa1':0.5,'kappa2':3.0,'kappa3':10.0,
    'type':'villar_3comp','name': 'villar_gw170817_3comp_mcmc'}
for i,param in enumerate(params):
    plot_chains(sample[:,i], param)
    p=calculate_param_best_fit(sample[:,i], prob, ndim, param)
    outdata[param]=p

with open(os.path.join(parent_dir,'data','models',
    'villar_gw170817_3comp_mod.dat'),'w') as f:
    json.dump(outdata, f)

