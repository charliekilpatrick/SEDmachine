import json
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from matplotlib import rc
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
import os
import sys
import pickle
import json

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
print(parent_dir)

import sedmachine
from common import utilities
from common import constants

machine = sedmachine.sedmachine()

figsize=10.0
rc('font',**{'family':'serif','serif':['Times'], 'size':2.2*figsize})
rc('text', usetex=True, color='k')

def compute_model_mag(bandpasses, times, parameters):

    machine.times = times

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

def load_data(data_file, filts=['K','H','J','y','z','i','r','V','g','B']):

    json_file = open(data_file)
    json_data = json.load(json_file)
    merger_time = float(json_data["GW170817"]["timeofmerger"][0]["value"])
    json_phot = json_data["GW170817"]["photometry"]

    all_data = {}
    for p in json_phot:
        if (('band' in p) and 
            ('u_time' in p) and 
            ('upperlimit' not in p) and 
            ('system' in p) and 
            ('e_magnitude' in p) and 
            (p['u_time'] == "MJD") and 
            ('model' not in p) and 
            (p['system'] == 'AB')):
            filt = p['band']
            if filt in filts:
                if filt not in all_data.keys():
                    all_data[filt]={'time':[], 'mag':[], 'magerr':[]}

                all_data[filt]['time'].append(float(p['time']) - merger_time)
                all_data[filt]['mag'].append(float(p['magnitude']))
                all_data[filt]['magerr'].append(float(p['e_magnitude']))

    for key in all_data.keys():
        sorted_indices = np.argsort(all_data[key]['time'])

        all_data[key]['time']=np.array(all_data[key]['time'])[sorted_indices]
        all_data[key]['mag']=np.array(all_data[key]['mag'])[sorted_indices]
        all_data[key]['magerr']=np.array(all_data[key]['magerr'])[sorted_indices]

    return(all_data)


data_file = os.path.join(parent_dir,"data","lightcurves","GW170817","GW170817.json")
model_table_file1 = os.path.join(parent_dir,"output","tables","mosfit_kn",
    "mosfit_kn_0.023000_0.2560_0.5000.ecsv")
model_table_file2 = os.path.join(parent_dir,"output","tables","mosfit_kn",
    "mosfit_kn_0.050000_0.1490_3.6500.ecsv")

filts=['K','H','J','y','z','i','r','V','g','B']

filt_map = {'K':'ukirt_K',
            'H':'ukirt_H',
            'J':'ukirt_J',
            'y':'PS1_y',
            'z':'sdss_z',
            'i':'sdss_i',
            'r':'sdss_r',
            'V':'johnson_V',
            'g':'sdss_g',
            'B':'johnson_B'}

model_table1 = Table.read(model_table_file1, format='ascii.ecsv')
model_table2 = Table.read(model_table_file2, format='ascii.ecsv')
all_data = load_data(data_file, filts=filts)

model_table = model_table1.copy()
for key in model_table1.keys():
    if key=='time': continue

    flux1 = 10**(-0.4 * (model_table1[key]))
    flux2 = 10**(-0.4 * (model_table2[key]))

    mag = -2.5 * np.log10(flux1 + flux2)

    model_table[key] = mag

# convert to app map
mu = 5*np.log10(40.0 * 1e6)-5

fig, ax = plt.subplots(2,1,figsize=(12, 10), dpi=600,
    gridspec_kw={'height_ratios': [3, 1]})
for i in ax[0].spines.keys(): ax[0].spines[i].set_linewidth(0.3*figsize)

ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())

ax[0].tick_params(direction='in', length=2*figsize,
            width=0.3*figsize, which='major', axis='both', colors='k',
            pad=figsize, top=True, bottom=True, left=True, right=True)
ax[0].tick_params(direction='in', length=figsize,
            width=0.3*figsize, which='minor', axis='both', colors='k',
            pad=0.2*figsize, top=True, bottom=True, left=True, right=True)

for i in ax[1].spines.keys(): ax[1].spines[i].set_linewidth(0.3*figsize)

ax[1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].yaxis.set_minor_locator(AutoMinorLocator())

ax[1].tick_params(direction='in', length=2*figsize,
            width=0.3*figsize, which='major', axis='both', colors='k',
            pad=figsize, top=True, bottom=True, left=True, right=True)
ax[1].tick_params(direction='in', length=figsize,
            width=0.3*figsize, which='minor', axis='both', colors='k',
            pad=0.2*figsize, top=True, bottom=True, left=True, right=True)


cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0,1,len(filts)))
offsets=3.0*(np.linspace(0,1,len(filts))-0.5)

# # Name, A_lambda
extinction = {
'u': 0.5126587253999999,
'g': 0.3994601958,
'r': 0.276344701,
'i': 0.2053537428,
'z': 0.1527454518,
'y': 0.134,
'B': 0.43852336359999994,
'V': 0.3316136412,
'R': 0.2623158234,
'I': 0.18201259299999997,
'J': 0.0857454674,
'H': 0.054301431399999996,
'K': 0.0365234572,
'Clear': 0.110054126,
'cyan': 0.35151045967199995,
'orange': 0.25633734464759994,
'w': 0.28311726260000003,
}

# Offset, Vega->AB mag
offsets_magsys = {
'u': 0.0,
'g': 0.0,
'r': 0.0,
'i': 0.0,
'z': 0.0,
'y': 0.0,
'B': 0.0,
'V': 0.0,
'R': 0.0,
'I': 0.0,
'J': 4.56-3.65,
'H': 4.71-3.32,
'K': 5.14-3.29,
'Clear': 0.0,
'cyan': 0.0,
'orange': 0.0,
'w': 0.0,
}

all_mags = []
all_residuals =[]
all_sigmas=[]
for i,filt in enumerate(filts):

    if filt not in all_data.keys():
        print(f'No data {filt}')

    model_filt = filt_map[filt]

    time = all_data[filt]['time']
    mag = all_data[filt]['mag']
    magerr = all_data[filt]['magerr']

    model_time = model_table['time']
    model_mags = model_table[model_filt]

    offsets[i]=float('%.2f'%offsets[i])
    if offsets[i]==0.0:
        label=filt
    elif offsets[i] < 0:
        label=filt+str(offsets[i])
    else:
        label=filt+'+'+str(offsets[i])

    ax[0].errorbar(time, mag - mu + offsets[i] - extinction[filt], yerr=magerr, 
        color=colors[len(filts)-1-i], fmt='o', linestyle='None', label=label, 
        markeredgecolor='k', markeredgewidth=1, markersize=1.0*figsize)
    
    ax[0].plot(model_time, model_mags+offsets[i]+offsets_magsys[filt],
        color=colors[len(filts)-1-i], linestyle='solid')

    for j in np.arange(len(time)):

        idx = np.argmin(np.abs(model_time-time[j]))
        residual = (mag[j]-mu-extinction[filt]-offsets_magsys[filt])-model_mags[idx]
        all_mags.append(mag[j] - mu + offsets[i] - extinction[filt] - magerr[j])
        all_mags.append(mag[j] - mu + offsets[i] - extinction[filt] + magerr[j])
        all_residuals.append(residual)
        all_sigmas.append(magerr[j])

        ax[1].errorbar(time[j], residual, yerr=magerr[j], 
            color=colors[len(filts)-1-i], marker='o',
            markeredgecolor='k', markeredgewidth=1, markersize=1.0*figsize)

xlim=[0.4,20]
resid_ylim=[-2,2]
yran=resid_ylim[1]-resid_ylim[0]

mag_range = np.max(all_mags)-np.min(all_mags)

ax[0].set_ylim([np.max(all_mags)+0.05*mag_range,np.min(all_mags)-0.05*mag_range])
ax[0].set_xlim(xlim)
ax[1].set_xlim(xlim)
ax[0].set_ylabel('Absolute Magnitude [AB mag]',fontsize=2.0*figsize)
ax[1].set_xlabel('Rest-frame Days from Merger',fontsize=2.0*figsize)

ax[1].set_ylim(resid_ylim)
ax[1].set_xlim(xlim)
ax[1].hlines(0,*xlim,linestyle='dashed',color='k')
# Get RMS from residuals
rms = np.sqrt(np.mean(np.array(all_residuals)**2))
chi2 = 1/len(all_residuals) * np.sum(np.array(all_residuals)**2 / np.array(all_sigmas)**2)

rms = float('%.3f'%rms)
chi2 = float('%.3f'%chi2)

ax[1].text(xlim[0]+0.07,resid_ylim[0]+0.68*yran,r'$\chi^{2}$='+str(chi2)+'\nRMS scatter='+str(rms)+' mag',
    fontsize=1.6*figsize)
ax[1].set_ylabel('Residual [mag]',fontsize=2.0*figsize)

ax[0].legend(fontsize=1.5*figsize,ncols=2)

plt.tight_layout()

plt.savefig(os.path.join(parent_dir, 'plots', 'rprocess_model.png'))
