from scipy.stats import chi2
import numpy as np
import json

def get_chi2_limit(ndim, significance=0.68):
    found = False
    maxval = 10

    while not found:
        maxval = maxval*10
        x = np.linspace(0, maxval, 100000)
        chi_dist = chi2.pdf(x, ndim)

        integral = [0.0]
        total = 0.0
        for i,t in enumerate(x):
            if i==0: continue
            dt = t-x[i-1]
            total += chi_dist[i]*dt
            integral.append(total)

        integral = np.array(integral)
        if all(integral < significance): 
            continue
        else:
            idx = np.argmin(np.abs(integral-significance))
            return(x[idx])

# Load mosfit json data files
def load_data(data_file, filts=['K','H','J','y','z','i','r','V','g',
    'B','U','u','UVW1','UVM2','UVW2']):

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
            ('model' not in p) and
            ('e_magnitude' in p or ('e_lower_magnitude' in p and 'e_upper_magnitude' in p)) and 
            (p['u_time'] == "MJD") and 
            (p['system'] == 'AB')):
            filt = p['band']
            if filt in filts:
                if 'e_magnitude' in p:
                    magerr = float(p['e_magnitude'])
                else:
                    magerr = np.max([float(p['e_upper_magnitude']),float(p['e_lower_magnitude'])])

                if 'telescope' in p:
                    if p['telescope'].lower()=='swift':
                        use_filt = 'swift_uvot_'+str(filt)
                    elif p['telescope'].lower()=='swope':
                        use_filt = 'swope_'+str(filt)
                    elif filt in ['u','g','r','i','z']:
                        use_filt = 'sdss_'+str(filt)
                    elif filt in ['J','H','K']:
                        use_filt = 'ukirt_'+str(filt)
                    elif filt in ['U','B','V','R','I']:
                        use_filt = 'johnson_'+str(filt)
                    elif filt in ['y']:
                        use_filt = 'PS1_y'
                    else:
                        raise Exception(f'ERROR: unknown filt {filt}')

                if use_filt not in all_data.keys():
                    all_data[use_filt]={'time':[], 'mag':[], 'magerr':[]}

                all_data[use_filt]['time'].append(float(p['time']) - merger_time)
                all_data[use_filt]['mag'].append(float(p['magnitude']))
                all_data[use_filt]['magerr'].append(float(magerr))

    for key in all_data.keys():
        sorted_indices = np.argsort(all_data[key]['time'])

        all_data[key]['time']=np.array(all_data[key]['time'])[sorted_indices]
        all_data[key]['mag']=np.array(all_data[key]['mag'])[sorted_indices]
        all_data[key]['magerr']=np.array(all_data[key]['magerr'])[sorted_indices]

    return(all_data)
