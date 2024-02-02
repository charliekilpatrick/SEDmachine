import numpy as np
import os
import json
import shutil
import time
import sys
from astropy.table import Table, hstack, Column

"""

Hacky script to generate mosfit models for kilonovae using arbitrary input
ejecta mass, velocity, and opacity.  Parses them into flat astropy.ecsv tables
and outputs as a single file for each model for a large number of UVOIR filters

"""

parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
print(parent_dir)

def format_time(t_elapsed):
    t_elapsed = float(t_elapsed)

    if t_elapsed>3600.0:
        t_elapsed = t_elapsed/3600.0
        t_elapsed = '%.4f'%t_elapsed
        t_elapsed = t_elapsed + ' hours'
    elif t_elapsed>60.0:
        t_elapsed = t_elapsed/60.0
        t_elapsed = '%.4f'%t_elapsed
        t_elapsed = t_elapsed + ' minutes'
    else:
        t_elapsed = '%.4f'%t_elapsed
        t_elapsed = t_elapsed + ' seconds'

    return(t_elapsed)


def clean_mosfit():
    for subdir in ['jupyter','products','models','modules']:
        shutil.rmtree(subdir)

def parse_mosfit():

    outfile = 'products/walkers.json'
    with open(outfile, 'r') as f:
        data = json.load(f)

    phot = data['rprocess']['photometry']

    times = np.array([float(r['time']) for r in phot])
    mags = np.array([float(r['magnitude']) for r in phot])
    filts = np.array([r['band'] for r in phot])
    uniq_times = sorted(np.unique(times))
    uniq_filts = np.unique(filts)

    ncols = len(uniq_filts)+1
    colnames = ['time']+list(uniq_filts)
    table = Table([[99.0]*len(uniq_times)]*ncols,
        names=colnames)

    for i,u_time in enumerate(uniq_times):
        mask = times==u_time

        s_mags = mags[mask]
        s_filts = filts[mask]

        table['time'][i] = u_time

        # Data are in Vega mag for JHK - update this to AB mag
        for f,m in zip(s_filts, s_mags):
            if f=="J":
                m=m+0.91
            if f=="H":
                m=m+1.39
            if f=="K":
                m=m+1.85
            table[f][i] = m

    return(table)

def run_mosfit(mejecta, vejecta, kappa, temperature=1000.0, Npoints=500):

    mstr = '%.6f'%mejecta
    vstr = '%.4f'%vejecta
    kstr = '%.4f'%kappa
    tstr = '%.4f'%temperature

    filename = f'mosfit_kn_{mstr}_{vstr}_{kstr}.dat'
    outdir = os.path.join(parent_dir, 'output','tables',f'mosfit_kn_{kstr}')
    fulloutfile = os.path.join(outdir, filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if os.path.exists(fulloutfile):
        return(1)

    vejecta = vejecta * 2.998e5
    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-instruments PS1 --band-list w g r i z y 2> /dev/null > /dev/null'
    os.system(cmd)

    ps1_table = parse_mosfit()
    for key in ps1_table.keys():
        if key=='time': continue
        ps1_table.rename_column(key, 'PS1_'+key)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-instruments UVOT --band-list U B V UVW1 UVM2 UVW2 2> /dev/null > /dev/null'
    os.system(cmd)

    uvot_table = parse_mosfit()
    for key in uvot_table.keys():
        if key=='time': continue
        if key in ['UVW1','UVM2','UVW2']:
            newkey = key.lower()
        else:
            newkey = key
        uvot_table.rename_column(key, 'swift_uvot_'+newkey)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-list U B V R I 2> /dev/null > /dev/null'
    os.system(cmd)

    johnson_table = parse_mosfit()
    for key in johnson_table.keys():
        johnson_table.rename_column(key, 'johnson_'+key)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-list u g r i z 2> /dev/null > /dev/null'
    os.system(cmd)

    sdss_table = parse_mosfit()
    for key in sdss_table.keys():
        if key=='time': continue
        sdss_table.rename_column(key, 'sdss_'+key)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-instruments ATLAS --band-list o c 2> /dev/null > /dev/null'
    os.system(cmd)

    atlas_table = parse_mosfit()
    for key in atlas_table.keys():
        if key=='time': continue
        atlas_table.rename_column(key, 'ATLAS_'+key)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-list J H K 2> /dev/null > /dev/null'
    os.system(cmd)

    ukirt_table = parse_mosfit()
    for key in ukirt_table.keys():
        if key=='time': continue
        ukirt_table.rename_column(key, 'ukirt_'+key)

    cmd = f'mosfit -m rprocess -i 0 -F kappagamma 100.0 mejecta {mejecta} texplosion 0 frad 1.0 kappa {kappa} avhost 0.0 nhhost 0.0 vejecta {vejecta} lumdist 0.00001 redshift 0.0 variance 0.0 ebv 0.0 temperature {temperature} -N 1 -S {Npoints} --max-time 30 --band-list open 2> /dev/null > /dev/null'
    os.system(cmd)

    clear_table = parse_mosfit()
    clear_table.rename_column('open','Clear')

    outtable = hstack([ps1_table,uvot_table,johnson_table,sdss_table,atlas_table,ukirt_table,
        clear_table])

    while True:
        found_bad = False
        for key in outtable.keys():
            if 'time' in key and key!='time':
                found_bad = True
                if 'time' in outtable.keys():
                    outtable.remove_column(key)
                else:
                    outtable.rename_column(key, 'time')

        if not found_bad: break

    vejecta=vejecta/(2.998e5)
    comment = [f'kappa = {kappa}',
               'type = rprocess',
               f'mass = {mejecta}',
               f'velocity = {vejecta}',
               'theta_obs = 0.0',
               f'tfloor = {temperature}']
    outtable.meta['comment']=comment

    outtable.write(fulloutfile, format='ascii.ecsv', overwrite=True)

    clean_mosfit()

    return(0)

if __name__=="__main__":
    N_M = 31
    N_v = 31
    # In solar mass in units of log_Msun
    MAX_M = -3.0
    MIN_M = -0.30102999566

    # Velocity in units of c
    MAX_v = 0.03
    MIN_v = 0.50

    kappa_vals = [0.5, 1.0, 3.0, 3.65, 10.0]

    temperature = 1000.0

    num_models = len(kappa_vals) * N_M * N_v

    t0 = time.time()

    durations = []
    i=1
    for kappa in [0.5, 1.0, 3.0, 3.65, 10.0]:
        for m_idx in np.arange(0, N_M):
            mass = 10**(MIN_M + (MAX_M - MIN_M) * float(m_idx)/(N_M-1))
            for v_idx in np.arange(0, N_v):
                velocity = MIN_v + (MAX_v - MIN_v) * float(v_idx)/(N_v-1)
                t_model_start = time.time()
                outcode = run_mosfit(mass, velocity, kappa, 
                    temperature=temperature)
                t_model_end = time.time()

                t_model_duration = t_model_end - t_model_start
                t_elapsed = t_model_end - t0
                if outcode==0: durations.append(t_model_duration)

                # Get average duration per model minus number of remaining models
                num_model_remain = num_models - i
                if len(durations)>0:
                    avg_duration = np.mean(durations)
                else:
                    avg_duration = 0.001
                t_predict = avg_duration * num_model_remain

                t_duratin = format_time(t_model_duration)
                t_elapsed = format_time(t_elapsed)
                t_predict = format_time(t_predict)

                mstr = '%.6f'%mass
                vstr = '%.4f'%velocity
                kstr = '%.4f'%kappa
                tstr = '%.4f'%temperature

                desc_str = f'mass={mstr}, velocity={vstr}, kappa={kstr}, temperature={tstr}'

                print(f'Model {i}/{num_models} ({desc_str}) took {t_duratin}, '+\
                    f'total elapsed {t_elapsed}, '+\
                    f'total remaining {t_predict}')

                i=i+1
