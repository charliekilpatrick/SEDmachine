# Dependencies and settings
import glob, sys, os, shutil, time, io, urllib, subprocess, contextlib, warnings, filecmp
from astropy.io import fits
from stwcs import updatewcs
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astroquery.mast import Observations
from astropy import utils
from drizzlepac import tweakreg,astrodrizzle
from contextlib import contextmanager
from astroscrappy import detect_cosmics
import numpy as np

# Suppresses warnings
warnings.filterwarnings('ignore')

base_url = 'http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id='
detectors = ['WFPC2','WFC3_UVIS1','WFC3_UVIS2','WFC3_IR','ACS_HRC','ACS_SBC','ACS_WFC']

acceptable_filters = {'F220W','F250W','F330W','F344N','F435W','F475W','F502N',
                      'F550M','F555W','F606W','F625W','F658N','F660N','F660N',
                      'F775W','F814W','F850LP','F892N','F098M','F105W','F110W',
                      'F125W','F126N','F127M','F128N','F130N','F132N','F139M',
                      'F140W','F153M','F160W','F164N','F167N','F200LP','F218W',
                      'F225W','F275W','F280N','F300X','F336W','F343N','F350LP',
                      'F373N','F390M','F390W','F395N','F410M','F438W','F467M',
                      'F469N','F475X','F487N','F502N','F547M','F600LP','F621M',
                      'F625W','F631N','F645N','F656N','F657N','F658N','F665N',
                      'F673N','F680N','F689M','F763M','F845M','F953N','F122M',
                      'F160BW','F185W','F218W','F255W','F300W','F375N','F380W',
                      'F390N','F437N','F439W','F450W','F569W','F588N','F622W',
                      'F631N','F673N','F675W','F702W','F785LP','F791W','F953N',
                      'F1042M'}

for det in detectors:
    for filt in acceptable_filters:

        name_lower = 'HST/'+det+'.'+filt.lower()
        name_upper = 'HST/'+det+'.'+filt.upper()

        out_name = ''
        if ('_' in det):
            out_det = det.split('_')[0]+'.'+det.split('_')[1]
            out_name = out_det+'.'+filt.upper()+'.dat'
        else:
            out_det = det+'.PC'
            out_name = out_det+'.'+filt.upper()+'.dat'

        print 'Out name is:',out_name,name_upper

        dat = ''
        url = base_url + name_lower
        dat = utils.data.download_file(url,cache=False,show_progress=False,timeout=120)
        if (os.stat(dat).st_size == 0):
          os.remove(dat)
        else:
          shutil.move(dat,out_name)
          continue

        url = base_url + name_upper
        dat = utils.data.download_file(url,cache=False,show_progress=False,timeout=120)
        if (os.stat(dat).st_size == 0):
          os.remove(dat)
        else:
          shutil.move(dat,out_name)
          continue
