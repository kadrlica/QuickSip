#!/usr/bin/env python

## based on des_run.py, plus extras

import numpy as np
import healpy as hp
import fitsio
import glob
import esutil
import matplotlib.pyplot as plt

import quicksip

# ADW: Configuration properties (should be a separate config file)
nside = 4096
nsidesout = None
ratiores = 4
mode = 1  # fully sequential

skyvar_max = 10000.0

testmode = True

catalog_name = 'Y3A2_MOF'
if testmode: catalog_name = 'Y3A2_MOF_TEST'
pixoffset = 15
#fnames = glob.glob('y3a1_y1-3_ccdinfo_coadd_quicksip_??????.fits')
#fnames = glob.glob('y3a2_quicksip_small.fits')
#fnames = glob.glob('y3a2_quicksip_medium.fits')
fnames = glob.glob('y3a2_quicksip_all_??????.fits')
fnames.sort()
outroot = 'maps'+str(pixoffset)+'pix'

#zpfile = '/nfs/slac/g/ki/ki19/des/erykoff/des/y3a1/calibration_testing/upload/y3a1_fgcm_all_v2_5_1.fits'

data = None
# Load multiple files (if necessary)
# ADW: This is not the best way to do this...
for fname in fnames:
    print("Loading %s..."%fname)
    tempData = fitsio.read(fname,ext=1,trim_strings=True)
    if data is None:
        data = tempData
    else:
        data = np.append(data,tempData)

# Free memory
tempData=None

# Exposures without zeropoints (probably want to remove?)
bad,=np.where(data['MAGZP'] == 0.0)
print "Number of images without zeropoints: ",bad.size

# Calculate average sky variance
skyvarAvg = (data['SKYVARA'][:] + data['SKYVARB'][:])/2.
use,=np.where((skyvarAvg < skyvar_max))

data = data[use]

### testmode: just use a spatial sub-selection
if testmode:
    use,=np.where((data['CRVAL1'] > 47.5) &
                  (data['CRVAL1'] < 51.0) &
                  (data['CRVAL2'] > -26.0) &
                  (data['CRVAL2'] < -22.0))

    data=data[use]


# Select the five bands
indg = np.where(data['BAND'] == 'g')
indr = np.where(data['BAND'] == 'r')
indi = np.where(data['BAND'] == 'i')
indz = np.where(data['BAND'] == 'z')
#sample_names = ['band_g', 'band_r', 'band_i', 'band_z']
#inds = [indg, indr, indi, indz]

# ADW: Just run on g-band for debugging
sample_names = ['band_g']
inds = [indg]

propertiesandoperations = [
    ('count', '', 'fracdet'),  # CCD count with fractional values in pixels partially observed.
    ('EXPTIME', '', 'total'), # Total exposure time, constant weighting.
    ('maglimit3', '', ''), # Magnitude limit (3rd method, correct)
    ('FWHM', 'coaddweights3', 'mean'),
    ('AIRMASS', 'coaddweights3', 'mean'),
    ('SKYBRITE', 'coaddweights3', 'mean'),
    ('SKYSIGMA', 'coaddweights3', 'mean')
    ]

quicksip.project_and_write_maps(
    mode=mode, propertiesweightsoperations=propertiesandoperations, 
    tbdata=data, catalogue_name=catalog_name, outrootdir=outroot, 
    sample_names=sample_names, inds=inds, nside=nside, ratiores=ratiores, 
    pixoffset=pixoffset, nsidesout=nsidesout
    )
