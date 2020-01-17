#!/usr/bin/env python

## based on des_run.py, plus extras

import numpy as np
import healpy as hp
import fitsio
import glob
import esutil
import matplotlib.pyplot as plt

import quicksip

nside = 4096
nsidesout = None
ratiores = 4
mode = 1  # fully sequential

skyvar_max = 10000.0

testmode = False

catalog_name = 'Y3A2_MOF'
if testmode: catalog_name = 'Y3A2_MOF_TEST'
pixoffset = 15
#fnames = glob.glob('y3a1_y1-3_ccdinfo_coadd_quicksip_??????.fits')
fnames = glob.glob('y3a2_quicksip_info.fits')
fnames = glob.glob('y3a2_quicksip_info_000001.fits')
fnames.sort()
outroot = 'maps'+str(pixoffset)+'pix/'

#zpfile = '/nfs/slac/g/ki/ki19/des/erykoff/des/y3a1/calibration_testing/upload/y3a1_fgcm_all_v2_5_1.fits'

#zps=fitsio.read(zpfile,ext=1)
#zpHash=zps['EXPNUM']*100+zps['CCDNUM']

sipData = None

for fname in fnames:
    tempData = fitsio.read(fname,ext=1,trim_strings=True)
    if sipData is None:
        sipData = tempData
    else:
        sipData = np.append(sipData,tempData)

tempData=None

# add additional columns...
dtype = sipData.dtype.descr
dtype.extend([('URAUL','f8'),
              ('URALL','f8'),
              ('URAUR','f8'),
              ('URALR','f8'),
              ('UDECUL','f8'),
              ('UDECLL','f8'),
              ('UDECUR','f8'),
              ('UDECLR','f8'),
              ('MAGZP','f8'),
              ('COADD_MAGZP','f8')])

sipData2 = np.zeros(sipData.size,dtype=dtype)
for name in sipData.dtype.names:
    sipData2[name][:] = sipData[name][:]

sipData = None

# set COADD_MAGZP
sipData2['COADD_MAGZP'][:] = 30.0

# get mag zps

sipHash = sipData2['EXPNUM']*100+sipData2['CCDNUM']

#a,b=esutil.numpy_util.match(zpHash,sipHash)

#sipData2['MAGZP'][b] = zps['FGCM_ZPT'][a]

# unsure what to do about these...where do they get zps?
bad,=np.where(sipData2['MAGZP'] == 0.0)
print "Print number of exposures without zeropoints: ",bad.size

skyvarAvg = (sipData2['SKYVARA'][:] + sipData2['SKYVARB'][:])/2.

#use,=np.where((skyvarAvg < skyvar_max) & (sipData2['MAGZP'] > 0.0))
use,=np.where((skyvarAvg < skyvar_max))

sipData2 = sipData2[use]

# and get the four corners...
sipData2['URALL'][:] = sipData2['URAMIN'][:]
sipData2['URALR'][:] = sipData2['URAMAX'][:]
sipData2['URAUL'][:] = sipData2['URAMIN'][:]
sipData2['URAUR'][:] = sipData2['URAMAX'][:]

sipData2['UDECLL'][:] = sipData2['UDECMIN'][:]
sipData2['UDECLR'][:] = sipData2['UDECMIN'][:]
sipData2['UDECUL'][:] = sipData2['UDECMAX'][:]
sipData2['UDECUR'][:] = sipData2['UDECMAX'][:]


### testmode
if testmode:
    use,=np.where((sipData2['URAMIN'] > 47.5) &
                  (sipData2['URAMAX'] < 51.0) &
                  (sipData2['UDECMIN'] > -26.0) &
                  (sipData2['UDECMAX'] < -22.0))

    sipData2=sipData2[use]


# Select the five bands
indg = np.where(sipData2['BAND'] == 'g')
indr = np.where(sipData2['BAND'] == 'r')
indi = np.where(sipData2['BAND'] == 'i')
indz = np.where(sipData2['BAND'] == 'z')
#sample_names = ['band_g', 'band_r', 'band_i', 'band_z']
#inds = [indg, indr, indi, indz]

# Just run on g-band for debugging
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

quicksip.project_and_write_maps(mode, propertiesandoperations, sipData2, catalog_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)
