
import numpy as np
import sys
import healpy as hp
import pyfits
from multiprocessing import Pool

from quicksip import *

nside = 1024 # Resolution of output maps
nsidesout = None # if you want full sky degraded maps to be written
ratiores = 1 # Superresolution/oversampling ratio
mode = 4 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel, 4: in healpix pixels
# For 4, the calculations are done in patches of sky based on a low-res healpix grid. 
# The argument of this script is the pixel index, which should be between 0 and 12*nside_low**2
# The output maps will only be written if there are images/ccds in that sky patch!
# The outputs can then be stitched / added directly into a full-sky healpix map.
ipixel_low = int(sys.argv[1])
nside_low = 1

catalogue_name = 'Y1A1NEW_COADD_SPT_test'
pixoffset = 15
fname = '/Users/bl/Dropbox/Projects/Quicksip/data/Y1A1_COADD_SPT_ASTROM_PSF_INFO'

tbdata = pyfits.open(fname+'.fits')[1].data
#tbdata = np.concatenate([pyfits.open(fname+str(i)+'.fits')[1].data for i in [2]])

# ------------------------------------------------------
# Read data

# Make sure we have the four corners of all coadd images
URALL = tbdata['URALL']
UDECLL = tbdata['UDECLL']
URAUR = tbdata['URAUR']
UDECUR = tbdata['UDECUR']
COADD_RA = (URALL+URAUR)/2
COADD_DEC = (UDECLL+UDECUR)/2
m1 = np.divide(tbdata['RAUR']-tbdata['RALL'],tbdata['DECUR']-tbdata['DECLL'])
m2 = (tbdata['RAUL']-tbdata['RALR'])/(tbdata['DECUL']-tbdata['DECLR'])
ang = - np.arctan(np.abs((m1-m2)/(1+m1*m2)))
vecLL = [ (URALL - COADD_RA), (UDECLL - COADD_DEC) ]
vecUR = [ (URAUR - COADD_RA), (UDECUR - COADD_DEC) ]
URAUL = COADD_RA + np.cos(ang) * (URALL - COADD_RA) - np.sin(ang) * (UDECLL - COADD_DEC)
UDECUL = COADD_DEC + np.sin(ang) * (URALL - COADD_RA) + np.cos(ang) * (UDECLL - COADD_DEC)
URALR = COADD_RA + np.cos(ang) * (URAUR - COADD_RA) - np.sin(ang) * (UDECUR - COADD_DEC)
UDECLR = COADD_DEC + np.sin(ang) * (URAUR - COADD_RA) + np.cos(ang) * (UDECUR - COADD_DEC)

# Where to write the maps ?
outroot = '/Users/bl/Dropbox/repos/QuickSip/test/'

# Select the five bands
ind = tbdata['BAND'] == 'g'
indg = np.where(ind)
ind = tbdata['BAND'] == 'r'
indr = np.where(ind)
sample_names = ['band_g', 'band_r']
inds = [indg, indr]

# What properties do you want mapped?
propertiesandoperations = [
    ('count', '', 'fracdet'), ('maglimit3', '', ''),  ('FWHM', '', 'mean'),
    ]

# What properties to keep when reading the images? Should at least contain propertiesandoperations and the image corners.
# Add 'WEIGHTA', 'WEIGHTB',  if necessary!
propertiesToKeep = ['COADD_ID', 'ID', 'TILENAME', 'BAND', 'AIRMASS', 'SKYBRITE', 'SKYSIGMA', 'EXPTIME'] \
    + ['FWHM', 'FWHM_MEAN', 'FWHM_PIXELFREE_MEAN', 'FWHM_FROMFLUXRADIUS_MEAN', 'NSTARS_ACCEPTED_MEAN'] \
    + ['RA', 'DEC', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'NAXIS1', 'NAXIS2']  \
    + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
    + ['RALL', 'RAUL', 'RAUR', 'RALR', 'DECLL', 'DECUL', 'DECUR', 'DECLR', 'URALL', 'UDECLL', 'URAUR', 'UDECUR', 'COADD_RA', 'COADD_DEC'] \
    + ['COADD_MAGZP', 'MAGZP']

# Create big table with all relevant properties. We will send it to the Quicksip library, which will do its magic.
tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [URAUL, UDECUL, URALR, UDECLR], names = propertiesToKeep + ['URAUL', 'UDECUL', 'URALR', 'UDECLR'])

# Fix ra/phi values beyond 360 degrees
for nm in ['RALL', 'RAUL', 'RAUR', 'RALR', 'URALL', 'URAUR', 'COADD_RA']:
    tbdata[nm] = np.mod(tbdata[nm], 360.0)

# Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, 
    ipixel_low=ipixel_low, nside_low=nside_low, nsidesout=nsidesout)

# ------------------------------------------------------
