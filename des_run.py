
import numpy as np
import healpy as hp
import pyfits
from multiprocessing import Pool

from quicksip import *

# ------------------------------------------------------
# DES_RUN
# ------------------------------------------------------
# Main runs for SVA1 and Y1A1, from the IMAGE tables.
# ------------------------------------------------------

nside = 4096 # Resolution of output maps
nsidesout = None # if you want full sky degraded maps to be written
ratiores = 4 # Superresolution/oversampling ratio
mode = 1 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel

catalogue_name = 'Y1A1NEW_COADD_DFULL'
pixoffset = 15 # How many pixels are being removed on the edge of each CCD? 15 for DES.
fname = 'data/Y1A1_COADD_DFULL_ASTROM_PSF_INFO.fits'
# Where to write the maps ? Make sure directory exists.
outroot = 'maps'+str(pixoffset)+'pix/'

tbdata = pyfits.open(fname)[1].data
print tbdata.dtype

# ------------------------------------------------------
# Read data

# Make sure we have the four corners of all coadd images.
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

# Select the five bands
indg = np.where(tbdata['BAND'] == 'g')
indr = np.where(tbdata['BAND'] == 'r')
indi = np.where(tbdata['BAND'] == 'i')
indz = np.where(tbdata['BAND'] == 'z')
indY = np.where(tbdata['BAND'] == 'Y')
sample_names = ['band_g', 'band_r', 'band_i', 'band_z', 'band_Y']
inds = [indg, indr, indi, indz, indY]

# What properties do you want mapped?
# Each each tuple, the first element is the quantity to be projected,
# the second is the weighting scheme, and the third is the operation.
propertiesandoperations = [
    ('count', '', 'fracdet'),  # CCD count with fractional values in pixels partially observed.
    ('EXPTIME', '', 'total'), # Total exposure time, constant weighting.
    ('maglimit3', '', ''), # Magnitude limit (3rd method, correct)
    ('FWHM_MEAN', '', 'mean'), # Mean FWHM (called FWHM_MEAN because it's already the mean per CCD)
    ('FWHM_MEAN', 'coaddweights3', 'mean'), # Same with with weighting corresponding to CDD noise
    ('FWHM_PIXELFREE_MEAN', '', 'mean'),
    ('FWHM_PIXELFREE_MEAN', 'coaddweights3', 'mean'),
    ('FWHM_FROMFLUXRADIUS_MEAN', '', 'mean'),
    ('FWHM_FROMFLUXRADIUS_MEAN', 'coaddweights3', 'mean'),
    ('NSTARS_ACCEPTED_MEAN', '', 'mean'),
    ('NSTARS_ACCEPTED_MEAN', 'coaddweights3', 'mean'),
    ('FWHM', '', 'mean'),
    ('FWHM', 'coaddweights3', 'mean'),
    ('AIRMASS', '', 'mean'),
    ('AIRMASS', 'coaddweights3', 'mean'),
    ('SKYBRITE', '', 'mean'),
    ('SKYBRITE', 'coaddweights3', 'mean'),
    ('SKYSIGMA', '', 'mean'),
    ('SKYSIGMA', 'coaddweights3', 'mean')
    ]

# What properties to keep when reading the images? Should at least contain propertiesandoperations and the image corners.
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
project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, nsidesout)

# ------------------------------------------------------
