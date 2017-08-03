
import numpy as np
from numpy.lib.recfunctions import stack_arrays
import sys
import healpy as hp
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import astropy.wcs
import astropy.io.fits as pyfits

from quicksip import *

nside = 2048*2 # Resolution of output maps
nsidesout = None # if you want full sky degraded maps to be written
ratiores = 2**2 # Superresolution/oversampling ratio
mode = 5 # 1: fully sequential, 2: parallel then sequential, 3: fully parallel, 4: in healpix pixels
# For 4, the calculations are done in patches of sky based on a low-res healpix grid. 
# The argument of this script is the pixel index, which should be between 0 and 12*nside_low**2
# The output maps will only be written if there are images/ccds in that sky patch!
# The outputs can then be stitched / added directly into a full-sky healpix map.
# For 5, the frac dec mask is additionally calculated from the raw images.
ipixel_low = int(sys.argv[1])
nside_low = 4

# Directory where the images are. For file hierarchy, see code below. 
local_dir = '/Users/bl/Downloads'

pixoffset = 0 # Unused in mode 5
undersample = 1 # Unused in mode 5

# If you want to run through all images and compute where (in which low-res pixels)
# all of them fall. This should be run once at a given nside_low, to see 
# how many images fall in each patch, and how many patches exist.
count_images_in_all_pixels = False  

# For the patch under consideration, download all images? Used in mode 5.
download_images = True
# If true, put username:password for desdm data in second argument of the script! 

num_processes = 4 # for multiprocessing for the two options above. Otherwise unused. 

# Run the main script to compute the maps.
create_maps = True

catalogue_name = 'y3a2_quicksip_info'
fname = '/Users/bl/Dropbox/repos/QuickSip/y3a2_quicksip_info_00000'
numfiles = 4

# Where to write the maps ?
outroot = '/Users/bl/Dropbox/repos/QuickSip/test/'

def read_table(fname):
    temp = pyfits.open(fname)[1].data
    propertiesToKeep = ['TILENAME', 'BAND', 'AIRMASS', 'SKYBRITE', 'SKYSIGMA', 'EXPTIME', 'FWHM'] \
        + ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'NAXIS1', 'NAXIS2']  \
        + ['PV1_'+str(i) for i in range(11)] + ['PV2_'+str(i) for i in range(11)] \
        + ['basepath', 'path', 'filename'] + ['RAC'+str(i+1) for i in range(4)] + ['DECC'+str(i+1) for i in range(4)]
    return np.core.records.fromarrays([temp[prop] for prop in propertiesToKeep], names = propertiesToKeep)

tbdata = stack_arrays([ read_table(fname+str(1)+'.fits') for i in range(numfiles)])
print('Loaded', numfiles, 'files. Total number of images:', len(tbdata))

# Select the five bands
ind = tbdata['BAND'] == 'g'
indg = np.where(ind)
ind = tbdata['BAND'] == 'r'
indr = np.where(ind)
sample_names = ['band_g']
inds = [indg]

# What properties do you want mapped?
propertiesandoperations = [
    ('count', '', 'fracdet'),  ('count', '', 'total'),  ('FWHM', '', 'mean'),
    ]

if count_images_in_all_pixels:

    import multiprocessing

    def map_images_to_lowres_pixels(propertyArray):
        pixoffset = 0
        img_ras_c, img_decs_c = computeCorners_WCS_TPV(propertyArray, pixoffset)
        img_phis_c = img_ras_c * np.pi/180
        img_thetas_c =  np.pi/2  - img_decs_c * np.pi/180
        pixels = hp.ang2pix(nside_low, img_thetas_c, img_phis_c, nest=True)
        return np.unique(pixels)

    for sample_name, ind in zip(sample_names, inds):
        with Pool(num_processes) as POOL:
            uniques, counts = np.unique(np.concatenate(POOL.map(map_images_to_lowres_pixels, tbdata[ind])), return_counts=True)
            print('For sample', sample_name)
            print('The unique low res pixels and counts are', [(a, b) for a, b in zip(uniques, counts)])
            print('This is', uniques.size, 'lowres pixels with maximum', np.max(counts), 'images.')

if download_images:

    import urllib3

    basic_auth = sys.argv[2]

    http = urllib3.PoolManager()
    urllib3.disable_warnings()
    headers = urllib3.util.make_headers(basic_auth=basic_auth)

    def download_image(propertyArray):
        img_ras_c, img_decs_c = computeCorners_WCS_TPV(propertyArray, pixoffset)
        img_phis_c = img_ras_c * np.pi/180
        img_thetas_c =  np.pi/2  - img_decs_c * np.pi/180
        img_pix = hp.ang2pix(nside_low, img_thetas_c, img_phis_c, nest=True)
        if ipixel_low in img_pix:
            fname_remote = propertyArray['basepath'].strip() + '/' + propertyArray['path'].strip() + '/' + propertyArray['filename'].strip()
            fname_local = local_dir + '/' + propertyArray['path'].strip() + '/' + propertyArray['filename'].strip()

            if not os.path.isdir(local_dir + '/' + propertyArray['path'].strip()):
                os.makedirs(local_dir + '/' + propertyArray['path'].strip())

            if not os.path.exists(fname_local):
                print("Downloading", propertyArray['path'].strip() + '/' + propertyArray['filename'].strip(), 'into', fname_local)
                request = http.request('GET', fname_remote, preload_content=False, headers=headers)
                data = request.read()
                with open(fname_local, 'wb') as fd:
                    fd.write(data)
                request.release_conn()
            if not os.path.exists(fname_local):
                stop

    for sample_name, ind in zip(sample_names, inds):
        with ThreadPool(num_processes) as POOL:
            POOL.map(download_image, tbdata[ind])

if create_maps:
    # Do the magic! Read the table, create Healtree, project it into healpix maps, and write these maps.
    project_and_write_maps(mode, propertiesandoperations, tbdata, catalogue_name, outroot, sample_names, inds, nside, ratiores, pixoffset, 
        ipixel_low=ipixel_low, nside_low=nside_low, nsidesout=nsidesout, undersample=undersample, local_dir=local_dir)

# ------------------------------------------------------
