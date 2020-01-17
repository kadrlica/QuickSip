
import numpy as np
import healpy as hp
from time import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy.random
import os, errno
import subprocess

import astropy.wcs
import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------------------------------------------------------------------------- #

# Make directory
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# Some unit definitions
arcsec_to_radians = 0.0000048481368111
degree_to_arcsec = 3600.0

# ---------------------------------------------------------------------------------------- #

# Write partial Healpix map to file
# indices are the indices of the pixels to be written
# values are the values to be written
def write_partial_map(filename, indices, values, nside, nest=False):
    if nside < 8192:
        fitsformats = [hp.fitsfunc.getformat(np.int32), hp.fitsfunc.getformat(np.float32)]
    else:
        fitsformats = [hp.fitsfunc.getformat(np.int64), hp.fitsfunc.getformat(np.float32)]
    column_names = ['PIXEL', 'SIGNAL']
    # maps must have same length
    assert len(set((len(indices), len(values)))) == 1, "Indices and values must have same length"
    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')
    firstpix = np.min(indices)
    lastpix = np.max(indices)
    npix = np.size(indices)
    cols=[]
    for cn, mm, fm in zip(column_names, [indices, values], fitsformats):
        cols.append(pyfits.Column(name=cn, format='%s' % fm, array=mm))
    if False: # Deprecated : old way to create table with pyfits before v3.3
        tbhdu = pyfits.new_table(cols)
    else:
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header['PIXTYPE'] = ('HEALPIX','HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header['ORDERING'] = (ordering, 'Pixel ordering scheme, either RING or NESTED')
    tbhdu.header['EXTNAME'] = ('xtension', 'name of this binary table extension')
    tbhdu.header['NSIDE'] = (nside,'Resolution parameter of HEALPIX')
    tbhdu.header['FIRSTPIX'] = (firstpix, 'First pixel # (0 based)')
    tbhdu.header['OBS_NPIX'] = npix
    tbhdu.header['GRAIN'] = 1
    tbhdu.header['OBJECT'] = 'PARTIAL'
    tbhdu.header['INDXSCHM'] = ('EXPLICIT', 'Indexing: IMPLICIT or EXPLICIT')
    tbhdu.writeto(filename,clobber=True)
    subprocess.call("gzip "+filename,shell=True)

# ---------------------------------------------------------------------------------------- #

# Find healpix ring number from z
def ring_num(nside, z, shift=0):
    # ring = ring_num(nside, z [, shift=])
    #     returns the ring number in {1, 4*nside-1}
    #     from the z coordinate
    # usually returns the ring closest to the z provided
    # if shift = -1, returns the ring immediatly north (of smaller index) of z
    # if shift = 1, returns the ring immediatly south (of smaller index) of z

    my_shift = shift * 0.5
    # equatorial
    iring = np.round( nside*(2.0 - 1.5*z) + my_shift )
    if (z > 2./3.):
        iring = np.round( nside * np.sqrt(3.0*(1.0-z)) + my_shift )
        if (iring == 0):
            iring = 1
    # south cap
    if (z < -2./3.):
       iring = np.round( nside * np.sqrt(3.0*(1.0+z)) - my_shift )
       if (iring == 0):
           iring = 1
       iring = int(4*nside - iring)
    # return ring number
    return int(iring)

# ---------------------------------------------------------------------------------------- #

# returns the z coordinate of ring ir for Nside
def  ring2z (nside, ir):
    fn = float(nside)
    if (ir < nside): # north cap
       tmp = float(ir)
       z = 1.0 - (tmp * tmp) / (3.0 * fn * fn)
    elif (ir < 3*nside): # tropical band
       z = float( 2*nside-ir ) * 2.0 / (3.0 * fn)
    else:   # polar cap (south)
       tmp = float(4*nside - ir )
       z = - 1.0 + (tmp * tmp) / (3.0 * fn * fn)
    # return z
    return z

# ---------------------------------------------------------------------------------------- #

# gives the list of Healpix pixels contained in [phi_low, phi_hi]
def in_ring(nside, iz, phi_low, phi_hi, conservative=True):
# nir is the number of pixels found
# if no pixel is found, on exit nir =0 and result = -1
    npix = hp.nside2npix(nside)
    ncap  = 2*nside*(nside-1) # number of pixels in the north polar cap
    listir = -1
    nir = 0

    # identifies ring number
    if ((iz >= nside) and (iz <= 3*nside)): # equatorial region
        ir = iz - nside + 1  # in {1, 2*nside + 1}
        ipix1 = ncap + 4*nside*(ir-1) #  lowest pixel number in the ring
        ipix2 = ipix1 + 4*nside - 1   # highest pixel number in the ring
        kshift = ir % 2
        nr = nside*4
    else:
        if (iz < nside): #  north pole
            ir = iz
            ipix1 = 2*ir*(ir-1)        #  lowest pixel number in the ring
            ipix2 = ipix1 + 4*ir - 1   # highest pixel number in the ring
        else:                         #    south pole
            ir = 4*nside - iz
            ipix1 = npix - 2*ir*(ir+1) #  lowest pixel number in the ring
            ipix2 = ipix1 + 4*ir - 1   # highest pixel number in the ring
        nr = int(ir*4)
        kshift = 1

    twopi = 2*np.pi
    shift = kshift * .5
    if conservative:
        # conservative : include every intersected pixels,
        # even if pixel CENTER is not in the range [phi_low, phi_hi]
        ip_low = round (nr * phi_low / twopi - shift)
        ip_hi  = round (nr * phi_hi  / twopi - shift)
        ip_low = ip_low % nr      # in {0,nr-1}
        ip_hi  = ip_hi  % nr      # in {0,nr-1}
    else:
        # strict : include only pixels whose CENTER is in [phi_low, phi_hi]
        ip_low = np.ceil (nr * phi_low / twopi - shift)
        ip_hi  = np.floor(nr * phi_hi  / twopi - shift)
        diff = (ip_low - ip_hi) % nr      # in {-nr+1,nr-1}
        if (diff < 0):
            diff = diff + nr # in {0,nr-1}
        if (ip_low >= nr):
            ip_low = ip_low - nr
        if (ip_hi  < 0 ):
            ip_hi  = ip_hi  + nr

    if phi_low <= 0.0 and phi_hi >= 2.0*np.pi:
        ip_low = 0
        ip_hi = nr - 1
    if (ip_low > ip_hi):
        to_top = True
    else:
        to_top = False
    ip_low = int( ip_low + ipix1 )
    ip_hi  = int( ip_hi  + ipix1 )

    ipix1 = int(ipix1)
    if (to_top):
        nir1 = int( ipix2 - ip_low + 1 )
        nir2 = int( ip_hi - ipix1  + 1 )
        nir  = int( nir1 + nir2 )
        if ((nir1 > 0) and (nir2 > 0)):
            listir   = np.concatenate( (np.arange(ipix1, nir2+ipix1), np.arange(ip_low, nir1+ip_low) ) )
        else:
            if nir1 == 0:
                listir   = np.arange(ipix1, nir2+ipix1)
            if nir2 == 0:
                listir   = np.arange(ip_low, nir1+ip_low)
    else:
        nir = int(ip_hi - ip_low + 1 )
        listir = np.arange(ip_low, nir+ip_low)

    return listir

# ---------------------------------------------------------------------------------------- #

# Linear interpolation
def lininterp(xval, xA, yA, xB, yB):

    slope = (yB-yA) / (xB-xA)
    yval = yA + slope * (xval - xA)
    return yval

# ---------------------------------------------------------------------------------------- #

# Test if val beints to interval [b1, b2]
def inInter(val, b1, b2):
    if b1 <= b2:
        return np.logical_and( val <= b2, val >= b1 )
    else:
        return np.logical_and( val <= b1, val >= b2 )

# ---------------------------------------------------------------------------------------- #

# Test if a list of (theta,phi) values below to a region defined by its corners (theta,phi) for Left, Right, Bottom, Upper
def in_region(thetavals, phivals, thetaU, phiU, thetaR, phiR, thetaL, phiL, thetaB, phiB):

    npts = len(thetavals)
    phis = np.ndarray( (npts, 4) )
    thetas = np.ndarray( (npts, 4) )
    inds_phi = np.ndarray( (npts, 4), dtype=bool )
    inds_phi[:,:] = False
    inds_theta = np.ndarray( (npts, 4), dtype=bool )
    inds_theta[:,:] = False

    if thetaU != thetaB:
        phis[:,0] = lininterp(thetavals, thetaB, phiB, thetaU, phiU)
        inds_phi[:,0] = inInter(thetavals, thetaB, thetaU)
    if thetaL != thetaU:
        phis[:,1] = lininterp(thetavals, thetaU, phiU, thetaL, phiL)
        inds_phi[:,1] = inInter(thetavals, thetaU, thetaL)
        inds_phi[phis[:,0]==phis[:,1],1] = False
    if thetaL != thetaR:
        phis[:,2] = lininterp(thetavals, thetaL, phiL, thetaR, phiR)
        inds_phi[:,2] = inInter(thetavals, thetaL, thetaR)
        inds_phi[phis[:,0]==phis[:,2],2] = False
        inds_phi[phis[:,1]==phis[:,2],2] = False
    if thetaR != thetaB:
        phis[:,3] = lininterp(thetavals, thetaR, phiR, thetaB, phiB)
        inds_phi[:,3] = inInter(thetavals, thetaR, thetaB)
        inds_phi[phis[:,0]==phis[:,3],3] = False
        inds_phi[phis[:,1]==phis[:,3],3] = False
        inds_phi[phis[:,2]==phis[:,3],3] = False

    if phiU != phiB:
        thetas[:,0] = lininterp(phivals, phiB, thetaB, phiU, thetaU)
        inds_theta[:,0] = inInter(phivals, phiB, phiU)
    if phiL != phiU:
        thetas[:,1] = lininterp(phivals, phiU, thetaU, phiL, thetaL)
        inds_theta[:,1] = inInter(phivals, phiU, phiL)
        inds_theta[thetas[:,0]==thetas[:,1],1] = False
    if phiL != phiR:
        thetas[:,2] = lininterp(phivals, phiL, thetaL, phiR, thetaR)
        inds_theta[:,2] = inInter(phivals, phiL, phiR)
        inds_theta[thetas[:,0]==thetas[:,2],2] = False
        inds_theta[thetas[:,1]==thetas[:,2],2] = False
    if phiR != phiB:
        thetas[:,3] = lininterp(phivals, phiR, thetaR, phiB, thetaB)
        inds_theta[:,3] = inInter(phivals, phiR, phiB)
        inds_theta[thetas[:,0]==thetas[:,3],3] = False
        inds_theta[thetas[:,1]==thetas[:,3],3] = False
        inds_theta[thetas[:,2]==thetas[:,3],3] = False

    ind = np.where(np.logical_and(inds_phi[:,:].sum(axis=1)>1, inds_theta[:,:].sum(axis=1)>1))[0]
    res = np.ndarray( (npts, ), dtype=bool )
    res[:] = False

    for i in ind:
        phival = phivals[i]
        thetaval = thetavals[i]
        phis_loc = phis[i,inds_phi[i,:]]
        thetas_loc = thetas[i,inds_theta[i,:]]
        res[i] = (phival >= phis_loc[0]) & (phival <= phis_loc[1]) & (thetaval >= thetas_loc[0]) & (thetaval <= thetas_loc[1])

    return res

# ---------------------------------------------------------------------------------------- #

# Computes healpix pixels of propertyArray.
# pixoffset is the number of pixels to truncate on the edges of each ccd image.
# ratiores is the super-resolution factor, i.e. the edges of each ccd image are processed
#   at resultion 4*nside and then averaged at resolution nside.
def computeHPXpix_sequ_new(nside, propertyArray, pixoffset=0, ratiores=4, coadd_cut=True, ipixel_low=None, nside_low=None):
    img_ras, img_decs = computeCorners_WCS_TPV(propertyArray, pixoffset)

    # Coordinates of coadd corners
    # RALL, t.DECLL, t.RAUL, t.DECUL, t.RAUR, t.DECUR, t.RALR, t.DECLR, t.URALL, t.UDECLL, t.URAUR, t.UDECUR
    if coadd_cut:
        coadd_ras = [propertyArray[v] for v in ['URAUL', 'URALL', 'URALR', 'URAUR']]
        coadd_decs = [propertyArray[v] for v in ['UDECUL', 'UDECLL', 'UDECLR', 'UDECUR']]
        coadd_phis = np.multiply(coadd_ras, np.pi/180)
        coadd_thetas =  np.pi/2  - np.multiply(coadd_decs, np.pi/180)
    else:
        coadd_phis = 0.0
        coadd_thetas = 0.0
    # Coordinates of image corners
    img_phis = img_ras * np.pi/180
    img_thetas =  np.pi/2  - img_decs * np.pi/180

    if ipixel_low is not None and nside_low is not None:
        img_pix = hp.ang2pix(nside_low, img_thetas, img_phis, nest=True)
        if ipixel_low not in img_pix:
            return None

    img_pix = hp.ang2pix(nside, img_thetas, img_phis, nest=False)
    pix_thetas, pix_phis = hp.pix2ang(nside, img_pix, nest=False)
    #img_phis = np.mod( img_phis + np.pi, 2*np.pi ) # Enable these two lines to rotate everything by 180 degrees
    #coadd_phis = np.mod( coadd_phis + np.pi, 2*np.pi ) # Enable these two lines to rotate everything by 180 degrees
    ind_U = 0
    ind_L = 2
    ind_R = 3
    ind_B = 1
    ipix_list = np.zeros(0, dtype=int)
    weight_list = np.zeros(0, dtype=float)
    # loop over rings until reached bottom
    iring_U = ring_num(nside, np.cos(img_thetas.min()), shift=0)
    iring_B = ring_num(nside, np.cos(img_thetas.max()), shift=0)
    ipixs_ring = []
    pmax = np.max(img_phis)
    pmin = np.min(img_phis)
    if (pmax - pmin > np.pi):
        ipixs_ring = np.int64(np.concatenate([in_ring(nside, iring, pmax, pmin, conservative=True) for iring in range(iring_U-1, iring_B+1)]))
    else:
        ipixs_ring = np.int64(np.concatenate([in_ring(nside, iring, pmin, pmax, conservative=True) for iring in range(iring_U-1, iring_B+1)]))

    ipixs_nest = hp.ring2nest(nside, ipixs_ring)
    npixtot = hp.nside2npix(nside)
    if ratiores > 1:
        subipixs_nest = np.concatenate([np.arange(ipix*ratiores**2, ipix*ratiores**2+ratiores**2, dtype=np.int64) for ipix in ipixs_nest])
        nsubpixperpix = ratiores**2
    else:
        subipixs_nest = ipixs_nest
        nsubpixperpix = 1

    rangepix_thetas, rangepix_phis = hp.pix2ang(nside*ratiores, subipixs_nest, nest=True)
    #subipixs_ring = hp.ang2pix(nside*ratiores, rangepix_thetas, rangepix_phis, nest=False).reshape(-1, nsubpixperpix)

    if (pmax - pmin > np.pi) or (np.max(coadd_phis) - np.min(coadd_phis) > np.pi):
        img_phis= np.mod( img_phis + np.pi, 2*np.pi )
        coadd_phis= np.mod( coadd_phis + np.pi, 2*np.pi )
        rangepix_phis = np.mod( rangepix_phis + np.pi, 2*np.pi )

    subweights = in_region(rangepix_thetas, rangepix_phis,
                                   img_thetas[ind_U], img_phis[ind_U], img_thetas[ind_L], img_phis[ind_L],
                                   img_thetas[ind_R], img_phis[ind_R], img_thetas[ind_B], img_phis[ind_B])
    if coadd_cut:
        subweights_coadd = in_region(rangepix_thetas, rangepix_phis,
                                   coadd_thetas[ind_U], coadd_phis[ind_U], coadd_thetas[ind_L], coadd_phis[ind_L],
                                   coadd_thetas[ind_R], coadd_phis[ind_R], coadd_thetas[ind_B], coadd_phis[ind_B])
        resubweights = np.logical_and(subweights, subweights_coadd).reshape(-1, nsubpixperpix)
    else:
        resubweights = subweights.reshape(-1, nsubpixperpix)

    sweights = resubweights.sum(axis=1) / float(nsubpixperpix)
    ind = (sweights > 0.0)
    
    return ipixs_ring[ind], sweights[ind], img_thetas, img_phis, resubweights[ind, :]

# ---------------------------------------------------------------------------------------- #

def computeHPXpix_CCDpixels(nside, propertyArray, pixoffset=0, ratiores=4, ipixel_low=None, nside_low=None, undersample=1):

    img_ras_c, img_decs_c = computeCorners_WCS_TPV(propertyArray, pixoffset)
    img_phis_c = img_ras_c * np.pi/180
    img_thetas_c =  np.pi/2  - img_decs_c * np.pi/180
    if ipixel_low is not None and nside_low is not None:
        img_pix = hp.ang2pix(nside_low, img_thetas_c, img_phis_c, nest=True)
        if ipixel_low not in img_pix:
            return None

    img_ras, img_decs = computeAllPixels_WCS_TPV(propertyArray, pixoffset, undersample)
    numccdpix = img_ras.size

    # Coordinates of image pixels
    img_phis = img_ras * np.pi/180
    img_thetas =  np.pi/2  - img_decs * np.pi/180

    nonunique_ipixs_ring = hp.ang2pix(nside, img_thetas, img_phis, nest=False)
    ccd_subipixs_nest = hp.ang2pix(nside*ratiores, img_thetas, img_phis, nest=True)
    ipixs_ring = np.unique(nonunique_ipixs_ring)

    # compute matrix for which CCD pixels go into which healpix pixel
    #unique_subipixs_nest, inverse, count = np.unique(subipixs_nest, return_inverse=True, return_counts=True)
    #idx_vals = np.where(count > 0)[0]
    # idx_vals_repeated = where(count > 1)[0]
    # vals_repeated = vals[idx_vals_repeated]
    #rows, cols = np.where(inverse == idx_vals[:, np.newaxis])
    #_, inverse_rows = np.unique(rows, return_index=True)
    #idx_nonunique_subipixs_nest = np.split(cols, inverse_rows[1:])

    ind_U = 0
    ind_L = 2
    ind_R = 3
    ind_B = 1
    ipixs_nest = hp.ring2nest(nside, ipixs_ring)
    npixtot = hp.nside2npix(nside)
    if ratiores > 1:
        subipixs_nest = np.concatenate([np.arange(ipix*ratiores**2, ipix*ratiores**2+ratiores**2, dtype=np.int64) for ipix in ipixs_nest])
        nsubpixperpix = ratiores**2
    else:
        subipixs_nest = ipixs_nest
        nsubpixperpix = 1

    ccdmask = subipixs_nest.reshape((ipixs_nest.size, nsubpixperpix, 1)) == ccd_subipixs_nest[None, None, :]

    #ccdmask = np.zeros((ipixs_nest.size, nsubpixperpix, ccd_subipixs_nest.size), dtype=bool)
    #ccdmask[:] = False
    #for i1, ipix_nest in enumerate(ipixs_nest):
    #    for i2, subipix_nest in enumerate(np.arange(ipix_nest*ratiores**2, ipix_nest*ratiores**2+ratiores**2, dtype=np.int64)):
    #        ccdmask[i1, i2, ccd_subipixs_nest == subipix_nest] = True

    rangepix_thetas, rangepix_phis = hp.pix2ang(nside*ratiores, subipixs_nest, nest=True)

    pmax = np.max(img_phis)
    pmin = np.min(img_phis)
    if (pmax - pmin > np.pi):
        img_phis= np.mod( img_phis + np.pi, 2*np.pi )
        rangepix_phis = np.mod( rangepix_phis + np.pi, 2*np.pi )

    # weights of subpixels falling in original rectangle CCD or not. 
    subweights = in_region(rangepix_thetas, rangepix_phis,
                           img_thetas_c[ind_U], img_phis_c[ind_U], img_thetas_c[ind_L], img_phis_c[ind_L],
                           img_thetas_c[ind_R], img_phis_c[ind_R], img_thetas_c[ind_B], img_phis_c[ind_B])
    resubweights = subweights.reshape(-1, nsubpixperpix)

    # normalized resubweights
    sweights = resubweights.sum(axis=1) / float(nsubpixperpix)
    ind = (sweights > 0.0)
    
    return ipixs_ring[ind], sweights[ind], img_thetas, img_phis, resubweights[ind, :], ccdmask

# ---------------------------------------------------------------------------------------- #
def computeAllPixels_WCS_TPV(propertyArray, pixoffset, undersample=1):
    xline = np.arange(1+pixoffset, propertyArray['NAXIS1']-pixoffset+1, undersample)
    yline = np.arange(1+pixoffset, propertyArray['NAXIS2']-pixoffset+1, undersample)
    x, y = np.meshgrid(xline, yline)
    ras, decs = xy2radec(x.ravel(), y.ravel(), propertyArray)
    return ras, decs

# Crucial routine: read properties of a ccd image and returns its corners in ra dec.
# pixoffset is the number of pixels to truncate on the edges of each ccd image.
def computeCorners_WCS_TPV(propertyArray, pixoffset):
    x = [1+pixoffset, propertyArray['NAXIS1']-pixoffset, propertyArray['NAXIS1']-pixoffset, 1+pixoffset, 1+pixoffset]
    y = [1+pixoffset, 1+pixoffset, propertyArray['NAXIS2']-pixoffset, propertyArray['NAXIS2']-pixoffset, 1+pixoffset]
    ras, decs = xy2radec(x, y, propertyArray)
    return ras, decs

# ---------------------------------------------------------------------------------------- #

# Performs WCS inverse projection to obtain ra dec from ccd image information.
def xy2radec(x, y, propertyArray):

    crpix = np.array( [ propertyArray['CRPIX1'], propertyArray['CRPIX2'] ] )
    cd = np.array( [ [ propertyArray['CD1_1'], propertyArray['CD1_2'] ],
                     [ propertyArray['CD2_1'], propertyArray['CD2_2'] ] ] )
    pv1 = [ float(propertyArray['PV1_'+str(k)]) for k in range(11) if k != 3 ] #  if k != 3
    pv2 = [ float(propertyArray['PV2_'+str(k)]) for k in range(11) if k != 3 ] #  if k != 3
    pv = np.array( [ [ [ pv1[0], pv1[2], pv1[5], pv1[9] ],
                                   [ pv1[1], pv1[4], pv1[8],   0.   ],
                                   [ pv1[3], pv1[7],   0.  ,   0.   ],
                                   [ pv1[6],   0.  ,   0.  ,   0.   ] ],
                                 [ [ pv2[0], pv2[1], pv2[3], pv2[6] ],
                                   [ pv2[2], pv2[4], pv2[7],   0.   ],
                                   [ pv2[5], pv2[8],   0.  ,   0.   ],
                                   [ pv2[9],   0.  ,   0.  ,   0.   ] ] ] )

    center_ra = propertyArray['CRVAL1'] * np.pi / 180.0
    center_dec = propertyArray['CRVAL2'] * np.pi / 180.0
    ras, decs = radec_gnom(x, y, center_ra, center_dec, cd, crpix, pv)
    ras = np.multiply( ras, 180.0 / np.pi )
    decs = np.multiply( decs, 180.0 / np.pi )
    if np.any(ras > 360.0):
        ras[ras > 360.0] -= 360.0
    if np.any(ras < 0.0):
        ras[ras < 0.0] += 360.0
    return ras, decs

# ---------------------------------------------------------------------------------------- #

# Deproject into ra dec values
def deproject_gnom(u, v, center_ra, center_dec):
    u *= arcsec_to_radians
    v *= arcsec_to_radians
    rsq = u*u + v*v
    cosc = sinc_over_r = 1./np.sqrt(1.+rsq)
    cosdec = np.cos(center_dec)
    sindec = np.sin(center_dec)
    sindec = cosc * sindec + v * sinc_over_r * cosdec
    tandra_num = -u * sinc_over_r
    tandra_denom = cosc * cosdec - v * sinc_over_r * sindec
    dec = np.arcsin(sindec)
    ra = center_ra + np.arctan2(tandra_num, tandra_denom)
    return ra, dec

# ---------------------------------------------------------------------------------------- #

def radec_gnom(x, y, center_ra, center_dec, cd, crpix, pv):
    p1 = np.array( [ np.atleast_1d(x), np.atleast_1d(y) ] )
    p2 = np.dot(cd, p1 - crpix[:, np.newaxis])
    u = p2[0]
    v = p2[1]
    usq = u*u
    vsq = v*v
    ones = np.ones(u.shape)
    upow = np.array([ ones, u, usq, usq*u ])
    vpow = np.array([ ones, v, vsq, vsq*v ])
    temp = np.dot(pv, vpow)
    p2 = np.sum(upow * temp, axis=1)
    u = - p2[0] * degree_to_arcsec
    v = p2[1] * degree_to_arcsec
    ra, dec = deproject_gnom(u, v, center_ra, center_dec)
    return ra, dec

# ---------------------------------------------------------------------------------------- #

# Class for a pixel of the map, containing trees of images and values
class NDpix:

    def __init__(self, propertyArray_in, inweights, ratiores):
        self.ratiores = ratiores
        self.nbelem = 1
        self.propertyArray = [propertyArray_in]
        if self.ratiores > 1:
            self.weights = np.array([inweights])

    def addElem(self, propertyArray_in, inweights):
        self.nbelem += 1
        self.propertyArray.append(propertyArray_in)
        if self.ratiores > 1:
            self.weights = np.vstack( (self.weights, inweights) )


    # Project NDpix into a single number
    # for a given property and operation applied to its array of images
    def project(self, property, weights, operation, fracdetvals=None):

        asperpix = 0.263
        A = np.pi*(1.0/asperpix)**2
        # Computes COADD weights
        if weights == 'coaddweights3' or weights == 'coaddweights2' or weights == 'coaddweights' or property == 'maglimit2' or property == 'maglimit' or property == 'maglimit3' or property == 'sigmatot':
            m_zpi = np.array([proparr['MAGZP'] for proparr in self.propertyArray])
            if property == 'sigmatot':
                m_zp = np.array([30.0 for proparr in self.propertyArray])
            else:
                m_zp = np.array([proparr['COADD_MAGZP'] for proparr in self.propertyArray])
                
            if weights == 'coaddweights' or property == 'maglimit':
                sigma_bgi = np.array([
                    1.0/np.sqrt((proparr['WEIGHTA']+proparr['WEIGHTB'])/2.0)
                    if (proparr['WEIGHTA']+proparr['WEIGHTB']) >= 0.0 else proparr['SKYSIGMA']
                    for proparr in self.propertyArray])
            if weights == 'coaddweights2' or property == 'maglimit2':
                sigma_bgi = np.array([
                    0.5/np.sqrt(proparr['WEIGHTA'])+0.5/np.sqrt(proparr['WEIGHTB'])
                    if (proparr['WEIGHTA']+proparr['WEIGHTB']) >= 0.0 else proparr['SKYSIGMA']
                    for proparr in self.propertyArray])
            if weights == 'coaddweights3' or property == 'maglimit3' or property == 'sigmatot':
                sigma_bgi = np.array([proparr['SKYSIGMA'] for proparr in self.propertyArray])
            sigpis = 100**((m_zpi-m_zp)/5.0)
            mspis = (sigpis/sigma_bgi)**2.0
            pis = (sigpis/sigma_bgi)**2.0
        elif weights == 'invsqrtexptime':
            pis = np.array([ 1.0 / np.sqrt(proparr['EXPTIME']) for proparr in self.propertyArray])
        else:
            pis = np.array([1.0 for proparr in self.propertyArray])
            
        pis = np.divide(pis, pis.mean())

        # No super-resolution or averaging
        if self.ratiores == 1:
            if property == 'count':
                vals = np.array([1.0 for proparr in self.propertyArray])
            elif property == 'sigmatot':
                return np.sqrt(1.0 / mspis.sum())
            elif property == 'maglimit3' or property == 'maglimit2' or property == 'maglimit':
                sigma2_tot = 1.0 / mspis.sum()
                return np.mean(m_zp) - 2.5*np.log10(10*np.sqrt(A*sigma2_tot) )
            else:
                vals = np.array([proparr[property] for proparr in self.propertyArray])
                vals = vals * pis
            if operation == 'mean':
                return np.mean(vals)
            if operation == 'median':
                return np.median(vals)
            if operation == 'total':
                return np.sum(vals)
            if operation == 'min':
                return np.min(vals)
            if operation == 'max':
                return np.max(vals)
            if operation == 'maxmin':
                return np.max(vals) - np.min(vals)
            if operation == 'fracdet':
                return 1.0

        # Retrieve property array and apply operation (with super-resolution)
        if property == 'count':
            vals = np.array([1.0 for proparr in self.propertyArray])
        elif property == 'maglimit2' or property == 'maglimit' or property == 'maglimit3' or property == 'sigmatot':
            vals = (sigpis/sigma_bgi)**2
        else:
            vals = np.array([proparr[property] for proparr in self.propertyArray])
            vals = vals * pis

        theweights = self.weights
        intweights = np.ceil(theweights)
        floatweightedarray = (theweights.T * vals).T
        intweightedarray = (intweights.T * vals).T
        counts = (intweights.T * pis).sum(axis=1)
        ind = counts > 0
        
        if property == 'maglimit' or property == 'maglimit2' or property == 'maglimit3':
            sigma2_tot =  1.0 / intweightedarray.sum(axis=0)
            maglims = np.mean(m_zp) - 2.5*np.log10(10*np.sqrt(A*sigma2_tot) )
            return maglims[ind].mean()
        if property == 'sigmatot':
            sigma2_tot =  1.0 / intweightedarray.sum(axis=0)
            return np.sqrt(sigma2_tot)[ind].mean()
        if operation == 'min':
            return np.min(vals)
        if operation == 'max':
            return np.max(vals)
        if operation == 'maxmin':
            return np.max(vals) - np.min(vals)
        if operation == 'mean':
            return (intweightedarray.sum(axis=0) / counts)[ind].mean()
        if operation == 'median':
            return np.ma.median(np.ma.array(intweightedarray, mask=np.logical_not(theweights)), axis=0)[ind].mean()
        if operation == 'total':
            return floatweightedarray.sum(axis=0)[ind].mean()
        if operation == 'fracdet':
            temp = intweightedarray.sum(axis=0)
            return temp[ind].size / float(temp.size)


# ---------------------------------------------------------------------------------------- #

# Project NDpix into a value
def projectNDpix(args):
    pix, property, weights, operation = args
    if pix != 0:
        return pix.project(self, property, weights, operation)
    else:
        return hp.UNSEEN



# Create a "healtree", i.e. a set of pixels with trees of images in them.
def makeHealTree_CCDpixels(args):
    samplename, nside, ipixel_low, nside_low, subipixels_low_nest, ratiores, pixoffset, tbdata, local_dir, undersample = args
    treemap = HealTree(nside)
    verbcount = 1000
    count = 0
    vccd_count = 0
    start = time()
    duration = 0
    print ('>', samplename, ': starting tree making')
    for i, propertyArray in enumerate(tbdata):
        count += 1
        start_one = time()
        res = treemap.addElem_CCDpixels(propertyArray, ratiores, pixoffset, ipixel_low, nside_low, subipixels_low_nest, local_dir=local_dir, undersample=undersample)
        vccd_count += res
        #if vccd_count == 1:
        #    break
        end_one = time()
        duration += float(end_one - start_one)
        if count == verbcount:
            print ('>', samplename, ': processed images', i-verbcount+1, '-', i+1, '(on '+str(len(tbdata))+') in %.2f' % duration, 'sec (~ %.3f' % (duration/float(verbcount)), 'per image)')
            count = 0
            duration = 0
    end = time()
    print ('>', samplename, ': tree making took : %.2f' % float(end - start), 'sec for', vccd_count, 'images')
    return treemap


# Create a "healtree", i.e. a set of pixels with trees of images in them.
def makeHealTree_partial(args):
    samplename, nside, ipixel_low, nside_low, subipixels_low_nest, ratiores, pixoffset, tbdata = args
    treemap = HealTree(nside)
    verbcount = 1000
    count = 0
    vccd_count = 0
    start = time()
    duration = 0
    print ('>', samplename, ': starting tree making')
    for i, propertyArray in enumerate(tbdata):
        count += 1
        start_one = time()
        res = treemap.addElem_partial(propertyArray, ratiores, pixoffset, ipixel_low, nside_low, subipixels_low_nest)
        vccd_count += res
        #if vccd_count == 1:
        #    break
        end_one = time()
        duration += float(end_one - start_one)
        if count == verbcount:
            print ('>', samplename, ': processed images', i-verbcount+1, '-', i+1, '(on '+str(len(tbdata))+') in %.2f' % duration, 'sec (~ %.3f' % (duration/float(verbcount)), 'per image)')
            count = 0
            duration = 0
    end = time()
    print ('>', samplename, ': tree making took : %.2f' % float(end - start), 'sec for', vccd_count, 'images')
    return treemap


# Create a "healtree", i.e. a set of pixels with trees of images in them.
def makeHealTree(args):
    samplename, nside, ratiores, pixoffset, tbdata = args
    treemap = HealTree(nside)
    verbcount = 1000
    count = 0
    start = time()
    duration = 0
    print ('>', samplename, ': starting tree making')
    for i, propertyArray in enumerate(tbdata):
        count += 1
        start_one = time()
        treemap.addElem(propertyArray, ratiores, pixoffset)
        end_one = time()
        duration += float(end_one - start_one)
        if count == verbcount:
            print ('>', samplename, ': processed images', i-verbcount+1, '-', i+1, '(on '+str(len(tbdata))+') in %.2f' % duration, 'sec (~ %.3f' % (duration/float(verbcount)), 'per image)')
            count = 0
            duration = 0
    end = time()
    print ('>', samplename, ': tree making took : %.2f' % float(end - start), 'sec for', len(tbdata), 'images')
    return treemap


class Image(object):

    def __init__(self,filename,image_hdu='SCI',mask_hdu='MSK'):
        self.filename = filename
        self.image_hdu = image_hdu
        self.mask_hdu = mask_hdu
        self._readfile(filename)
        self._create_wcs()

    @property
    def corners(self):
        corners = []
        for i in range(1,5):
            corners.append( [self.header['RAC%i'%i],self.header['DECC%d'%i]] )
        return SkyCoord(np.array(corners),unit=u.deg,frame='icrs')

    @property
    def center(self):
        return SkyCoord(self.header['RA_CENT'],self.header['DEC_CENT'],unit=u.deg,frame='icrs')

    def _readfile(self,filename):
        self.fits = pyfits.open(filename)
        self.header = self.fits[self.image_hdu].header
        self.data = self.fits[self.mask_hdu].data

    def _create_wcs(self):
        self.wcs = astropy.wcs.WCS(self.header)

    def get_radius(self, epsilon=0.0):
        sep = self.center.separation(self.corners)
        return np.max(sep.deg)+epsilon
        
    def healpixify(self, nside=4096, nest=False):
        # Determine the radius of the image
        radius = self.get_radius(epsilon=0.01)
        center = self.center

        vec = hp.ang2vec(np.radians(90. - center.dec.deg), np.radians(center.ra.deg))

        inclusive, fact = False, 4
        hpx = hp.query_disc(nside, vec, np.radians(radius), inclusive, fact, nest)

        theta, phi = hp.pix2ang(nside, hpx, nest)
        ra, dec = np.degrees(phi), 90. - np.degrees(theta)

        xpix,ypix = self.wcs.wcs_world2pix(ra,dec,0)
        xpix,ypix = np.round([xpix, ypix]).astype(int)
        shape = self.data.shape
        sel = (xpix > 0) & (xpix < shape[1]) \
            & (ypix > 0) & (ypix < shape[0])
        xpix = xpix[sel]
        ypix = ypix[sel]

        return hpx[sel], self.data[ypix,xpix], theta[sel], phi[sel]

# ---------------------------------------------------------------------------------------- #

# Class for multi-dimensional healpix map that can be
# created and processed in parallel.
class HealTree:

    # Initialise and create array of pixels
    def __init__(self, nside):
        self.nside = nside
        self.npix = 12*nside**2
        self.pixlist = np.zeros(self.npix, dtype=object)

    # Process image and absorb its properties
    def addElem(self, propertyArray, ratiores, pixoffset):
        # Retrieve pixel indices
        ipixels, weights, thetas_c, phis_c, subpixring_weights = computeHPXpix_sequ_new(self.nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
        # For each pixel, absorb image properties
        for ii, (ipix, weight) in enumerate(zip(ipixels, weights)):
            if self.pixlist[ipix] == 0:
                self.pixlist[ipix] = NDpix(propertyArray, subpixring_weights[ii, :], ratiores)
            else:
                self.pixlist[ipix].addElem(propertyArray, subpixring_weights[ii, :])

    # Process image and absorb its properties
    def addElem_CCDpixels2(self, propertyArray, ratiores, pixoffset, ipixel_low, nside_low, subipixels_low_nest, local_dir='.', undersample=1):
        img_ras_c, img_decs_c = computeCorners_WCS_TPV(propertyArray, pixoffset)
        img_phis_c = img_ras_c * np.pi/180
        img_thetas_c =  np.pi/2  - img_decs_c * np.pi/180
        img_pix = hp.ang2pix(nside_low, img_thetas_c, img_phis_c, nest=True)
        if ipixel_low not in img_pix:
            return 0

        fname_local = local_dir + '/' + propertyArray['path'].strip() + '/' + propertyArray['filename'].strip()
        print(fname_local)
        hdulist = pyfits.open(fname_local)        
        header = hdulist['SCI'].header
        flatmask = hdulist['MSK'].data[::undersample, ::undersample].T.ravel()
        hdulist.close()

        origin = 0
        wcs = astropy.wcs.WCS(header)
        xline = np.arange(origin+pixoffset, propertyArray['NAXIS1']-pixoffset+origin, undersample)
        yline = np.arange(origin+pixoffset, propertyArray['NAXIS2']-pixoffset+origin, undersample)
        x, y = np.meshgrid(xline, yline)
        xy = np.vstack((x.ravel(), y.ravel())).T
        img_radecs = wcs.all_pix2world(xy, origin)
        img_ras, img_decs = img_radecs[:, 0], img_radecs[:, 1]
        img_phis = img_ras * np.pi/180
        img_thetas =  np.pi/2  - img_decs * np.pi/180

        nonunique_ipixs_ring = hp.ang2pix(self.nside, img_thetas, img_phis, nest=False)
        ccd_subipixs_nest = hp.ang2pix(self.nside*ratiores, img_thetas, img_phis, nest=True)
        ipixs_ring = np.unique(nonunique_ipixs_ring)
        ipixs_nest = hp.ring2nest(self.nside, ipixs_ring)
        if ratiores > 1:
            nsubpixperpix = ratiores**2
            subipixs_nest = np.concatenate([np.arange(ipix*nsubpixperpix, ipix*nsubpixperpix+nsubpixperpix, dtype=np.int64) 
                for ipix in ipixs_nest]).reshape((ipixs_nest.size, nsubpixperpix))
        else:
            nsubpixperpix = 1
            subipixs_nest = ipixs_nest.reshape((ipixs_nest.size, nsubpixperpix))
        #if ratiores < 2**3 and self.nside <= 4096:
        #ccdmask = subipixs_nest[:, :, None] == ccd_subipixs_nest[None, None, :]
        binmask = 0*flatmask
        binmask[(flatmask & 2047) == 0] = 1
        del flatmask
        #binmask[:] = 1

        def frac(arr):
            if len(arr) == 0:
                return 0
            else:
                return arr.sum() / float(arr.size)

        for ii, ipix in enumerate(ipixs_ring):
            if ipix in subipixels_low_nest:
                #if ratiores < 2**3 and self.nside <= 4096:
                #resubweights = np.array([frac(binmask[ccdmask[ii, ii2, :]]) 
                #        for ii2 in range(nsubpixperpix)])
                #else:
                resubweights = np.array([frac(binmask[ccd_subipixs_nest == subipixs_nest[ii, ii2]]) 
                        for ii2 in range(nsubpixperpix)])
                if self.pixlist[ipix] == 0:
                    self.pixlist[ipix] = NDpix(propertyArray, resubweights, ratiores)
                else:
                    self.pixlist[ipix].addElem(propertyArray, resubweights)
        return 1

    # Process image and absorb its properties
    def addElem_CCDpixels(self, propertyArray, ratiores, pixoffset, ipixel_low, nside_low, subipixels_low_nest, 
                coadd_cut=True, local_dir='.', undersample=1):

        t1 = time()
        img_ras_c, img_decs_c = computeCorners_WCS_TPV(propertyArray, pixoffset)
        img_phis_c = img_ras_c * np.pi/180
        img_thetas_c =  np.pi/2  - img_decs_c * np.pi/180
        img_pix = hp.ang2pix(nside_low, img_thetas_c, img_phis_c, nest=True)
        if ipixel_low not in img_pix:
            return 0
        t2 = time()

        fname_local = local_dir + '/' + propertyArray['path'].strip() + '/' + propertyArray['filename'].strip()
        img = Image(fname_local)
        t3 = time()

        subipix_nest, subval, subpix_thetas, subpix_phis = img.healpixify(self.nside*ratiores, nest=True)
        ipix_nest = subipix_nest // ratiores**2
        subipix_nest_indices = subipix_nest % ratiores**2
        t4 = time()

        sel = (subval & 2047) == 0
        ipix_nest = ipix_nest[sel]
        subipix_nest = subipix_nest[sel]
        subpix_thetas, subpix_phis = subpix_thetas[sel], subpix_phis[sel]
        subipix_nest_indices = subipix_nest_indices[sel]
        unique_ipix_nest = np.intersect1d(ipix_nest, subipixels_low_nest)
        t5 = time()

        if coadd_cut:
            ind_U, ind_L, ind_R, ind_B = 0, 2, 3, 1
            #coadd_ras = [propertyArray[v] for v in ['URAUL', 'URALL', 'URALR', 'URAUR']]
            #coadd_decs = [propertyArray[v] for v in ['UDECUL', 'UDECLL', 'UDECLR', 'UDECUR']]
            coadd_ras = [float(propertyArray[v]) for v in ['RAC3', 'RAC2', 'RAC1', 'RAC4']]
            coadd_decs = [float(propertyArray[v]) for v in ['DECC3', 'DECC2', 'DECC1', 'DECC4']]
            coadd_phis = np.multiply(coadd_ras, np.pi/180)
            coadd_thetas =  np.pi/2  - np.multiply(coadd_decs, np.pi/180)
            #coadd_phis= np.mod( coadd_phis + np.pi, 2*np.pi )
            pmax = np.max(subpix_phis)
            pmin = np.min(subpix_phis)
            if (pmax - pmin > np.pi) or (np.max(coadd_phis) - np.min(coadd_phis) > np.pi):
                coadd_phis= np.mod( coadd_phis + np.pi, 2*np.pi )
                subpix_phis = np.mod( rangepix_phis + np.pi, 2*np.pi )

        for ii, theipix_nest in enumerate(unique_ipix_nest):
            resubweights = np.repeat(False, ratiores**2)
            if coadd_cut:
                mask = ipix_nest == theipix_nest
                #resubweights[subipix_nest_indices[mask]] = True
                temp = in_region(subpix_thetas[mask], subpix_phis[mask],
                                          coadd_thetas[ind_U], coadd_phis[ind_U], coadd_thetas[ind_L], coadd_phis[ind_L],
                                          coadd_thetas[ind_R], coadd_phis[ind_R], coadd_thetas[ind_B], coadd_phis[ind_B])
                resubweights[subipix_nest_indices[mask]] = temp
            else:
                resubweights[subipix_nest_indices[ipix_nest == theipix_nest]] = True
            if self.pixlist[theipix_nest] == 0:
                self.pixlist[theipix_nest] = NDpix(propertyArray, resubweights, ratiores)
            else:
                self.pixlist[theipix_nest].addElem(propertyArray, resubweights)

        t6 = time()
        #print('Times:', t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, (t6-t5)/unique_ipix_nest.size)
        return 1

    # Process image and absorb its properties
    def addElem_CCDpixels_withoutastropywcs(self, propertyArray, ratiores, pixoffset, ipixel_low, nside_low, local_dir='.', undersample=1):
        # Retrieve pixel indices
        out = computeHPXpix_CCDpixels(self.nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores, ipixel_low=ipixel_low, nside_low=nside_low, undersample=undersample)
        if out is not None:
            unique_ipixs_ring, sweights, img_thetas, img_phis, subpixring_weights, ccdmask = out
            fname_local = local_dir + '/' + propertyArray['path'].strip() + '/' + propertyArray['filename'].strip()

            hdulist = pyfits.open(fname_local)
            flatmask = hdulist[2].data[::undersample, ::undersample].T.ravel()
            hdulist.close()

            binmask = 0*flatmask
            binmask[(flatmask & 2047) == 0] = 1
            #binmask[:] = 1

            def frac(arr):
                if len(arr) == 0:
                    return 0
                else:
                    return arr.sum() / float(arr.size)

            for ii, (ipix, weight) in enumerate(zip(unique_ipixs_ring, sweights)):
                resubweights = np.array([frac(binmask[ccdmask[ii, ii2, :]]) for ii2 in range(ccdmask.shape[1])])
                print(ii)
                print(resubweights)
                print(subpixring_weights[ii, :])
                if self.pixlist[ipix] == 0:
                    self.pixlist[ipix] = NDpix(propertyArray, resubweights, ratiores)
                else:
                    self.pixlist[ipix].addElem(propertyArray, resubweights)
            return 1
        return 0

    # Process image and absorb its properties
    def addElem_partial(self, propertyArray, ratiores, pixoffset, ipixel_low, nside_low, subipixels_low_nest):
        # Retrieve pixel indices
        out = computeHPXpix_sequ_new(self.nside, propertyArray, pixoffset=pixoffset, coadd_cut=True, ratiores=ratiores, ipixel_low=ipixel_low, nside_low=nside_low)
        if out is not None:
            ipixels, weights, thetas_c, phis_c, subpixring_weights = out
            ipixels_nest = hp.ring2nest(self.nside, ipixels)
            mask = np.in1d(ipixels_nest, subipixels_low_nest)
            # For each pixel, absorb image properties
            for ii, (ipix_nest, weight) in enumerate(zip(ipixels_nest, weights)):
                if mask[ii] == True:
                    if self.pixlist[ipix_nest] == 0:
                        self.pixlist[ipix_nest] = NDpix(propertyArray, subpixring_weights[ii, :], ratiores)
                    else:
                        self.pixlist[ipix_nest].addElem(propertyArray, subpixring_weights[ii, :])
            return 1
        return 0

     # Project HealTree into partial Healpix map
     # for a given property and operation applied to its array of images
    def project_partial(self, property, weights, operation, pool=None):
        ind = np.where(self.pixlist != 0)[0]
        verbcount = ind.size / 10
        count = 0
        start = time()
        duration = 0
        signal = np.zeros(ind.size)
        for i, pix in enumerate(self.pixlist[ind]):
            count += 1
            start_one = time()
            signal[i] = pix.project(property, weights, operation)
            end_one = time()
            duration += float(end_one - start_one)
            if count == verbcount:
                print ('>', property, weights, operation, ': processed pixels', i-verbcount+1, '-', i+1, '(on '+str(ind.size)+') in %.1e' % duration, 'sec (~ %.1e' % (duration/float(verbcount)), 'per pixel)')
                count = 0
                duration = 0
        end = time()
        print ('> Projection', property, weights, operation, ' took : %.2f' % float(end - start), 'sec for', ind.size, 'pixels')
        #signal = [pix.project(property, weights, operation) for pix in self.pixlist[ind]]
        return ind, signal

     # Project HealTree into regular Healpix map
     # for a given property and operation applied to its array of images
    def project(self, property, weights, operation, pool=None):
        outmap = np.zeros(self.npix)
        outmap.fill(hp.UNSEEN)
        if pool is None:
            for ipix, pix in enumerate(self.pixlist):
                if pix != 0:
                    outmap[ipix] = pix.project(property, weights, operation)
        else:
            outmap = np.array( pool.map( projectNDpix, [ (pix, property, weights, operation) for pix in self.pixlist ] ) )
        return outmap

# ---------------------------------------------------------------------------------------- #

def makeHpxMap(args):
    healtree, property, weights, operation = args
    return healtree.project(property, weights, operation)

# ---------------------------------------------------------------------------------------- #

def makeHpxMap_partial(args):
    healtree, property, weights, operation = args
    return healtree.project_partial(property, weights, operation)

# ---------------------------------------------------------------------------------------- #

def addElemHealTree(args):
    healTree, propertyArray, ratiores = args
    healTree.addElem(propertyArray, ratiores)

# ---------------------------------------------------------------------------------------- #

# Process image and absorb its properties
def addElem(args):
    iarr, tbdatadtype, propertyArray, nside, propertiesToKeep, ratiores = args
    propertyArray.dtype = tbdatadtype
    print ('Processing image', iarr, propertyArray['RA'])
    # Retrieve pixel indices
    ipixels, weights, thetas_c, phis_c = computeHPXpix_sequ_new(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
    print ('Processing image', iarr, thetas_c, phis_c)
    # For each pixel, absorb image properties
    for ipix, weight in zip(ipixels, weights):
        if globalTree[ipix] == 0:
            globalTree[ipix] = NDpix(propertyArray, propertiesToKeep, weight=weight)
        else:
            globalTree[ipix].addElem(propertyArray, propertiesToKeep, weight=weight)

# ---------------------------------------------------------------------------------------- #

# Read and project a Healtree into Healpix maps, and write them.
def project_and_write_maps(mode, propertiesweightsoperations, tbdata, catalogue_name, outrootdir, sample_names, inds, nside, ratiores, pixoffset, ipixel_low=1, nside_low=1, nsidesout=None, local_dir='.', undersample=1):

    resol_prefix = 'nside'+str(nside)+'_oversamp'+str(ratiores)
    outroot = outrootdir + '/' + catalogue_name + '/' + resol_prefix + '/'
    mkdir_p(outroot)
    if mode == 1: # Fully sequential
        for sample_name, ind in zip(sample_names, inds):
            treemap = makeHealTree( (catalogue_name+'_'+sample_name, nside, ratiores, pixoffset, np.array(tbdata[ind])) )
            for property, weights, operation in propertiesweightsoperations:
                cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
                if nsidesout is None:
                    fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                    print ('Creating and writing', fname)
                    write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)
                else:
                    cutmap_indices_nest = hp.ring2nest(nside, cutmap_indices)
                    outmap_hi = np.zeros(hp.nside2npix(nside))
                    outmap_hi.fill(0.0) #outmap_hi.fill(hp.UNSEEN)
                    outmap_hi[cutmap_indices_nest] = cutmap_signal
                    for nside_out in nsidesout:
                        if nside_out == nside:
                            outmap_lo = outmap_hi
                        else:
                            outmap_lo = hp.ud_grade(outmap_hi, nside_out, order_in='NESTED', order_out='NESTED')
                        resol_prefix2 = 'nside'+str(nside_out)+'from'+str(nside)+'o'+str(ratiores)
                        outroot2 = outrootdir + '/' + catalogue_name + '/' + resol_prefix2 + '/'
                        mkdir_p(outroot2)
                        fname = outroot2 + '_'.join([catalogue_name, sample_name, resol_prefix2, property, weights, operation]) + '.fits'
                        print ('Writing', fname)
                        hp.write_map(fname, outmap_lo, nest=True)
                        subprocess.call("gzip "+fname,shell=True)


    if mode == 3: # Fully parallel
        pool = Pool(len(inds))
        print ('Creating HealTrees')
        treemaps = pool.map( makeHealTree,
                         [ (catalogue_name+'_'+samplename, nside, ratiores, pixoffset, np.array(tbdata[ind]))
                           for samplename, ind in zip(sample_names, inds) ] )

        for property, weights, operation in propertiesweightsoperations:
            print ('Making maps for', property, weights, operation)
            outmaps = pool.map( makeHpxMap_partial,
                            [ (treemap, property, weights, operation) for treemap in treemaps ] )
            for sample_name, outmap in zip(sample_names, outmaps):
                fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                print ('Writing', fname)
                cutmap_indices, cutmap_signal = outmap
                write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)


    if mode == 2:  # Parallel tree making and sequential writing
        pool = Pool(len(inds))
        print ('Creating HealTrees')
        treemaps = pool.map( makeHealTree,
                         [ (catalogue_name+'_'+samplename, nside, ratiores, pixoffset, np.array(tbdata[ind]))
                           for samplename, ind in zip(sample_names, inds) ] )

        for property, weights, operation in propertiesweightsoperations:
            for sample_name, treemap in zip(sample_names, treemaps):
                fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '.fits'
                print ('Writing', fname)
                #outmap = makeHpxMap( (treemap, property, weights, operation) )
                #hp.write_map(fname, outmap, nest=False)
                cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
                write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=False)

    if mode >= 4: # Fully sequential with ccds grouped in pixel
        ratiores2 = 2 ** (np.log(nside / nside_low) / np.log(2))
        subipixs_nest = np.arange(ipixel_low*ratiores2**2, ipixel_low*ratiores2**2+ratiores2**2, dtype=np.int64)
        #subipixs_ring = hp.nest2ring(nside, subipixs_nest)
        #invert_subipixs_ring = np.setdiff1d(np.arange(hp.nside2npix(nside)), subipixs_ring)
        for sample_name, ind in zip(sample_names, inds):
            if mode == 5:
                treemap = makeHealTree_CCDpixels( (catalogue_name+'_'+sample_name, nside, ipixel_low, nside_low, subipixs_nest, ratiores, pixoffset, np.array(tbdata[ind]), local_dir, undersample,) )
            if mode == 4:
                treemap = makeHealTree_partial( (catalogue_name+'_'+sample_name, nside, ipixel_low, nside_low, subipixs_nest, ratiores, pixoffset, np.array(tbdata[ind])) )
            #treemap.pixlist[invert_subipixs_ring] = np.zeros(invert_subipixs_ring.size, dtype=object)
            for property, weights, operation in propertiesweightsoperations:
                cutmap_indices, cutmap_signal = makeHpxMap_partial( (treemap, property, weights, operation) )
                if np.sum(treemap.pixlist != 0) > 0:
                    if nsidesout is None:
                        fname = outroot + '_'.join([catalogue_name, sample_name, resol_prefix, property, weights, operation]) + '_' + str(ipixel_low) + '.fits'
                        print ('Creating and writing', fname)
                        write_partial_map(fname, cutmap_indices, cutmap_signal, nside, nest=True)
                    else:
                        stop
# ---------------------------------------------------------------------------------------- #


if False:
    fname = '/Users/bl/Downloads/OPS/finalcut/Y2A1-2mass/Y2-2368/20140828/D00352819/p01/red/immask/D00352819_g_c45_r2368p01_immasked.fits.fz'
    hdulist = pyfits.open(fname)
    #imgdata1 = hdulist[1].data
    #imgdata2 = hdulist[2].data
    #imgdata3 = hdulist[3].data
    #hdulist.close()
    header = hdulist['SCI'].header
    flatmask = hdulist['MSK'].data.T.ravel()
    hdulist.close()
    binmask = np.zeros(flatmask.size, dtype=bool)
    binmask[(flatmask & 2047) == 0] = True
    ind = np.where((flatmask & 2047) != 0)[0]
    print(binmask, binmask.sum(), binmask.size, np.sum((flatmask & 2047) != 0))

    np.save('D00469620_Y_c01_r2371p01_mask.npy', binmask)
    np.save('D00469620_Y_c01_r2371p01_mask2.npy', ~binmask)
    np.save('D00469620_Y_c01_r2371p01_mask3.npy', ind)
    np.save('D00469620_Y_c01_r2371p01_mask4.npy', np.packbits(binmask))
    np.save('D00469620_Y_c01_r2371p01_mask5.npy', np.packbits(~binmask))
    np.savez_compressed('D00469620_Y_c01_r2371p01_maskb.npy', binmask)
    np.savez_compressed('D00469620_Y_c01_r2371p01_maskb2.npy', ~binmask)
    np.savez_compressed('D00469620_Y_c01_r2371p01_maskb3.npy', ind)
    np.savez_compressed('D00469620_Y_c01_r2371p01_maskb4.npy', np.packbits(binmask))
    np.savez_compressed('D00469620_Y_c01_r2371p01_maskb5.npy', np.packbits(~binmask))
    hdu = pyfits.BinTableHDU.from_columns([pyfits.Column(name='mask', array=binmask, format='L')])
    hdu.writeto('D00469620_Y_c01_r2371p01_mask.fits')
    hdu = pyfits.BinTableHDU.from_columns([pyfits.Column(name='ind', array=ind, format='I')])
    hdu.writeto('D00469620_Y_c01_r2371p01_mask2.fits')
    hdu = pyfits.BinTableHDU.from_columns([pyfits.Column(name='mask', array=np.packbits(binmask), format='B')])
    hdu.writeto('D00469620_Y_c01_r2371p01_mask3.fits')

    stop
    #hdulist = pyfits.open('/Users/bl/Dropbox/Projects/quicksip/data/Y1A1_SPT_IMAGEINFO_and_COADDINFO.fits')

    hdulist = pyfits.open('/Users/bl/Dropbox/repos/QuickSip/y3a2_quicksip_info_000004.fits')
    tbdata = hdulist[1].data
    hdulist.close()

    iccd = 10

    propertyArray = tbdata[iccd]
    #coadd_ras = [propertyArray[v] for v in ['URAUL', 'URALL', 'URALR', 'URAUR']]
    #coadd_decs = [propertyArray[v] for v in ['UDECUL', 'UDECLL', 'UDECLR', 'UDECUR']]
    coadd_ras = [float(propertyArray[v]) for v in ['RAC3', 'RAC2', 'RAC1', 'RAC4']]
    coadd_decs = [float(propertyArray[v]) for v in ['DECC3', 'DECC2', 'DECC1', 'DECC4']]
    text = [str(i+1) for i in range(4)]

    print(coadd_ras)
    print(coadd_decs)
    plt.plot(coadd_ras, coadd_decs)
    for i in range(4):
        plt.text(coadd_ras[i], coadd_decs[i], text[i])
    plt.show()

    stop

    #pix_thetas, pix_phis = hp.pix2ang(nside, ipixs_ring, nest=False)

    local_dir = '.'
    #local_dir = '/archive_data/desarchive/'

    fname_remote = tbdata[iccd]['basepath'] + '/' + tbdata[iccd]['path'] + '/' + tbdata[iccd]['filename']
    fname_local = local_dir + '/' + tbdata[iccd]['path'] + '/' + tbdata[iccd]['filename']

    password_manager = urllib2.HTTPPasswordMgrWithPriorAuth()
    password_manager.add_password(None, fname_remote, user, password, is_authenticated=True)
    auth_manager = urllib2.HTTPBasicAuthHandler(password_manager)
    opener = urllib2.build_opener(auth_manager)


    #os.makedirs(local_dir + tbdata[iccd]['path'])

    if not os.path.isdir(local_dir + '/' + tbdata[iccd]['path']):
        os.makedirs(local_dir + '/' + tbdata[iccd]['path'])

    if not os.path.exists(fname_local):
        with open(fname_local, 'wb') as fd:
            with opener.open(fname_remote) as f:
                fd.write(f.read())

    undersample = 2

    hdulist = pyfits.open(fname_local)
    flatmask = hdulist[2].data.T[::undersample, ::undersample].ravel()
    hdulist.close()


    nside = 1024
    ratiores = 2
    out = computeHPXpix_CCDpixels(nside, tbdata[iccd], pixoffset=0, ratiores=ratiores, coadd_cut=False, ipixel_low=None, nside_low=None, undersample=undersample)
    ipixs_ring, sweights, img_thetas, img_phis, resubweights, ccdmask = out


    fig, axs = plt.subplots(4, 4)
    axs = axs.ravel()
    cols = ['red', 'green', 'yellow', 'blue', 'orange', 'purple']
    for i1, ipix_ring in enumerate(ipixs_ring[:16]):
        axs[i1].scatter(img_thetas, img_phis, marker='.', s=1, color='k', alpha=0.4)
        for i2 in range(ccdmask.shape[1]):
            mask = ccdmask[i1, i2, :]
            axs[i1].scatter(img_thetas[mask], img_phis[mask], marker='.', s=3, color=cols[i2])
        axs[i1].set_xlim([0.9999*np.min(img_thetas), 1.0001*np.max(img_thetas)])
        axs[i1].set_ylim([0.9999*np.min(img_phis), 1.0001*np.max(img_phis)])
    fig.tight_layout()
    plt.show()

    #TODO: USE https://github.com/kadrlica/healpixify/blob/master/bin/healpixify

def test():
    fname = '/Users/bl/Dropbox/Projects/Quicksip/data/SVA1_COADD_ASTROM_PSF_INFO.fits'
    #fname = '/Users/bl/Dropbox/Projects/Quicksip/data/Y1A1_IMAGEINFO_and_COADDINFO.fits'
    pixoffset = 10
    hdulist = pyfits.open(fname)
    tbdata = hdulist[1].data
    hdulist.close()
    nside = 1024
    ratiores = 4
    treemap = HealTree(nside)
    #results = pool.map(treemap.addElem, [imagedata for imagedata in tbdata])
    print (tbdata.dtype)
    #ind = np.ndarray([0])
    ind = np.where( tbdata['band'] == 'i' )
    import numpy.random
    ind = numpy.random.choice(ind[0], 1 )
    print ('Number of images :', len(ind))
    hpxmap = np.zeros(hp.nside2npix(nside))
    ras_c = []
    decs_c = []
    for i, propertyArray in enumerate(tbdata[ind]):
        ras_c.append(propertyArray['RA'])
        decs_c.append(propertyArray['DEC'])
    plt.figure()
    for i, propertyArray in enumerate(tbdata[ind]):
        print (i)
        propertyArray.dtype = tbdata.dtype
        listpix, weights, thetas_c, phis_c, listpix_sup = computeHPXpix_sequ_new(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
        #listpix2, weights2, thetas_c2, phis_c2 = computeHPXpix_sequ(nside, propertyArray, pixoffset=pixoffset, ratiores=ratiores)
        hpxmap = np.zeros(hp.nside2npix(nside))
        hpxmap[listpix] = weights
        hpxmap_sup = np.zeros(hp.nside2npix(ratiores*nside))
        hpxmap_sup[listpix_sup] = 1.0
        listpix_hi, weights_hi, thetas_c_hi, phis_c_hi, superind_hi = computeHPXpix_sequ_new(ratiores*nside, propertyArray, pixoffset=pixoffset, ratiores=1)
        hpxmap_hi = np.zeros(hp.nside2npix(ratiores*nside))
        hpxmap_hi[listpix_hi] = weights_hi
        hpxmap_hitolo = hp.ud_grade(hpxmap_hi, nside)
        hp.gnomview(hpxmap_hi, title='hpxmap_hi', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
        hp.gnomview(hpxmap_sup, title='hpxmap_sup', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
        hp.gnomview(hpxmap_hitolo, title='hpxmap_hitolo', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
        hp.gnomview(hpxmap, title='hpxmap', rot=[propertyArray['RA'], propertyArray['DEC']], reso=0.2)
        #plt.plot(phis_c, thetas_c)
        thetas, phis = hp.pix2ang(nside, listpix)
        #plt.scatter(phis, thetas, color='red', marker='o', s=50*weights)
        #plt.scatter(propertyArray['RA']*np.pi/180, np.pi/2 - propertyArray['DEC']*np.pi/180)
        #plt.text(propertyArray['RA']*np.pi/180, np.pi/2 - propertyArray['DEC']*np.pi/180, str(i))
    plt.show()
    stop

#if __name__ == "__main__":
#    test()
