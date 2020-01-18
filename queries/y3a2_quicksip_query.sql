-- This query downloads the necessary geometry, quality, and zeropoint info for each CCD.
-- The query takes ~300s (2020-01-18) and can be executed with:
--   easyaccess -s dessci -l [this file name]

SELECT CAST(i.expnum AS NUMBER(8)) AS expnum,
       CAST(i.ccdnum AS NUMBER(4)) AS ccdnum,
       CAST(i.band AS VARCHAR(1)) AS band,
       i.tilename,
       -- CCD center
       i.ra_cent, i.dec_cent,
       -- CCD performance metrics
       i.skyvara,i.skyvarb,
       i.skysigma,i.fwhm,i.exptime,i.airmass,i.skybrite,
       -- CCD location and distortion terms
       i.crpix1,i.crpix2,
       i.crval1,i.crval2,i.cunit1,i.cunit2,i.cd1_1,i.cd1_2,
       i.cd2_1,i.cd2_2,i.pv1_0,i.pv1_1,i.pv1_2,i.pv1_3,
       i.pv1_4,i.pv1_5,i.pv1_6,i.pv1_7,i.pv1_8,i.pv1_9,i.pv1_10,
       i.pv2_0,i.pv2_1,i.pv2_2,i.pv2_3,
       i.pv2_4,i.pv2_5,i.pv2_6,i.pv2_7,i.pv2_8,i.pv2_9,i.pv2_10,
       i.naxis1,i.naxis2,
       -- CCD zeropoint
       z.mag_zero as magzp
       -- CCD filename
       CAST('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/' AS VARCHAR(60)) AS basepath,
       a.path, a.filename||a.compression as filename
FROM y3a2_image i, y3a2_image i2, 
     y3a2_file_archive_info a, y3a2_zeropoint z
WHERE i.filetype='coadd_nwgint'
AND i2.expnum = i.expnum AND i2.ccdnum = i.ccdnum AND i2.filetype='red_immask'
AND i2.filename = a.filename
AND z.expnum = i.expnum AND z.ccdnum = i.ccdnum
AND z.source='FGCM' AND z.version='v2.0' AND z.flag < 16
; > y3a2_quicksip_all.fits

-- To make smaller subsamples for testing query

--AND rownum <= 10000
--; > y3a2_quicksip_small.fits

--AND rownum <= 500000
--; > y3a2_quicksip_medium.fits
