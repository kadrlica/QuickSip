select cast(i.expnum as NUMBER(8)) as expnum,
       cast(i.ccdnum as NUMBER(4)) as ccdnum,
       i.skyvara,i.skyvarb,i.tilename,
       cast(i.band as VARCHAR(1)) as band,
       i.skysigma,i.fwhm,i.exptime,i.airmass,i.skybrite,
       i.crpix1,i.crpix2,
       i.crval1,i.crval2,i.cunit1,i.cunit2,i.cd1_1,i.cd1_2,
       i.cd2_1,i.cd2_2,i.pv1_0,i.pv1_1,i.pv1_2,i.pv1_3,
       i.pv1_4,i.pv1_5,i.pv1_6,i.pv1_7,i.pv1_8,i.pv1_9,i.pv1_10,
       i.pv2_0,i.pv2_1,i.pv2_2,i.pv2_3,
       i.pv2_4,i.pv2_5,i.pv2_6,i.pv2_7,i.pv2_8,i.pv2_9,i.pv2_10,
       i.naxis1,i.naxis2,
       c.uramin,c.uramax,
       c.udecmin,c.udecmax,
       cast(c.crossra0 as VARCHAR(1)) as crossra0,
       cast('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/' as VARCHAR(60)) as basepath,
       a.path, a.filename||a.compression as filename
from y3a2_image i, y3a2_coaddtile_geom c,
     y3a2_image i2, y3a2_file_archive_info a
where i.tilename=c.tilename and i.filetype='coadd_nwgint'
and i2.expnum = i.expnum and i2.ccdnum = i.ccdnum and i2.filetype='red_immask'
and i2.filename = a.filename
and rownum < 10
; > y3a2_quicksip_info.fits