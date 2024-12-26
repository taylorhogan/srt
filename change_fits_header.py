from astropy.io import fits
from numpy.core.defchararray import equal
import os



def change_fits_header (in_path, keyword, oldval, newval, out_path):

    img=fits.open(in_path)
    img_data=img[0].data
    header=img[0].header
    if header[keyword] == oldval:
        print (in_path)
        header[keyword] = newval
        fits.writeto(out_path, img_data, header, overwrite=True)





rootdir =  "/Users/taylorhogan/Desktop/calibration_new"

for subdir, dirs, files in os.walk(rootdir):
     for file in files:
         if file.endswith (".fits"):
             path = os.path.join(subdir,file)
             change_fits_header(path, "filter", "L", "l", path)
