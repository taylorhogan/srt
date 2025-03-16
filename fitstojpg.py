from fnmatch import fnmatch

from astropy.visualization import ZScaleInterval
from astropy.io import fits
from matplotlib import pyplot as plt
import os
from pathlib import Path
import fnmatch
from PIL import Image

def convert_to_jpg2(in_file):
    out_file = in_file.replace('.fits', '.jpg')
    with fits.open(in_file) as hdul:
        # Get the image data from the first HDU
        image_data = hdul[0].data
        zscale = ZScaleInterval()
        #image_data = zscale(image_data)

        # Scale the image data to 0-255 for 8-bit image
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
        image_data = image_data.astype('uint8')

        # Create a PIL Image object
        img = Image.fromarray(image_data)

        # Save the image as a JPEG
        img.save("foo.jpg")


def convert_to_jpg(in_file):
    print(in_file)
    out_file = in_file.replace('.fits', '.jpg')
    data = fits.getdata(in_file)
    zscale = ZScaleInterval()

    plt.plot(zscale(data))

    plt.imshow(zscale(data), cmap="gray")
    path_list = in_file.split(os.sep)
    l = len(path_list)
    dso = path_list[l-4]
    filename = path_list[l-1]

    title = dso + "\n" + filename
    plt.title(title)

    #plt.savefig("foo.jpg")

    plt.savefig(out_file)
    return out_file


def convert_dir(dir_name):
    for cur, _dirs, files in os.walk(dir_name):
        for candidate in fnmatch.filter(files, '*.fits'):
            in_file = os.path.join(cur, candidate)
            new_file = in_file.replace('.fits', '.jpg')
            if not os.path.exists(new_file):
                print(in_file)
                convert_to_jpg(in_file)


def get_latest_file(directory, extension):
    latest_file = None
    latest_time = 0

    # Recursively search for files with the given extension
    for file_path in Path(directory).rglob(f"*.{extension}"):
        file_mod_time = file_path.stat().st_mtime
        # Update latest file if this file is more recent
        if file_mod_time > latest_time:
            latest_time = file_mod_time
            latest_file = file_path

    return latest_file

# convert_to_jpg("base_images/2024-09-11_20-21-36_Ha_-19.90_300.00s_1x1_0030.fits")
#f = get_latest_file("/Users/taylorhogan/Desktop/NGC 6888/2024-09-11", "fits")
#convert_to_jpg(str(f))

#convert_to_jpg("base_images/2024-09-11_20-21-36_Ha_-19.90_300.00s_1x1_0030.fits")
# convert_to_jpg("base_images/2024-11-14_00-51-54_R_1767_300.00s_1x1_0086.fits")
