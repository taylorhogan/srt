
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline



def fit_to_3d_plot(fits_path):
    # Open the FITS file
    with fits.open(fits_path) as hdul:
        # Assume the data is in the primary HDU for simplicity
        data = hdul[0].data

    fit_to_3d_plot_data(data)

def fit_to_3d_plot_data(data, output_file, title):


    # Check if the data is 2D, if not, we'll need to adjust
    if data.ndim != 2:
        raise ValueError("Expected 2D image data in FITS file")

    # Create x and y indices for each pixel
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')

    # Customize the plot
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_zlabel('Intensity')
    ax.set_title(title)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()
    plt.savefig(output_file)
    plt.clf()



def linear_stretch(image, min_percent=1, max_percent=99):
    # Calculate the histogram of the image
    hist, bin_edges = np.histogram(image, bins=1000)

    # Calculate cumulative distribution function
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]  # Normalize

    # Find the bin edges corresponding to the min and max percentage of data
    min_val = bin_edges[np.searchsorted(cdf, min_percent / 100)]
    max_val = bin_edges[np.searchsorted(cdf, max_percent / 100)]

    # Apply linear stretch
    stretched = (image - min_val) / (max_val - min_val)

    # Clip values to ensure they are between 0 and 1
    return np.clip(stretched, 0, 1)


def plot_fits_image(original_data, stretched_data, output_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Original image
    ax1.imshow(original_data, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Stretched image
    ax2.imshow(stretched_data, cmap='gray')
    ax2.set_title('Linearly Stretched Image')
    ax2.axis('off')

    plt.show()
    plt.savefig(output_file)
    plt.clf()


def update_files_files (file_path, data):
    with fits.open(file_path) as hdu:
        hdu[0].data = data


def compare_fits_files(file1_path, file2_path):
    # Open the FITS files
    with fits.open(file1_path) as hdul1, fits.open(file2_path) as hdul2:

        # Check if both files have at least one HDU with data
        if len(hdul1) > 0 and len(hdul2) > 0:
            data1 = hdul1[0].data
            data2 = hdul2[0].data

            # Check if data shapes match before comparison
            if data1.shape == data2.shape:
                # Calculate the difference
                diff = data1 - data2

                # Calculate Mean Squared Error (MSE)
                mse = np.mean((diff) ** 2)

                # Calculate maximum difference
                max_diff = np.max(np.abs(diff))

                # Check if the images are identical
                if np.all(diff == 0):
                    print("The images are identical.")
                else:
                    print(f"Mean Squared Error: {mse}")
                    print(f"Maximum difference: {max_diff}")
                    print("The images are not identical.")
                    return diff, mse, max_diff

                # Optionally, you can visualize or analyze the differences further
                # For instance, you could plot the difference or compare specific regions
            else:
                print("Error: The dimensions of the image data do not match.")
                return None
        else:
            print("Error: At least one file does not contain image data.")
            return None



def create_graphs ():
    control = '/Users/taylorhogan/Desktop/tt/final.fit'
    exposure_time_min = 5
    files = ['/Users/taylorhogan/Desktop/tt/4.fit', '/Users/taylorhogan/Desktop/tt/8.fit',
             '/Users/taylorhogan/Desktop/tt/16.fit', '/Users/taylorhogan/Desktop/tt/32.fit',
             '/Users/taylorhogan/Desktop/tt/64.fit']
    x = np.array([])
    y = np.array([])
    cur_min = 4

    #first show the user the stretched final image
    fits_file = control
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        data = hdul[0].data  # Assuming data is in the primary HDU

    # Perform linear stretch
    stretched_image = linear_stretch(data)

    # Plot the original and stretched images for comparison
    plot_fits_image(data, stretched_image, "../final_stretch.png")

    fit_to_3d_plot_data(stretched_image, "../final_stretched_3d.png", "Final Streched Data")

    for f in files:
        fcompare = f
        diff, mse, max_diff = compare_fits_files(control, fcompare)
        x = np.append(x, cur_min * exposure_time_min)
        y = np.append(y, mse)
        cur_min = cur_min * 2

    x_new = np.linspace(x.min(), x.max(), 300)  # More points for a smoother curve
    spl = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
    y_smooth = spl(x_new)

    # Create the plot
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('exposure minutes')
    plt.ylabel('Difference from Final')
    plt.title('Exposures vs MSE')

    # Display the plot of difference between final and different exposure times
    plt.show()
    plt.savefig("snrevolution.png")
    plt.clf()


    diff, mse, max_diff = compare_fits_files(control,files[4])
    stretched_data_of_diff = linear_stretch(diff)
    plot_fits_image(diff, stretched_data_of_diff, "../final_stretched_difference.png")

    fit_to_3d_plot_data(diff, "../final_stretched_difference_3d.png", "Difference Between Final and Before Final")
    fit_to_3d_plot_data(stretched_data_of_diff, "../final_stretched_difference_3d.png", "Difference Between Final and Before Final")

if __name__ == '__main__':
    create_graphs()

