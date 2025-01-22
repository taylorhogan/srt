

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_to_3d_plot(fits_path):
    # Open the FITS file
    with fits.open(fits_path) as hdul:
        # Assume the data is in the primary HDU for simplicity
        data = hdul[0].data

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
    ax.set_title('3D Plot of FITS Image Data')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()


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


def plot_fits_image(original_data, stretched_data):
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

                # Optionally, you can visualize or analyze the differences further
                # For instance, you could plot the difference or compare specific regions
            else:
                print("Error: The dimensions of the image data do not match.")
        else:
            print("Error: At least one file does not contain image data.")


# Example usage
file1 = '/Users/taylorhogan/Desktop/tt/final.fit'
file2 = '/Users/taylorhogan/Desktop/tt/64.fit'
compare_fits_files(file1, file2)
file2 = '/Users/taylorhogan/Desktop/tt/32.fit'
compare_fits_files(file1, file2)
file2 = '/Users/taylorhogan/Desktop/tt/16.fit'
compare_fits_files(file1, file2)
file2 = '/Users/taylorhogan/Desktop/tt/8.fit'
compare_fits_files(file1, file2)
file2 = '/Users/taylorhogan/Desktop/tt/4.fit'
compare_fits_files(file1, file2)
x = [128,64,32,16,8,4]
y = [0,0.0002190212981076911, 0.0007361714378930628,0.000737244903575629,0.0007360181771218777,0.0006927199428901076]

# Create the plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('Number of exposures')
plt.ylabel('Difference from Final')
plt.title('Exposures vs MSE')

# Display the plot
plt.show()

fit_to_3d_plot(file2)

# Path to your FITS file
fits_file = file1

# Open the FITS file
with fits.open(fits_file) as hdul:
    data = hdul[0].data  # Assuming data is in the primary HDU

# Perform linear stretch
stretched_image = linear_stretch(data)

# Plot the original and stretched images for comparison
plot_fits_image(data, stretched_image)