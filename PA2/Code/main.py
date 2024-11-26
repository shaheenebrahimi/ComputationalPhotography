# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import skimage.feature
from math import ceil

# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path) # read the input image
    info = np.iinfo(image.dtype) # get information about the image type (min max values)
    image = image.astype(float) / info.max # normalize the image into range 0 and 1

    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535
    
    # Separating the glass image into R, G, and B channels
    b = image[: height, :]
    g = image[height: 2 * height, :]
    r = image[2 * height: 3 * height, :]
    return r, g, b

# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0) # y shift
    shifted = np.roll(shifted, shift[1], axis = 1) # x shift
    return shifted

def apply_filter(im, sigma, type='canny'):
    if type == 'canny':
        return skimage.feature.canny(
            image=im,
            sigma=sigma,
            low_threshold=0.5,
            high_threshold=1.0,
        )
    elif type == 'gaussian':
        return skimage.filters.gaussian(
            image=im,
            sigma=sigma
        )
    else:
        print("Unknown Filter")

def shrink_bounds(y_bounds, x_bounds, border, samples=1, axis=1):
    # shrink along axis: 0 is y, 1 is x

    tol = 0.05 * min(y_bounds[1] - y_bounds[0], x_bounds[1] - x_bounds[0])
    rows, cols = np.where(border > 0)
    rows = rows[np.where(rows > y_bounds[0]) and np.where(rows < y_bounds[1])]
    cols = cols[np.where(cols > x_bounds[0]) and np.where(rows < x_bounds[1])]
    dir0 = cols if axis == 0 else rows
    dir1 = rows if axis == 0 else cols

    sampled = np.random.choice(dir0, size=samples, replace=False)
    minBound = y_bounds[0] if axis == 0 else x_bounds[0]
    maxBound = y_bounds[1] if axis == 0 else x_bounds[1]
    for i in sampled:
        ind = np.where(dir0 == i)
        for j in dir1[ind]:
            if j - minBound < tol:
                minBound = j
            else:
                maxBound = j
    
    if axis == 0:
        y_bounds = (minBound, maxBound)
    else:
        x_bounds = (minBound, maxBound)

    return y_bounds, x_bounds

def get_image_bounds(im):
    y_bounds, x_bounds = (0,im.shape[0]), (0,im.shape[1])

    # Filter out white border
    border = apply_filter(im, 2.6) # harsh filter for hard edges
    y_bounds, x_bounds = shrink_bounds(y_bounds, x_bounds, border, samples=3, axis=1)
    y_bounds, x_bounds = shrink_bounds(y_bounds, x_bounds, border, samples=3, axis=0)

    # Shrink more for black border
    shrink_factor = ((int)((y_bounds[1] - y_bounds[0]) * 0.025), (int)((x_bounds[1] - x_bounds[0]) * 0.025))
    y_bounds = (y_bounds[0] + shrink_factor[0], y_bounds[1] - shrink_factor[0])
    x_bounds = (x_bounds[0] + shrink_factor[1], x_bounds[1] - shrink_factor[1])

    return y_bounds, x_bounds

def crop_image(im, y_bounds, x_bounds):
    return im[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]

def edge_detection(im):
    edges = skimage.feature.canny(
        image=im,
        sigma=1.0,
        low_threshold=0.1,
        high_threshold=0.2,
    )
    return edges.astype(float)

def get_downscale_list(im, levels=4):
    images = []
    scale = 1 / (1 << levels-1)
    for i in range(1, levels): 
        downscaled = skimage.transform.resize(im, (im.shape[0] * scale, im.shape[1] * scale))
        images.append(downscaled)
        scale *= 2
    return images + [im]

def pyramid_shift(im1, im2, levels=4):
    im1_list = get_downscale_list(im1, levels)
    im2_list = get_downscale_list(im2, levels)
    shift = find_shift(im1_list[0], im2_list[0])
    for i in range(1, levels):
        y_range = (2 * shift[0] - 3, 2 * shift[0] + 3)
        x_range = (2 * shift[1] - 3, 2 * shift[1] + 3)
        shift = find_shift(im1_list[i], im2_list[i], y_range, x_range)
    return shift

# Compute vertical and horizontal shift to minimize SSD error
def find_shift(im1, im2, y_bounds=(-30,30), x_bounds=(-30,30)):
    best_shift = (0, 0)
    min_ssd = float('inf')
    for y in range(int(y_bounds[0]), int(y_bounds[1])): # iter over vert shifts
        for x in range(int(x_bounds[0]), int(x_bounds[1])): # iter over horiz shifts
            shifted_im1 = circ_shift(im1, (y, x)) # shift im1
            ssd = np.sum((shifted_im1 - im2) ** 2) # compute SSD
            if ssd < min_ssd:
                min_ssd = ssd
                best_shift = (y, x) # y offset, x offset
    return best_shift

def adjust_contrast(im):
    min_pixel = np.min(im)
    max_pixel = np.max(im)
    return (im - min_pixel) / (max_pixel - min_pixel)


if __name__ == '__main__':
    # Setting the input output file path
    imageDir = '../Images/'
    imageName = 'turkmen.tif'
    outDir = '../Outputs/'
    
    # Get r, g, b channels from image strip
    r, g, b = read_strip(imageDir + imageName)

    # Remove Border
    y_bounds, x_bounds = get_image_bounds(b)
    y_bounds = (y_bounds[0], y_bounds[1])
    x_bounds = (x_bounds[0], x_bounds[1])
    print("Cropped bounds: ", y_bounds, x_bounds)
    r_cropped = crop_image(r, y_bounds, x_bounds)
    g_cropped = crop_image(g, y_bounds, x_bounds)
    b_cropped = crop_image(b, y_bounds, x_bounds)

    # Task 1: Brute force
    # rShift = find_shift(r_cropped, b_cropped)
    # gShift = find_shift(g_cropped, b_cropped)
    # print("Red Channel Shift:", rShift)
    # print("Green Channel Shift:", gShift)

    # Task 2: Image Pyramid
    # rShift = pyramid_shift(r_cropped, b_cropped)
    # gShift = pyramid_shift(g_cropped, b_cropped)
    # print("Red Channel Shift:", rShift)
    # print("Green Channel Shift:", gShift)

    # EC: Better features
    edges = edge_detection(b_cropped)
    rShift = pyramid_shift(edge_detection(r_cropped), edges)
    gShift = pyramid_shift(edge_detection(g_cropped), edges)
    print("Red Channel Shift:", rShift)
    print("Green Channel Shift:", gShift)

    finalB = b
    finalG = circ_shift(g, gShift)
    finalR = circ_shift(r, rShift)


    # Putting together the aligned channels to form the color image
    finalImage = np.stack((finalR, finalG, finalB), axis = 2)

    # EC: Adjust Contrast
    # finalImage = adjust_contrast(finalImage)

    # Writing the image to the Results folder
    plt.imsave(outDir + imageName[:-4] + '.jpg', finalImage)
