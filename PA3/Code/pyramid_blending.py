

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import skimage.transform as sktr
import math

kernel_size = 11
sigma = 2.2
variance = sigma**2
gauss_const = 1 / (2 * math.pi * variance)
gaussian2D = np.empty(shape=(kernel_size,kernel_size))
for i in range(kernel_size):
    for j in range(kernel_size):
        gaussian2D[i][j] = gauss_const * math.e ** (-((i - int(kernel_size/2))**2 + (j - int(kernel_size/2))**2) / (2 * variance))


# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target



def get_gaussian_pyramid(image, levels):
    pyramid = []
    for i in range(levels):
        pyramid.append(image)
        image = sktr.resize(image, output_shape=(image.shape[0]/2, image.shape[1]/2, image.shape[2]))
    return pyramid

def get_laplacian_pyramid(image, levels):
    pyramid = []
    gaussian_pyramid = get_gaussian_pyramid(image, levels) # 0th is largest
    for i in range(levels-1):
        reconstruction = sktr.resize(gaussian_pyramid[i+1], output_shape=gaussian_pyramid[i].shape)
        laplacian = gaussian_pyramid[i] - reconstruction
        pyramid.append(laplacian)
    pyramid.append(gaussian_pyramid[-1]) # last level: gaussian = laplace

    return pyramid

# Pyramid Blend
def PyramidBlend(source, mask, target, levels=8):
    # initialize pyramids
    gm_pyramid = get_gaussian_pyramid(mask, levels)
    ls_pyramid = get_laplacian_pyramid(source, levels)
    lt_pyramid = get_laplacian_pyramid(target, levels)

    # create blended pyramid
    blended_pyramid = []
    for i in range(levels):
        blended_image = ls_pyramid[i] * gm_pyramid[i] + lt_pyramid[i] * (1.0 - gm_pyramid[i])    
        blended_pyramid.append(blended_image)

    # collapse blended pyramid
    blended_pyramid.reverse() # smallest to largest
    final_image = blended_pyramid[0]
    for i in range(1, levels):
        upscaled = sktr.resize(final_image, output_shape=(blended_pyramid[i].shape[0], blended_pyramid[i].shape[1], blended_pyramid[i].shape[2]))
        final_image = blended_pyramid[i] + upscaled

    return np.clip(final_image, 0.0, 1.0)


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 2

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    
    ### The main part of the code ###

    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target, 9)
    

    
    # Writing the result

    plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
