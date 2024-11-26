# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from skimage import color

# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def SeamCarve(input, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.
    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    # Helper functions
    dirs = [[-1,-1], [-1,0], [-1,1]]
    def getMinParent(M, i, j):
        min_parent = np.inf
        min_index = -1
        for dir in dirs: # check all 3 parents for min path
            m_parent, n_parent = i + dir[0], j + dir[1]
            if m_parent >= 0 and m_parent < m and n_parent >= 0 and n_parent < n and M[m_parent][n_parent] < min_parent:
                min_parent = M[m_parent][n_parent]
                min_index = n_parent
        return min_parent, min_index
    
    def rotate_images(img, mask, cw=False):
        return (np.rot90(img,3), np.rot90(mask, 3)) if cw else (np.rot90(img), np.rot90(mask))

    # Compute seam energies
    is_vertical = widthFac < 1
    width, height = input.shape[1], input.shape[0]
    width_prime, height_prime = int(widthFac * width), int(heightFac * height)
    iterations = width - width_prime if is_vertical else height - height_prime
    # Rotate 90 degrees if horizontal seams
    output, mask = (input, mask) if is_vertical else rotate_images(input, mask)

    for seam in range(iterations):
        print('seam', seam)
        # Convert to grayscale
        img = color.rgb2gray(output)
        
        # Compute energy function (increase for mask)
        gy, gx = np.gradient(img)
        # E = np.abs(gx) + np.abs(gy) + mask * 100.0
        E = np.abs(gx) + np.abs(gy)
       
        # Create scoring matrix
        m, n, c = output.shape
        M = E.copy() # M[i,j] = E[i,j] + path
        for i in range(m):
            for j in range(n):
                if i > 0:
                    parent, _ = getMinParent(M, i, j)
                    assert(not np.isinf(parent))
                    M[i][j] += parent

        # Find optimal seam
        indices = [] # to remove per row
        j = np.argmin(M[-1]) # lowest energy seam
        for i in range(m-1,-1,-1): # trace up
            indices.append(j)
            _, j = getMinParent(M, i, j) # update j
        indices.reverse() # set 0 index first

        # Remove seam
        indicies = np.arange(n) != np.array(indices)[:,None] # reconifigure as mask
        output = output[indicies].reshape(m,-1,c)
        mask = mask[indicies].reshape(m,-1)

    # Rotate 90 degrees if horizontal seams
    if not is_vertical: output, mask = rotate_images(output, mask, cw=True)
    return output, (width_prime, height_prime)


# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 0.5; # To reduce the width, set this parameter to a value less than 1
heightFac = 1;  # To reduce the height, set this parameter to a value less than 1
N = 4 # number of images

for index in range(N, N + 1):

    input, mask = Read(str(index).zfill(2), inputDir)

    # Performing seam carving. This is the part that you have to implement.
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    # Writing the result
    plt.imsave("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)