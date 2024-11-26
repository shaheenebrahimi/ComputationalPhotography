# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, linalg

dirs = [[-1,0], [1,0], [0,-1], [0,1]]

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

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Poisson Blend
def PoissonBlend(source, mask, target, isMix):
    # get data
    mask = mask[:,:,0] # get rid of third component
    masked_rows, masked_cols = np.nonzero(mask)
    M, N, _ = source.shape

    # sparse matrix vars
    rows, cols, vals = [], [], []
    pixel_mapping = {}

    def get_mapping(x):
        if x not in pixel_mapping: pixel_mapping[x] = len(pixel_mapping)
        return pixel_mapping[x]
    
    def set_sparse_entry(row, col, val):
        rows.append(row)
        cols.append(col)
        vals.append(val)

    # create A matrix
    for k, (i, j) in enumerate(zip(masked_rows, masked_cols)):
        valid_neighbors = 0
        for dir in dirs:
            i_nei, j_nei = i + dir[0], j + dir[1]
            if i_nei >= 0 and i_nei < M and j_nei >= 0 and j_nei < N: # neighbor inbounds
                valid_neighbors += 1
                if mask[i_nei, j_nei]: # neighbor inside of mask (unknown)
                    set_sparse_entry(k, get_mapping((i_nei, j_nei)), -1)
        set_sparse_entry(k, get_mapping((i, j)), valid_neighbors)
    
    A = csr_matrix((vals, (rows, cols)), shape = (len(pixel_mapping), len(pixel_mapping)))

    # evaluate per channel
    blended_img = target
    for channel in range(3):
        b = [] # create b matrix for channel
        for k, (i, j) in enumerate(zip(masked_rows, masked_cols)):
            eq_b = 0
            for dir in dirs:
                i_nei, j_nei = i + dir[0], j + dir[1]
                if i_nei >= 0 and i_nei < M and j_nei >= 0 and j_nei < N: # neighbor inbounds
                    source_gradient = source[i, j, channel] - source[i_nei, j_nei, channel]
                    target_gradient = target[i, j, channel] - target[i_nei, j_nei, channel]
                    mixed_gradient = target_gradient if abs(target_gradient) > abs(source_gradient) else source_gradient
                    eq_b += mixed_gradient if isMix else source_gradient
                    if not mask[i_nei, j_nei]: # neighbor outside of mask
                        eq_b += target[i_nei, j_nei, channel]
            b.append(eq_b)
        
        # solve linear system
        x = linalg.spsolve(A, np.array(b))

        # blend image
        for (i, j), index in pixel_mapping.items():
            blended_img[i, j, channel] = np.clip(x[index], 0.0, 1.0)

    return blended_img

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Inputs/'
    outputDir = '../Outputs/'
    
    # False for source gradient, true for mixing gradients
    isMix = True

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88], [-50,0], [-35,-40]]

    # main area to specify files and display blended image
    for index in range(8,len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])
        
        ### The main part of the code ###
    
        # Implement the PoissonBlend function
        poissonOutput = PoissonBlend(source, mask, target, isMix)

        
        # Writing the result
                
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
