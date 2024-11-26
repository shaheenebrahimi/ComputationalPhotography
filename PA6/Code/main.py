import numpy as np
import matplotlib.pyplot as plt
import os
import random
from gsolve import gsolve
import skimage

# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Results/'

_lambda = 50

calibSetName = 'Office'

# Parsing the input images to get the file names and corresponding exposure
# values
filePaths, exposures = ParseFiles(calibSetName, inputDir)

""" Task 1 """

# Load images
imgs = [ (plt.imread(path) * 255).astype(int) for path in filePaths ]

height, width, channels = imgs[0].shape

# Sample the images
samples = []
P = len(imgs)
N = int(5 * 256/(P - 1))
for _ in range(N):
    i, j = random.randrange(0, height), random.randrange(0, width)
    sample = [ img[i][j] for img in imgs ] # get pixel for each image
    samples.append(sample)
Z = np.array(samples)

# Log exposure time
B = np.log(np.array(exposures))

# Create the triangle function
Z_min, Z_max = np.min(samples), np.max(samples)
Z_mid = 0.5 * (Z_min + Z_max)
def triangle_function(z):
    return z - Z_min if z <= Z_mid else Z_max - z
w = np.fromiter(map(triangle_function, [i for i in range(256)]), int)

# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
crf = [ gsolve(Z[:,:,c], B, _lambda, w)[0] for c in range(channels) ] # g inverse function for each channel

def plotCRF(x):
    plt.plot([ crf[0][xi] for xi in x ], x, 'r')
    plt.plot([ crf[1][xi] for xi in x ], x, 'g')
    plt.plot([ crf[2][xi] for xi in x ], x, 'b')
    plt.show()

plotCRF([ i for i in range(256) ])

# """ Task 2 """
# Reconstruct the radiance map using the calculated CRF
E = np.zeros(shape=(height,width,channels)) # radiance
for i in range(height):
    for j in range(width):
        for k in range(channels):
            numerator, denominator = [], []
            for p in range(P):
                Zp, g = imgs[p][i,j,k], crf[k]
                numerator.append(w[Zp] * (g[Zp] - B[p]))
                denominator.append(w[Zp])
            numerator, denominator = np.sum(numerator), np.sum(denominator)
            if denominator != 0: 
                E[i,j,k] = np.exp(np.divide(numerator, denominator))
            else:
                E[i,j,k] = 1.0
            # E[i,j,k] = np.exp(np.sum([ w[imgs[p][i,j,k]] * (crf[k][imgs[p][i,j,k]] - B[p]) for p in range(P) ]) \
            #     / np.sum([ w[imgs[p][i,j,k]] for p in range(P) ]))

""" Task 3 """

# Constants
gamma = 0.2
sigma = 1.8
dR = 5

# Perform global tone-mapping
T = (E / np.max(E)) ** gamma

# Perform local tone-mapping
# I = np.mean(E, axis=2) # intensity
# chromiance = E / I[:, :, np.newaxis]

# L = np.log2(I) # log intensity
# B = skimage.filters.gaussian(L, sigma) # base
# D = L - B # detail

# offset = np.max(B)
# scale = dR / (np.max(B) - np.min(B))
# Bp = (B - offset) # apply offset and scale

# base = np.full((height,width), 2.0) 
# exponent = Bp + D
# O = np.power(base, exponent) # reconstruct log intensity

# result = chromiance * O[:, :, np.newaxis] # put back colors
# result = result ** gamma # gamma compression
# result = np.clip(result, 0.0, 1.0) # clip

# Ouput image
plt.imsave("{}/{}.jpg".format(outputDir, calibSetName), T)


