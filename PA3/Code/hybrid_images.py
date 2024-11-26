"""
Credit: Alyosha Efros
""" 


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr

isGray = True

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def conv3D(image, kernel):
    r = scipy.signal.convolve2d(image[:,:,0], kernel, boundary='symm', mode='same')
    g = scipy.signal.convolve2d(image[:,:,1], kernel, boundary='symm', mode='same')
    b = scipy.signal.convolve2d(image[:,:,2], kernel, boundary='symm', mode='same')
    return np.stack((r,g,b), axis=2)


if __name__ == "__main__":

    imageDir = '../Images/'
    outDir = '../Results/'

    im1_name = 'doomguy.jpeg'
    im2_name = 'masterchief.jpeg'

    # 1. load the images
	
	# Low frequency image
    im1 = plt.imread(imageDir + im1_name) # read the input image
    info = np.iinfo(im1.dtype) # get information about the image type (min max values)
    im1 = im1.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
	# High frequency image
    im2 = plt.imread(imageDir + im2_name) # read the input image
    info = np.iinfo(im2.dtype) # get information about the image type (min max values)
    im2 = im2.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)
    
    if isGray:
        im1_aligned = np.mean(im1_aligned, axis=2)
        im2_aligned = np.mean(im2_aligned, axis=2)
	
    
    # 3. apply filters and combine images
    kernel_size = 15
    sigma = 3.75
    variance = sigma**2
    gauss_const = 1 / (2 * math.pi * variance)

    # make guassian
    gaussian2D = np.empty(shape=(kernel_size,kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian2D[i][j] = gauss_const * math.e ** (-((i - int(kernel_size/2))**2 + (j - int(kernel_size/2))**2) / (2 * variance))
    

    # apply filters
    if isGray:
        im_low = scipy.signal.convolve2d(im1_aligned, gaussian2D, boundary='symm', mode='same')
        im_high = im2_aligned - scipy.signal.convolve2d(im2_aligned, gaussian2D, boundary='symm', mode='same')
    else:
        im_low = conv3D(im1_aligned, gaussian2D)
        im_high = im2_aligned - conv3D(im2_aligned, gaussian2D)

    # combine and normalize
    im = im_low + im_high
    if isGray:
        im = im / im.max()
    else:
        # im[:,:,0] = im[:,:,0] / im[:,:,0].max()
        # im[:,:,1] = im[:,:,1] / im[:,:,1].max()
        # im[:,:,2] = im[:,:,2] / im[:,:,2].max()
        im = np.clip(im, 0.0, 1.0)

    if isGray:
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im, cmap='gray')
    else:
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im)
    
    pass
