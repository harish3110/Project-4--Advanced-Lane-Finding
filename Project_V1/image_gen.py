import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
#from tracker import tracker

# read in saved imgpoints and objpoints form pickle file
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb")) # load an read pickled data for variables mtx, dist
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient= 'x', sobel_kernel = 3, thresh = (0,255)):
    ## calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel = 3, mag_thresh = (0,255)):
    ## calculate gradient magnitude
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    ## calculate direction threshold
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(dir_grad)
    # apply threshold
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary_output

def color_threshold(image, sthresh = (0, 255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

#def window_mask(width, height, img_ref, center, level):
    #output = np.zeros_like(img_ref)
    #output[int(img_ref.shape[0])]

# make a list of test images
images = glob.glob('./test_images/test*.jpg')
print("number of test images: {}".format(len(images)))
#exit()

for idx, fname in enumerate(images):
    print('working on', fname)
    # read in image
    img = cv2.imread(fname)
    # undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixel of interest
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255)) #12
    grady = abs_sobel_thresh(img, orient='y', thresh=(35,255)) #25
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255

    # work on defining perspective transform area
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76 # percent of bottom trapezoid height
    mid_width = .08 # percent of middle trapezoid height
    height_pct = .62 # percent of trapezoid height
    bottom_trim = .935 # percent from top to bottom to avoid car hood
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25 # 25 percent offset, can be increased for steeper turns but downside is increase in noise
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    #result = img # to check undistortion effects
    #result = preprocessImage # to check thresholds
    result = warped # to check warped perspective transform

    write_name = './test_images/tracked'+ str(idx+1)+'.jpg'
    cv2.imwrite(write_name, result)
