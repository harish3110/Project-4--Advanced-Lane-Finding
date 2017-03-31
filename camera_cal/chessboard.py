import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# read in and make a list of calibration images
images = glob.glob('./calibration*.jpg')
print("number of calibration images: {}".format(len(images)))


#arrays to store object points and image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space

#create object points
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9 , 0:6].T.reshape(-1, 2)

for fname in images:
    # read in each image
    img = mpimg.imread(fname)

    # convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # if found add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display chessboard corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.imshow(img)
