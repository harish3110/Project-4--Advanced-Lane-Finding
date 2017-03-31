import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#create object points
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

#arrays to store object points and image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space

# create a list of calibration images
images = glob.glob('./calibration*.jpg')
print("number of calibration images: {}".format(len(images)))

# step through list and find chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    #plt.imshow(img)
    #exit()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # if found add object points, image points
    if ret == True:
        print('working on', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        write_name = 'corners_found'+ str(idx)+'.jpg'
        cv2.imwrite(write_name, img)

# load image for reference
img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# compute camera calibration given image points and object points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save these results(dist, mtx) for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open('./calibration_pickle.p', 'wb'))
print("Pickle file created with mtx and dist variables saved for later use!")
