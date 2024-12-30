import numpy as np 
import cv2 as cv 
import glob 
import os 
import matplotlib.pylab as plt 

import numpy as np
import cv2 as cv
import os
import glob

def calibrate(showPics=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'my_chessboards') #chessboards
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    #of chessboard
    nRows = 9
    nCols = 6

    #termination criteria 
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #ESP = epsilon = algorithm stops when the error is less than the specified threshold (0.001)
    #Useful in refining the corner points in cv.cornerSubPix

    #MAX ITERations = 30

    #world points current
    worldPtsCur = np.zeros((nRows*nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    #find corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            #cv.cornerSubPix(image, corners, winSize, zeroZone, criteria)

            #winSize = The size of the search window for each corner's refinement, expressed as (width, height).
            #in this case, a window of 11x11 pixels is used around each corner to refine its position.

            #zeroZone = A parameter that defines a region around the corner where no smoothing occurs.
            #the value (-1, -1) disables this feature, meaning no exclusion zone is applied.
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11,11), (-1, -1), termCriteria)
            
            imgPtsList.append(cornersRefined)

            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow('chessboard', imgBGR)
                cv.waitKey(1000)
    cv.destroyAllWindows()

    #calibrate 
    #reprojection error, camera matrix, distortion coefficient, rottation vectors, translation vectors
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera Matrix: \n', camMatrix)
    print('Distortion Coeff: \n', distCoeff)
    print('Reproj Error (pixels): {:.4f}'.format(repError))

    #save calibration parameters 
    curFolder = os.path.dirname(os.path.abspath(__file__))

    #the NPY file format is a binary data format specifically designed to store NumPy array objects. 
    #it’s optimized for efficient storage and retrieval of large numerical datasets.
    paramPath = os.path.join(curFolder, 'calibration.npz')

    np.savez(paramPath,
             repError = repError, 
             camMatrix = camMatrix,
             distCoeff = distCoeff,
             rvecs = rvecs,
             tvecs = tvecs)
    return camMatrix, distCoeff

def removeDistortion(camMatrix, distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root, 'chessboards/left12.jpg')
    img = cv.imread(imgPath)
    height, width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))
    imgUndist = cv.undistort(img, camMatrix, distCoeff, None, camMatrixNew)

    #sense changes 
    cv.line(img, (500, 300), (200, 300), (255, 0, 0), 2)
    cv.line(imgUndist, (500, 300), (200, 300), (255, 0, 0), 2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runRemoveDistortion():
    camMatrix, distCoeff = calibrate()
    removeDistortion(camMatrix, distCoeff)

if __name__ == '__main__':
    runRemoveDistortion()
