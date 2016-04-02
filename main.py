import cv2 as cv
import numpy as np
from drawMatches import drawMatches
from functools import partial
import sys

# Base class to hold information of the image and features
class Image:
        name = None
        original = None
        gray = None
        keypoints = None
        descriptors = None

def createImage(directory,filename):
        img = Image()
        img.name = filename
        img.original = cv.imread(directory+filename)

        maxSize = 750
        # Need this for images too large otherwise you get Not enough memory errors
        while(img.original.shape[0] > 500 or img.original.shape[1] > 500):
            img.original     = cv.resize(img.original, (img.original.shape[1]/2, img.original.shape[0]/2) )

        img.gray = cv.cvtColor(img.original, cv.COLOR_BGR2GRAY)
        return img

def showImage(img):
        cv.imshow( img.name, img.original )

def generateCorners(img):
        sift = cv.SIFT()
        kp,des = sift.detectAndCompute(img.gray,None)
        img.keypoints = kp
        img.descriptors = des
        return img

def matchFeatures(bf,imgA, imgB):
        matches     = bf.knnMatch(imgA.descriptors, imgB.descriptors, k=2)
        goodMatches = []
        kp1         = imgA.keypoints
        kp2         = imgB.keypoints

        # Select goodMatchs based on this match criteria
        for m,n in matches:
                if(m.distance < 0.75*n.distance):
                        goodMatches.append(m)

        goodMatches = goodMatches[0:20]
        src_pts     = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts     = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        H,r         = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        # Careful sizes are inverted in OpenCV !! shape[1] == width, shape[0] == height
        size        = (imgA.gray.shape[1] + imgB.gray.shape[1], imgB.gray.shape[0] + imgA.gray.shape[0] )
        result      = cv.warpPerspective(imgA.original, H, size)
        width       = imgB.gray.shape[1]
        height      = imgB.gray.shape[0]
        max_width   = size[0]
        #beta        = 5.0

        #resultCopy  =  result.copy()
        #resultCopy[:0] = 0
        result[ 0:height, 0:width] = imgB.original
        #result = cv.addWeighted(result, 0.7, resultCopy, 0.5, 0.0)

        # Returns the image resulting and computed with corners in case
        # there is more matching to be done
        imageResult = Image()
        imageResult.original = result
        imageResult.gray = cv.cvtColor( result, cv.COLOR_BGR2GRAY)  
        imageResult.gray = cv.equalizeHist(imageResult.gray)
        imageResult = generateCorners(imageResult)
        imageResult.name = "(" + imgA.name + "," +imgB.name + ")"
        return imageResult

# Main execution starts here
cases = {
    "0": [ "rsz_03_camera.jpg" ,"rsz_02_camera.jpg", "rsz_01_camera.jpg" ],
    "1": [ "cel_03.jpg" ,"cel_02.jpg", "cel_01.jpg" ],
    "2": [ "02.jpg" ,"01.jpg" ]
}

# Code only to test faster using command line parameters
if( len(sys.argv) > 1):
    filenames = cases.get(sys.argv[1], None)
    if( filenames == None ):
            filenames = cases.get("0", None)
else:
    filenames = cases.get("0", None)

# Create images objcets using createImage function that receives the directory and filename
# Partial application for the directory
files = map( partial(createImage, "images/") , filenames)

# Show all images with showImage function, maps over array
#map(showImage, files)

# generate images with keypoints and descriptors information
imagesWithKeypoints = map(generateCorners, files)

# Match features and align images using BFMatcher
# TODO: Improve matcher with other kind of matcher ?
resultImage = reduce( partial(matchFeatures, cv.BFMatcher()) , imagesWithKeypoints)

# Show result image
showImage( resultImage )

# Waits for ESC to destroy all windows
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
