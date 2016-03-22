import cv2 as cv
import numpy as np
from drawMatches import drawMatches
from functools import partial

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
        img.gray = cv.cvtColor(img.original, cv.COLOR_BGR2GRAY)

        # Need this for images too large otherwise you get Not enough memory errors
        while(img.gray.shape[0] > 1000 or img.gray.shape[1] > 1000):
            img.gray = cv.resize(img.gray, (img.gray.shape[1]/2, img.gray.shape[0]/2) )
        return img

def showImage(img):
        cv.imshow( img.name, img.gray )

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

        src_pts     = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts     = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        H,r         = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        # Careful sizes are inverted in OpenCV !! shape[1] == width, shape[0] == height
        size        = (imgA.gray.shape[1] + imgB.gray.shape[1], imgB.gray.shape[0])
        result      = cv.warpPerspective(imgA.gray, H, size)
        width       = imgB.gray.shape[1]
        height      = imgB.gray.shape[0]
        max_width   = size[0]
        result[ 0:height, 0:width] = imgB.gray

        # Returns the image resulting and computed with corners in case
        # there is more matching to be done
        imageResult = Image()
        imageResult.original = result;
        imageResult.gray = result;
        imageResult = generateCorners(imageResult)
        return imageResult

# Main execution starts here
filenames = [ "rsz_03_camera.jpg" ,"rsz_02_camera.jpg", "rsz_01_camera.jpg" ]
# Create images objcets using createImage function that receives the directory and filename
# Partial application for the directory
files = map( partial(createImage, "images/") , filenames)

# Show all images with showImage function, maps over array
map(showImage, files)

# generate images with keypoints and descriptors information
imagesWithKeypoints = map(generateCorners, files)

# Match features and align images using BFMatcher
# TODO: Improve matcher with other kind of matcher ?
resultImage = reduce( partial(matchFeatures, cv.BFMatcher()) , imagesWithKeypoints)

# Show result image
cv.imshow("Final Image: ", resultImage.gray)

# Waits for ESC to destroy all windows
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
