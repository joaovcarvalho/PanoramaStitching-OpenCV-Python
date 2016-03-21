import cv2 as cv
import numpy as np
from functools import partial

orb = cv.features2d.ORB()

def concatDir(string):
        return "images/"+string

def createTupleImgName(filename):
        return (filename, cv.imread(filename))

def showImage(tup):
        cv.imshow( tup[0], tup[1] )

def generateCorners(tup):
        gray = cv.cvtColor(tup[1], cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray,2,3,0.04)
        dst = cv.dilate(dst,None)
        tup[1][dst>0.01*dst.max()]=[0,0,255]
        showImage( ("Result " + tup[0], tup[1] ) )

filenames = [ "01.jpg", "02.jpg" ]
filenames = map( concatDir, filenames)
files = map( createTupleImgName, filenames)

map(showImage, files)
map(generateCorners, files)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
