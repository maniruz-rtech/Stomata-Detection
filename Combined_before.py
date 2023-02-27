#here import numpy and cv
from distutils.spawn import spawn
from posixpath import split
from re import M
import numpy as np
import cv2
 
VIDEO = 'hdg2-2_1_5_3_A.avi'
DILATION = 3
DILATION_ITER = 4
BLURSIZE = 3
SHIFTING=14
DILATION1=4
#VideoCapture()=Class for video capturing from video files, image sequences or cameras.
cap = cv2.VideoCapture(VIDEO)


HEAT = None
HEATDIV = None
iter = 0
cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0
while True:
    ret, frame = cap.read()
    if iter == 0:
        OUT = frame[:,:,0].astype('uint8')
    if ret == False:
	    break
    filtered= np.copy(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype('uint8')
    #cv2.imshow("showing gray",gray)
    blur = cv2.GaussianBlur(gray,(BLURSIZE,BLURSIZE),5)
    #cv2.imshow("showing blur",blur)
    #binary = blur > 10
    #ret1, th1 = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("showing binary",th1)
    #ret1, blur= blur>cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("showing blur",blur)
    edges = cv2.Canny(blur, threshold1=15, threshold2=15)
    del blur
    #cv2.imshow("showing edges",edges)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATION,DILATION))
    #cv2.imshow("showing kernel",kernel)
    erode_surface=cv2.erode(edges,kernel1, iterations=5)
    #cv2.imshow("showing_erossion",erode_surface)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATION1,DILATION1))
    edges1 = cv2.dilate(erode_surface,kernel,iterations = DILATION_ITER)
    #cv2.imshow("showing_erossion",edges1)
    eroded=cv2.erode(edges1,kernel, iterations=DILATION_ITER)
    #cv2.imshow("showing_erossion",eroded)
    del erode_surface
    openning= cv2.morphologyEx(eroded,cv2.MORPH_OPEN,kernel1)
    #cv2.imshow('frame', openning)
    # R,G,B= split(openning)
    # cv2.imshow("red", R)
    #closing= cv2.morphologyEx(eroded,cv2.MORPH_CLOSE,kernel1)
    #check= openning/closing
    #cv2.imshow('frame', closing)
    #border= edges1 ^ closing
#np.dot(frame,border); frame.dot(border)
    #same error=ValueError: operands could not be broadcast together with shapes (960,1280,3) (960,1280) 
    #MASK= np.dot(np.ones(frame),np.ones(border)).shape
    #MASK=filtered*border
    #masked = frame.dot(border)
    #cv2.imshow(masked)
    MASK = (HEAT==iter)
    OUT[MASK] = openning[MASK]
    #cv2.imshow('frame', OUT)

    #cv2.imshow('mask', MASK.astype("float64"))
    #cv2.imshow('Row Data',frame)
    #cv2.imshow('surf stack', border)
    #cv2.imshow('frame', projection)
    iter += 1
# Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()
#projection = np.max(MASK, axis=0)
# Closes all the frames
cv2.destroyAllWindows()
# R,G,B= split(OUT)
# cv2.imshow("red", R)
# OUT= OUT/R,G,B
#for i in cap.NumberOfFrames():
cv2.imwrite(VIDEO.replace('.avi','_3.png'), ((OUT*255)).astype("uint8"))