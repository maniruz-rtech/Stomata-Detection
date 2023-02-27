#!/usr/bin/env python3
#here import numpy and cv
import numpy as np
import cv2
import tifffile
from scipy.ndimage import filters
from datetime import datetime
from skimage.morphology import binary_erosion, binary_dilation, ball
 
VIDEO = 'hdg2-2_1_5_3_A.avi'
DILATION = 5
DILATION_ITER = 5
BLURSIZE = 101
#VideoCapture()=Class for video capturing from video files, image sequences or cameras.
cap = cv2.VideoCapture(VIDEO)


HEAT = None
HEATDIV = None
iter = 0
# Check if video/camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  # if frame is read correctly ret is True
  #Our operations on the frame come here inside the frame
    ret, frame = cap.read()
    if iter == 0:
        '''Draw a heatmap overlay over an image.'''
        HEAT = np.zeros_like(frame[:,:,0]).astype('uint8')
        HEATDIV = np.zeros_like(frame[:,:,0]).astype('uint8')
    if ret == False:
	    break
    print("Converting to 8 bit")
    frame = frame.astype(np.uint8)
    filtered= np.copy(frame)
    for idx,plane in enumerate(filtered):
        filtered[idx] = cv2.GaussianBlur(plane,(3,3),0)
    #binary= filtered>cv2.threshold(filtered,20, 255,cv2.THRESH_BINARY_INV)
    # del filtered
    print("detecting Edges")
    edges = cv2.Canny(filtered, threshold1=20, threshold2=20)
   
    print("shifting binary object down")
    shift_mag = int(20+(4/2))
    down_shift = edges[0:0:-shift_mag]
    padding = np.zeros((shift_mag, edges.shape[0], edges.shape[0]),dtype=np.uint8)
    down_shift = np.append(padding, down_shift, axis=0)
#has tried with np.r_, np.c_, concatenate, also changed to axix to 1
    print("Shifting binary object up")
    shift_mag = int(20+(4/2))
    up_shift = edges[0:0:-shift_mag]
    padding = np.zeros((shift_mag, edges.shape[1], edges.shape[2]))
    up_shift = np.append(padding, up_shift, axis=0)
    del edges
    
    print("Generating mask")
    mask = up_shift - down_shift
    del up_shift
    del down_shift
    mask = mask > 0

    print("Masking frame")
    masked = frame * mask
         
    print("Projecting data")
    projection = np.max(masked, axis=0)   
    
    print("Opening viewer")
    cv2.imshow(frame, masked, projection)
    cv2.imshow('frame', frame)
	
    iter += 1	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.imwrite(VIDEO.replace('.avi','stomata.png'), (HEAT/np.max(HEAT)*255).astype("uint8"))

cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0
while True:
    ret, frame = cap.read()
    if iter == 0:
        OUT = frame[:,:,0].astype('float64')/255
    if ret == False:
	    break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float64')/255
    MASK = (HEAT==iter)
    OUT[MASK] = gray[MASK]
	
    #cv2.imshow('mask', MASK.astype("float64"))
    cv2.imshow('frame', OUT)
    iter += 1
# Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


cv2.imwrite(VIDEO.replace('.avi','.png'), (OUT*255).astype("uint8"))
