# here import numpy and cv
from distutils.spawn import spawn
from posixpath import split
from re import M
import numpy as np
import cv2


def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value


trackbar_value = 0


VIDEO = 'hdg2-2_1_5_3_A.avi'
k1=2
k2=4
k3=2
k4=2
ero=0
dil=1
dil_err=2
Blur1 = 31
Blur2 = 5
perc = 80
# VideoCapture()=Class for video capturing from video files, image sequences or cameras.
cap = cv2.VideoCapture(VIDEO)


HEAT = None
HEATDIV = None
iter = 0
cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0

while True:
    ret, frame = cap.read()

    frame
    if iter == 0:
        # <<<< you werejust saving the first frame and adding nothing to it
        OUT = frame[:, :, 0].astype('float64')*0
        cell = frame[:, :, 0].astype('float')*0
    if ret == False:
        break


    filtered = np.copy(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype('uint8')
    blur1 = cv2.GaussianBlur(filtered,(Blur1,Blur1),3)
    cv2.imshow('blur1 ',blur1)
    blur2 = cv2.GaussianBlur(filtered,(Blur2,Blur2),2)
    cv2.imshow('blur2 ',blur2)
    res = blur1.astype('float')-blur2.astype('float')


    res-=np.amin(res)
    res/=np.amax(res)
    #print(np.percentile(res,perc))
    res = (res>np.percentile(res,perc)).astype("uint8")*255

    #cv2.imshow('heatmap', res)
    
    
    # gray_filtered = cv2.inRange(res, 154, 255)
    # #v2.imshow("gray_filtered", gray_filtered)
    # edges = cv2.Canny(gray_filtered, threshold1=15, threshold2=15)
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1,k1))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    # erosion = cv2.erode(edges, kernel1, iterations=ero)
    # dilation = cv2.dilate(erosion, kernel, iterations=dil)
    # dil_erode = cv2.erode(dilation, kernel1, iterations=dil_err) 
    # kernel_op = np.ones((k3,k3),np.uint8)
    # openning= cv2.morphologyEx(dil_erode,cv2.MORPH_OPEN,kernel_op)
    # kernel_cl = np.ones((k4,k4),np.uint8)
    # closing= cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel_cl)
    #cv2.imshow("output", closing)
    # gray_filtered2 = cv2.inRange(blur, 176, 255)
    # cv2.imshow("gray_filtered2", gray_filtered2)
    # edges2 = cv2.Canny(gray_filtered2, threshold1=15, threshold2=15)
    # kernel12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1,k1))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    # erosion2 = cv2.erode(edges2, kernel12, iterations=ero)
    # dilation2 = cv2.dilate(erosion2, kernel2, iterations=dil)
    # dil_erode2 = cv2.erode(dilation2, kernel12, iterations=dil_err)
    # openning2= cv2.morphologyEx(dil_erode2,cv2.MORPH_OPEN,(k3,k3))
    # closing2= cv2.morphologyEx(openning2,cv2.MORPH_CLOSE,(k4,k4))
    # gray_filtered2 = cv2.inRange(res, 176, 255)
    kel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion1 = cv2.erode(res, kel1, iterations=1)
    kernel_op1 = np.ones((2,2),np.uint8)
    openning1= cv2.morphologyEx(erosion1,cv2.MORPH_OPEN,kernel_op1)
    kernel_cl1 = np.ones((3,3),np.uint8)
    closing1= cv2.morphologyEx(openning1,cv2.MORPH_CLOSE,kernel_cl1)
    #cv2.imshow("gray_filtered", gray_filtered2)
    # edges2 = cv2.Canny(gray_filtered2, threshold1=15, threshold2=15)
    # erosion2 = cv2.erode(edges2, kernel1, iterations=ero)
    # dilation2 = cv2.dilate(erosion2, kernel, iterations=dil)
    # dil_erode2 = cv2.erode(dilation2, kernel1, iterations=dil_err)
    # openning2= cv2.morphologyEx(dil_erode2,cv2.MORPH_OPEN,(k3,k3))
    # closing2= cv2.morphologyEx(openning2,cv2.MORPH_CLOSE,(k4,k4))
    #closing=closing+gray_filtered2
    # closing-=np.amin(closing)
    # closing/=np.amax(closing)
    # closing = (closing>np.percentile(closing.perc)).astype("uint8")/255

   # MASK=closing*closing2
    #cv2.imshow("Masking", closing)

    #MASK = (HEAT==iter)
    #OUT[MASK] = openning[MASK]
    # res+=res
    # #cv2.imshow('cell', cell)
    # closing=closing+closing1
    # res += closing1
    # cv2.imshow('frame', closing1)
    #cv2.imshow('frame', OUT)

    #cv2.imshow('mask', MASK.astype("float64"))
    #cv2.imshow('Row Data',frame)
    #cv2.imshow('surf stack', border)
    #cv2.imshow('frame', projection)
    OUT+=gray
    res+=closing1
    #cv2.imshow("this Is final",res)
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
# for i in cap.NumberOfFrames():

# test= np.max(OUT)*111
# <<<< fixed it so its properly capped, you can also add sth like OUT/np.amax(OUT)*255
cv2.imwrite(VIDEO.replace('.avi', '_combi_modified.png'), ((OUT/iter)).astype("uint8"))
cv2.imwrite(VIDEO.replace('.avi', '_combine_modified.png'), (res).astype("uint8"))
