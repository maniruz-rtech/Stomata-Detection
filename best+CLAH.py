# here import numpy and cv
from distutils.spawn import spawn
from posixpath import split
from re import M
import numpy as np
import cv2
fgbg = cv2.createBackgroundSubtractorMOG2()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
VIDEO = 'hdg2-2_1_5_3_A.avi'
Blur1 = 11
Blur2 = 5
perc = 11
k1=2
k2=2
k3=3
k4=2
ero=1
dil=1
dil_err=1

# VideoCapture()=Class for video capturing from video files, image sequences or cameras.
cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0

while True:
    ret, frame = cap.read()

    frame
    if iter == 0:
        # <<<< you werejust saving the first frame and adding nothing to it
        OUT = frame[:, :, 0].astype('float32')*0
        cell = frame[:, :, 0].astype('float64')*0
    if ret == False:
        break


    filtered = np.copy(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype('uint8')


    blur1 = cv2.GaussianBlur(gray,(Blur1,Blur1),5)
    #cv2.imshow("blur1",blur1)
    blur2 = cv2.GaussianBlur(gray,(Blur2,Blur2),1)
    # cv2.imshow("blur2",blur2)
    res1 = blur1.astype('float')-blur2.astype('float')
    res = blur1.astype('float')-blur2.astype('float')


    res-=np.amin(res)
    res/=np.amax(res)
    #print(np.percentile(res,perc))
    res = (res>np.percentile(res,perc)).astype("uint8")*255
    res = clahe.apply(res)
    ret, res = cv2.threshold(res, 156, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('heatmap', res)
    kernel_op = np.ones((k3,k3),np.uint8)
    openning= cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel_op)
    kernel_cl = np.ones((k4,k4),np.uint8)
    closing= cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel_cl)

    kernel_e = np.ones((k1,k1),np.uint8)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_o)
    kernel_d = np.ones((k2,k2),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_c)
    erosion = cv2.erode(closing, kernel_e, iterations=ero)
    dilation = cv2.dilate(erosion, kernel_d, iterations=dil)
    dil_erode = cv2.erode(dilation, kernel_e, iterations=dil_err) 
    #gray_filtered= cv2.adaptiveThreshold(dil_erode,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
    #OUT += gray_filtered /np.amax(res1)
    #dil_erode=fgbg.apply(dil_erode)
    OUT += dil_erode
    cv2.imshow('frame', OUT)
    iter += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cv2.imwrite(VIDEO.replace('.avi', 'best_clah.png'), ((OUT)/iter).astype("uint8"))
