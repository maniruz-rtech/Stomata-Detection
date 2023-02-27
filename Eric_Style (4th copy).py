# here import numpy and cv
from distutils.spawn import spawn
from posixpath import split
import numpy as np
import cv2


def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value


trackbar_value = 0


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


HEAT = None
HEATDIV = None
iter = 0
cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0
cv2.namedWindow("Processed Image")
cv2.createTrackbar("Erosion", "Processed Image", 0, 10, updateValue)

while True:
    ret, frame = cap.read()

    frame
    if iter == 0:
        # <<<< you werejust saving the first frame and adding nothing to it
        OUT = frame[:, :, 0].astype('float32')*0
        cv2.imshow("first out", OUT)
        cell = frame[:, :, 0].astype('float64')*0
    if ret == False:
        break


    filtered = np.copy(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype('uint8')
    blur1 = cv2.GaussianBlur(gray,(Blur1,Blur1),5)
    #cv2.imshow("blur1",blur1)
    blur2 = cv2.GaussianBlur(gray,(Blur2,Blur2),1)
    # cv2.imshow("blur2",blur2)
    res = blur1.astype('float')-blur2.astype('float')


    res-=np.amin(res)
    res/=np.amax(res)
    #print(np.percentile(res,perc))
    res = (res>np.percentile(res,perc)).astype("uint8")*255

    #cv2.imshow('heatmap', res)
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
    
    #_ , gray_filtered = cv2.threshold(dil_erode, 0, 255,cv2.THRESH_BINARY_INV)
    #gray_filtered= cv2.adaptiveThreshold(dil_erode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    gray_filtered= cv2.adaptiveThreshold(dil_erode,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
    OUT += gray_filtered/np.amax(res)
    cv2.imshow("after first mask",OUT)
    #out1=OUT
    #OUT= (res+OUT)/0.5
    
    #OUT= (OUT+res)/0.5
    #OUT/= np.amax(res)
    #OUT+=out1
    #print("shape and lyer",OUT.shape)
    cv2.imshow('frame', OUT)
    iter += 1
    #print ("frame Number:", iter)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cv2.imwrite(VIDEO.replace('.avi', 'Eric_Style (4th copy).png'), ((OUT*255/iter)).astype("float"))
