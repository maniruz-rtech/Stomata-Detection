
# Python code for Background subtraction using OpenCV
import numpy as np
import cv2
iter = 0
OUT = None
VIDEO = 'hdg2-2_1_5_3_A.avi'
cap = cv2.VideoCapture(VIDEO)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.createBackgroundSubtractorKNN()
#this doesn't support to image processing
#fgbg = cv2.createBackgroundSubtractorGMG()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#fgbg = bgs.SuBSENSE()
while True:
    ret, frame = cap.read()
    if iter == 0:
        # <<<< you werejust saving the first frame and adding nothing to it
        OUT = frame[:, :, 0].astype('float64')*0
        cell = frame[:, :, 0].astype('float64')*0
    if ret == False:
        break
    #filtered = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
################for stoma################
    gray_filtered = cv2.inRange(gray, 156, 255)
    #gray_filtered = cv2.inRange(blur, 176, 255)
    fgmask = fgbg.apply(gray_filtered)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cla1=clahe.apply(fgmask)
    blur = cv2.GaussianBlur(cla1, (5, 5), 5)
    edges = cv2.Canny(blur, 100,200, 1, L2gradient=True)
################for cell edges#############
    gray_filtered2 = cv2.inRange(gray, 176, 255)
    #gray_filtered = cv2.inRange(blur, 176, 255)
    fgmask2 = fgbg.apply(gray_filtered2)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cla2=clahe.apply(fgmask2)
    blur2 = cv2.GaussianBlur(cla2, (5, 5), 5)
    edges2 = cv2.Canny(blur2, 100,200, 1, L2gradient=True)
    cv2.imshow("this 2nd edges",edges2)
#################combine output############
    edges+=edges2
    #cv2.imshow("this this combination",edges)
    
    cv2.rectangle(gray_filtered, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(gray_filtered, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    OUT += edges
    cv2.imshow('fgmask', blur)
    cv2.imshow('frame',gray_filtered )
    iter += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
cv2.imwrite(VIDEO.replace('.avi','yoo_3.png'), ((OUT*255/iter)).astype("uint8"))
