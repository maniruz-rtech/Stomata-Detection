import cv2
import numpy as np


def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value


trackbar_value = 0

img = cv2.imread("test1.png", 0)
# kernel = np.ones((5, 5), np.uint8)
cv2.namedWindow("Processed Image")
cv2.createTrackbar("Threshold", "Processed Image", 0, 255, updateValue)
cv2.createTrackbar("first erode", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Dilation", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Erode after Dilate", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Kernel_Erosion", "Processed Image",0,20,updateValue)
cv2.createTrackbar("Kernel_Dialation", "Processed Image",0,20,updateValue)
cv2.createTrackbar("kernel_opening", "Processed Image",0,20,updateValue)
cv2.createTrackbar("kernel_closing", "Processed Image",0,20,updateValue)
while True:
    thresh = cv2.getTrackbarPos('Threshold', "Processed Image")
    ero = cv2.getTrackbarPos('first erode', "Processed Image")
    dil = cv2.getTrackbarPos('Dilation', "Processed Image")
    dil_err = cv2.getTrackbarPos("Erode after Dilate", "Processed Image")
    k1 = cv2.getTrackbarPos("Kernel_Erosion", "Processed Image") or 1
    k2 = cv2.getTrackbarPos("Kernel_Dialation", "Processed Image") or 1
    k3= cv2.getTrackbarPos("kernel_opening", "Processed Image") or 1
    k4= cv2.getTrackbarPos("kernel_closing", "Processed Image") or 1
    blur = cv2.GaussianBlur(img, (5, 5), 5)
    
    
    gray_filtered = cv2.inRange(blur, 176, 255,cv2.THRESH_TOZERO)
    #cv2.imshow("gray_filtered", gray_filtered)
    #edges = cv2.Canny(gray_filtered, threshold1=15, threshold2=15)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1,k1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    erosion = cv2.erode(gray_filtered, kernel1, iterations=ero)
    dilation = cv2.dilate(erosion, kernel, iterations=dil)
    dil_erode = cv2.erode(dilation, kernel, iterations=dil_err)
    kernel_op = np.ones((k3,k3),np.uint8)
    openning= cv2.morphologyEx(dil_erode,cv2.MORPH_OPEN,kernel_op)
    kernel_cl = np.ones((k4,k4),np.uint8)
    closing= cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel_cl)
    
    cv2.imshow("Original", img)
    # cv2.imshow("Erosion", erosion)
    # cv2.imshow("Dilation", dilation)
    # cv2.imshow("Erode after Dilate", dil_erode)
    cv2.imshow("Processed Image", closing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
