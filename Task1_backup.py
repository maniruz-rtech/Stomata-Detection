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
    
    
    gray_filtered = cv2.inRange(blur, thresh, 255)
    gray_filtered1 = cv2.inRange(gray_filtered, 0, 155)
    #cv2.imshow("gray_filtered", gray_filtered)
    #edges = cv2.Canny(gray_filtered, threshold1=15, threshold2=15)
    edges = cv2.Canny(gray_filtered1, 50, 100, 5, L2gradient = True)
    cv2.imshow("edges", edges)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_NONE)
    #print("this are", contours)

    drawing_img = np.zeros_like(edges)
    cv2.drawContours(drawing_img, contours, 0, (255), 1)
    # type = int thinningType = THINNING_ZHANGSUEN
    # thin = cv2.threshold(edges, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    #thin = cv2.ximgproc.thinning(edges)
    cv2.imshow("thined", drawing_img)

    kernel_o = np.ones((k1,k1),np.uint8)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_o)
    kernel_c = np.ones((k2,k2),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_c)
    erosion = cv2.erode(edges, kernel_o, iterations=ero)
    dilation = cv2.dilate(erosion, kernel_c, iterations=dil)
    dil_erode = cv2.erode(dilation, kernel_o, iterations=dil_err)
    kernel_op = np.ones((k3,k3),np.uint8)
    openning= cv2.morphologyEx(dil_erode,cv2.MORPH_OPEN,kernel_op)
    kernel_cl = np.ones((k4,k4),np.uint8)
    closing= cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel_cl)
    
    # gray_filtered2 = cv2.inRange(blur, 176, 255)
    # gray_filtered3 = cv2.inRange(gray_filtered2, 0, 155)
    # cv2.imshow("gray_filtered", gray_filtered3)
    # edges2 = cv2.Canny(gray_filtered2, threshold1=15, threshold2=15)
    # erosion2 = cv2.erode(edges2, kernel1, iterations=ero)
    # dilation2 = cv2.dilate(erosion2, kernel, iterations=dil)
    # dil_erode2 = cv2.erode(dilation2, kernel1, iterations=dil_err)
    # openning2= cv2.morphologyEx(dil_erode2,cv2.MORPH_OPEN,(k3,k3))
    # closing2= cv2.morphologyEx(openning2,cv2.MORPH_CLOSE,(k4,k4))
    #border= dilation ^ dil_erode
    #maske1= img * border
    # art=closing+ gray_filtered2
    #cv2.imshow("interfear", art)
    cv2.imshow("Original", img)
    # cv2.imshow("Erosion", erosion)
    # cv2.imshow("Dilation", dilation)
    # cv2.imshow("Erode after Dilate", dil_erode)
    cv2.imshow("Processed Image", closing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
