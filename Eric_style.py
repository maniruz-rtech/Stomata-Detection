import numpy as np
import cv2

def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value


trackbar_value = 0

cv2.namedWindow("Processed Image")
cv2.createTrackbar("blur1_iteratoin", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("blur2_iteratoin", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("percentile", "Processed Image", 0, 255, updateValue)
cv2.createTrackbar("Kernel_blur1", "Processed Image",0, 50, updateValue)
cv2.createTrackbar("Kernel_blur2", "Processed Image",0, 50, updateValue)
cv2.createTrackbar("kernel_opening", "Processed Image",0,20,updateValue)
cv2.createTrackbar("kernel_closing", "Processed Image",0,20,updateValue)
cv2.createTrackbar("first erode", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Dilation", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Erode after Dilate", "Processed Image", 0, 10, updateValue)
cv2.createTrackbar("Kernel_Erosion", "Processed Image",0,20,updateValue)
cv2.createTrackbar("Kernel_Dialation", "Processed Image",0,20,updateValue)


# Blur1 = 31
# Blur2 = 5
# perc = 80
img = cv2.imread("frame117.png")
while True:
    itr1 = cv2.getTrackbarPos("blur1_iteratoin", "Processed Image")

    itr2 = cv2.getTrackbarPos("blur2_iteratoin", "Processed Image")
    perc = cv2.getTrackbarPos("percentile", "Processed Image")
    Blur1 = cv2.getTrackbarPos("Kernel_blur1", "Processed Image") or 31
    if Blur1%2==0: 
           Blur1+=1 #if Blur1 is an even number this will add 1 with that even number
    Blur2 = cv2.getTrackbarPos("Kernel_blur2", "Processed Image") or 5
    if Blur2%2==0: 
           Blur2+=1 #if Blur1 is an even number this will add 1 with that even number
    ero = cv2.getTrackbarPos('first erode', "Processed Image")
    dil = cv2.getTrackbarPos('Dilation', "Processed Image")
    dil_err = cv2.getTrackbarPos("Erode after Dilate", "Processed Image")
    k1 = cv2.getTrackbarPos("Kernel_Erosion", "Processed Image") or 1
    # if k1%2==0: 
    #        k1+=1 #if Blur1 is an even number this will add 1 with that even number
    k2 = cv2.getTrackbarPos("Kernel_Dialation", "Processed Image") or 1
    # if k2%2==0: 
    #        k2+=1 #if Blur1 is an even number this will add 1 with that even number
    k3= cv2.getTrackbarPos("kernel_opening", "Processed Image") or 1
    # if k3%2==0: 
    #        k3+=1 #if Blur1 is an even number this will add 1 with that even number
    k4= cv2.getTrackbarPos("kernel_closing", "Processed Image") or 1 
    # if k4%2==0: 
    #        k4+=1 #if Blur1 is an even number this will add 1 with that even number
    
    
    blur1 = cv2.GaussianBlur(img,(Blur1,Blur1),itr1)
    cv2.imshow("blur1",blur1)
    blur2 = cv2.GaussianBlur(img,(Blur2,Blur2),itr2)
    cv2.imshow("blur2",blur2)
    res = blur1.astype('float')-blur2.astype('float')


    res-=np.amin(res)
    res/=np.amax(res)
    print(np.percentile(res,perc))
    res = (res>np.percentile(res,perc)).astype("uint8")*255

    cv2.imshow('heatmap', res)
    kernel_op = np.ones((k3,k3),np.uint8)
    openning= cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel_op)
    kernel_cl = np.ones((k4,k4),np.uint8)
    closing= cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel_cl)
    kernel_o = np.ones((k1,k1),np.uint8)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_o)
    kernel_c = np.ones((k2,k2),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_c)
    erosion = cv2.erode(closing, kernel_o, iterations=ero)
    dilation = cv2.dilate(erosion, kernel_c, iterations=dil)
    dil_erode = cv2.erode(dilation, kernel_o, iterations=dil_err)
    _ , gray_filtered = cv2.threshold(dil_erode, 0, 255,cv2.THRESH_BINARY_INV)
    #gray_filtered= cv2.adaptiveThreshold(dil_erode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)


    #cv2.imshow("Original", img)
    # cv2.imshow("Erosion", erosion)
    # cv2.imshow("Dilation", dilation)
    # cv2.imshow("Erode after Dilate", dil_erode)
    #OUT += gray_filtered
    #out1=OUT
    OUT= (gray_filtered+res)/0.5
    
    #OUT= (OUT+res)/0.5
    OUT/= np.amax(res)
    #OUT+=out1
    #print("shape and lyer",OUT.shape)
    cv2.imshow("Processed Image", OUT)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()