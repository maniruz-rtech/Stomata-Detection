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
while True:
    thresh = cv2.getTrackbarPos('Threshold', "Processed Image")

    blur = cv2.GaussianBlur(img, (5, 5), 5)
    
    
    gray_filtered = cv2.inRange(blur, thresh, 255,cv2.THRESH_TOZERO)

    
    cv2.imshow("Original", img)
    # cv2.imshow("Erosion", erosion)
    # cv2.imshow("Dilation", dilation)
    # cv2.imshow("Erode after Dilate", dil_erode)
    cv2.imshow("Processed Image", gray_filtered)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
