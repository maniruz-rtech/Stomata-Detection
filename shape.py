
import numpy as np
import imutils
import argparse
import cv2


class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape


img = cv2.imread("test1.png", 0)
while True:
    resized = imutils.resize(frame, width=300)
    ratio = frame.shape[0] / float(resized.shape[0])
    # The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # x, thresh1 = cv2.threshold(blurred, 120, 255, cv2.THRESH_TOZERO) #120 value
    x, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ai line a problem hoicilo IM2 barti rakhar jonno, ar ai line ta rekhe deya hoice jate pore bujhte subidha hoi.
        # IM2, cnts, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts, hierarchy = cv2.findContours(
        thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("number of shapes{}" .format(len(cnts)))

        # find contours in the thresholded image and initialize the shape detector
        # findContours return three values.IM2= binary image,cnts= contours and hierarchy
        # cv2.RETR_EXTERNAL= we can find the exteriar of the contours.

        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

        # loop over the contours
    for cnt in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            # moments() used to find the center,area, and the perimeter of the controls
        M = cv2.moments(cnt);
            # finding the center of the contour(cX,cY)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]) * ratio)
        else:
            cX = 0
        if M["m01"] != 0:
            cY = int((M["m01"] / M["m00"]) * ratio)
        else:
            cY = 0
            # sending (cX,CY) the detect() function
        shape = sd.detect(cnt)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
        cnt = cnt.astype("float")
        cnt *= ratio
        cnt = cnt.astype("int")
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
            # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
            # epsilon = error_rate* actual_arc_lenght
            # epsilon find the maximum value from the contour to approximated contour
            # the arcLenght() take a contour and return its perimeter of the shape in the image/contour
        epsilon = 0.03*cv2.arcLength(cnt,True)
            # use approxPolyDp to approximate a polygon
        approx = cv2.approxPolyDP(cnt,epsilon,True)
            # if this is more, circles are not detected
            
            # triangle, red
        if len(approx)==3:
            cv2.drawContours(frame,[cnt],0,(0,0,255),3)
                # frame= image,0= for all contours; (0,0,255)=for colour; 3= thikness
            # pentagon in white
        elif len(approx) == 5:
            cv2.drawContours(frame,[cnt],0,(255,255,255),3)
            
             # rectanle, blue
        elif len(approx)==4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(255,0,0),2)

            # circle, yellow
        elif len(approx)> 10:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(frame,ellipse,(0,255,255),2)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
                # cv2.putText(frame, "Circle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0))
            
            # bigger rectangle, blue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(255,0,0),2)
            

            
        # cv2.drawContours(frame, contours,-1, (255,255,0), 1) 
    cv2.imshow('Image with contours',frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
