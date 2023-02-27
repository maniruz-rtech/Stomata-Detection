#!/usr/bin/env python3

import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from distutils.spawn import spawn
from posixpath import split
from re import M
VIDEO = 'hdg2-2_1_5_3_A.avi'

cap = cv2.VideoCapture(VIDEO)
iter = 0

# import all image files with the .jpg extension
#images = glob.glob ("happy_faces/*.jpg")
image_data1 = []

image_data = []
while True:
    ret, frame = cap.read()
    if iter == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float64')/255
    if ret == False:
	    break
    iter += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

image_data1 = gray
for img in range(len(image_data1)):
    this_image = cv2.imread(img, 1)
    image_data.append(this_image)

avg_image = image_data[0]
for i in range(len(image_data)):
    if i == 0:
        pass
    
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

cv2.imwrite('avg_happy_face.png', avg_image)
avg_image = cv2.imread('avg_happy_face.png')
plt.imshow(avg_image)
plt.show()


# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()