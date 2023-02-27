#here import numpy and cv
import numpy as np
import cv2
 
VIDEO = 'hdg2-2_1_5_3_A.avi'
DILATION = 5
DILATION_ITER = 5
BLURSIZE = 101
#VideoCapture()=Class for video capturing from video files, image sequences or cameras.
cap = cv2.VideoCapture(VIDEO)


HEAT = None
HEATDIV = None
iter = 0
# Check if video/camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  # if frame is read correctly ret is True
  #Our operations on the frame come here inside the frame
    ret, frame = cap.read()
    if iter == 0:
        HEAT = np.zeros_like(frame[:,:,0]).astype('float64')
        HEATDIV = np.zeros_like(frame[:,:,0]).astype('float64')
    if ret == False:
	    break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    edges = cv2.Canny(image=gray, threshold1=90, threshold2=90)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATION,DILATION))
    edges = cv2.dilate(edges,kernel,iterations = DILATION_ITER)
    blur = cv2.GaussianBlur(edges,(BLURSIZE,BLURSIZE),0)
	
    frame[:,:,0] = blur[:,:]
    blur = blur.astype('float64')/255

    MASK = (HEATDIV < blur)

    HEAT[MASK] = np.zeros_like(HEAT)[MASK]+iter

    HEATDIV[MASK] = blur[MASK]

    cv2.imshow('heatmap', HEAT/np.max(HEAT))
    cv2.imshow('frame', frame)
	
    iter += 1	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.imwrite(VIDEO.replace('.avi','_heat.png'), (HEAT/np.max(HEAT)*255).astype("uint8"))

cap = cv2.VideoCapture(VIDEO)
OUT = None
iter = 0
while True:
    ret, frame = cap.read()
    if iter == 0:
	#float64 increase the brightness of the image and 255 is just dividing the image nothing happening here
        OUT = frame[:,:,0].astype('float64')/255
    if ret == False:
	    break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float64')/255
    MASK = (HEAT==iter)
    OUT[MASK] = gray[MASK]
	
    #cv2.imshow('mask', MASK.astype("float64"))
    cv2.imshow('frame', OUT)
    iter += 1
# Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


cv2.imwrite(VIDEO.replace('.avi','focus.png'), (OUT*255).astype("uint8"))
