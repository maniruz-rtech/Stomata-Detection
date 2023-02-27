import numpy as np
import cv2

def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value

trackbar_value = 0
   
Blur1 = 31
Blur2 = 5
perc = 80
img = cv2.imread("test1.png", 0)
cv2.namedWindow("Processed Image")
cv2.createTrackbar("gker","Processed Image", 0, 255, updateValue)
cv2.createTrackbar("gker1","Processed Image", 0, 255, updateValue)


while True:
    Blur1 = cv2.getTrackbarPos('gker',"Processed Image") or 31
    Blur2 = cv2.getTrackbarPos('gker1',"Processed Image") or 5
    
    blur1 = cv2.GaussianBlur(img,(Blur1,Blur1),0)
    blur2 = cv2.GaussianBlur(img,(Blur2,Blur2),0)
    cv2.imshow("blur2",blur2)
    res = blur1.astype('float')-blur2.astype('float')


    res-=np.amin(res)
    res/=np.amax(res)
    #print(np.percentile(res,perc))
    res = (res>np.percentile(res,perc)).astype("uint8")*255

    cv2.imshow("Original", img)
    cv2.imshow('Processed Image', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

 
'''
VIDEO = 'hdg2-2_1_5_3_A.avi'
DILATION = 5
DILATION_ITER = 5
BLURSIZE = 101
 
cap = cv2.VideoCapture(VIDEO)


HEAT = None
HEATDIV = None
iter = 0

while(cap.isOpened()):
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
        OUT = frame[:,:,0].astype('float64')/255
    if ret == False:
	    break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float64')/255
    MASK = (HEAT==iter)
    OUT[MASK] = gray[MASK]
	
    #cv2.imshow('mask', MASK.astype("float64"))
    cv2.imshow('frame', OUT)
    iter += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cv2.imwrite(VIDEO.replace('.avi','.png'), (OUT*255).astype("uint8"))
'''