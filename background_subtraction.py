import cv2
import numpy as np
import sys

# read input video
VIDEO = 'hdg2-2_1_5_3_A.avi'
cap = cv2.VideoCapture(VIDEO)
if (cap.isOpened()== False):
    print("!!! Failed to open video")
    sys.exit(-1)

# retrieve input video frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
#print('* Input Video settings:', frame_width, 'x', frame_height, '@', fps)

# adjust output video size
frame_height = int(frame_height / 2)
#print('* Output Video settings:', frame_width, 'x', frame_height, '@', fps)

# create output video
#video_out = cv2.VideoWriter('traffic_out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
#video_out = cv2.VideoWriter('traffic_out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height), True)

# create MOG
backSub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=156, detectShadows=True)
iter = 0
while (True):
    # retrieve frame from the video
    ret, frame = cap.read() # 3-channels
    if (frame is None):
        break

    # resize to 50% of its original size
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # gaussian blur helps to remove noise
    blur = cv2.GaussianBlur(frame, (7,7), 0)
    #cv2.imshow('frame_blur', blur)

    # subtract background
    fgmask = backSub.apply(blur) # single channel
    #cv2.imshow('fgmask', fgmask)

    # concatenate both frames horizontally and write it as output
    fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR) # convert single channel image to 3-channels
    out_frame = cv2.hconcat([blur, fgmask_bgr]) # 
    #print('output=', out_frame.shape) # shape=(360, 1280, 3)

    cv2.imshow('output', out_frame)
    #video_out.write(out_frame)
    iter += 1

    # quick pause to display the windows
    if (cv2.waitKey(1) == 27):
        break
    
# release resources
cap.release()
#video_out.release()
cv2.destroyAllWindows()

cv2.imwrite(VIDEO.replace('.avi', 'background_subtraction.png'), ((fgmask_bgr)/iter).astype("uint8"))

