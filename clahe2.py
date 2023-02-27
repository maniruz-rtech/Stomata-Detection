from __future__ import print_function
import cv2
import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              Opencv2. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='hdg2-2_1_5_3_A.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
#capture = cv2.VideoCapture('hdg2-2_1_5_3_A.avi') #(cv2.samples.findFileOrKeep(args.input))
img = cv2.imread("frame117.png")
while True:
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    fgMask = backSub.apply(img)
        
        
    #cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    
