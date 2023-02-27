#!/usr/bin/env python3
# -*- coding: utf-8 -*
import glob
import cv2
import numpy
import tifffile
# cap = cv2.VideoCapture('./hdg2-2_1_5_3_A.avi')

# success, frame = cap.read()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
width = 2560
height = 2160

datfiles = glob.glob('*.avi')

with tifffile.TiffWriter('datfiles.tif', bigtiff=False) as tif:
    for datfile in datfiles:
        data = numpy.fromfile(
            datfile,
            count=width * height * 3 // 2,  # 12 bit packed
            offset=4,  # 4 byte integer header
            dtype=numpy.uint8,
        ).astype(numpy.uint16)
        image = numpy.zeros(width * height, numpy.uint16)
        image[0::2] = (data[1::3] & 15) | (data[0::3] << 4)
        image[1::2] = (data[1::3] >> 4) | (data[2::3] << 4)
        image.shape = height, width
        tifffile.imsave(image, photometric='minisblack', compression=None, metadata=None)