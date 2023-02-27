import cv2
import os
import sys
import numpy
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

os.chdir('/home/maniruz/Downloads/Stomata')
# MIN_MATCH_COUNT = 10

def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)

    
    # enhanced_img = numpy.array(temp0)
    # filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 10
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

    return temp2


def get_descriptors(img):
	#CLAHE=Contrast Limited Adaptive Histogram Equalization
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
	# Threshold
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# Normalize to 0 and 1 range
	img[img == 255] = 1
	img_inv = cv2.bitwise_not(img)
	
	#Thinning
	skeleton = skeletonize(img)
	skeleton = removedot(skeleton)
	skeletonF32 = numpy.float32(skeleton)

	# Harris corners= combination of edge and corner detector
	harris_corners = cv2.cornerHarris(skeletonF32, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125

	# Extract keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))

	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, des = orb.compute(img, keypoints)

	return (skeleton,keypoints,des);


def main():
	# the arguments (as separated by spaces) on the command-line
	image_name = sys.argv[1]

	#load image
	img1 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	#sending the image to the function called get_descriptors()
	# used to __get__, __set__ or __delete__ the 
	ske1, kp1, des1 = get_descriptors(img1)
	
	image_name = sys.argv[2]
	img2 = cv2.imread("database/" + image_name, cv2.IMREAD_GRAYSCALE)
	ske2, kp2, des2 = get_descriptors(img2)

	ske1 = cv2.normalize(ske1, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	ske2 = cv2.normalize(ske2, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	# plt.imshow(img1)
	plt.imshow(ske2)
	# Matching between descriptors img
	#BFMatcher takes two optional params good for SIFT and SURF
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x:x.distance)

	# Calculate score
	score = 0;
	for match in matches:
		score += match.distance
	score_threshold = 33
	if score/len(matches) < score_threshold:
		print("Fingerprint matches.")
		# Plot keypoints
		img3 = cv2.drawKeypoints(ske1, kp1, outImage=None)
		img4 = cv2.drawKeypoints(ske2, kp2, outImage=None)

		f, axarr = plt.subplots(1, 2)
		axarr[0].imshow(img3)
		axarr[1].imshow(img4)

		plt.show()
		# Plot matches
		img5 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], flags=2, outImg=None)
		plt.imshow(img5)
		plt.show()
		img6 = cv2.drawMatches(ske1, kp1, ske2, kp2, matches[:20], flags=2, outImg=None)
		plt.imshow(img6)
		plt.show()
	else:
		print("Fingerprint does not match.")


	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append([m])
	# cv2.drawMatchesKnn expects list of lists as matches.
	img7 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)
	plt.imshow(img7), plt.show()

if __name__ == "__main__":
	try:
		main()
	except:
		raise
