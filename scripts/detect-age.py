""" Age Detection Using pre-trained models

For this exemples I'm going to test using pre-trained models. Later on I'll be trying using
AlexNet to make the entire training and evaluating set.

https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/
"""

import os
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

''' Why treat as classification problem instead of an regression problem?

Well, it is more difficult to the model identify the precise age than the range of it,
If you think, guess ages could be an hard issue even for us, because people try to hide
their ages. So the model also will struggle in such task. Treating as an classification
problem will relax a bit the problem making it easier for model to train.
'''

# Defining age buckers for our problem.
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"
]

''' There will be two dectection models, the first one is the face detection, to make 
sure our model is being evaluating just the Region Of Interest, leaving the noise of the 
image behind, the second one is for age detection.'''

# Load pre-trained models from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# More about blob thing below
# https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(
    image, 1.0, size=(300, 300),
	mean=(104.0, 177.0, 123.0)
)

print("[INFO] computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# extract the ROI of the face and then construct a blob from
		# *only* the face ROI
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(
            face, 1.0, size=(227, 227),
			mean=(78.4263377603, 87.7689143744, 114.895847746),
			swapRB=False
        )
		# make predictions on the age and find the age bucket with
		# the largest corresponding probability
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]
		# display the predicted age to our terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))
		# draw the bounding box of the face along with the associated
		# predicted age
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(
            image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
            (0, 0, 255), 2
        )

# display the output image
cv2.imwrite("/age-detection/image.png", image)
