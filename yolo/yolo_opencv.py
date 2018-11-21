# Usage example:  python yolo_opencv.py --video=dataset/color.avi
#                 python yolo_opencv.py --image=bird.jpg
# /media/data/datasets/Kinect2017-10/Datasets/20171123_Hung_lan1_23-11-2017__11-05-57/Kinect_1

# cp /media/data/datasets/Kinect2017-10/Datasets/20171123_Hung_lan1_23-11-2017__11-05-57/Kinect_1/color.avi /home/nguyenductrung/hog_kmean/hog_kmean/yolo/dataset
# cp /home/nguyenductrung/darknet/darknet-v2-b/darknet/train_data_v2/backup_1/train_yolo_v2_10000.weights /home/nguyenductrung/hog_kmean/hog_kmean/yolo/data


import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

from shutil import copyfile
from matplotlib import pyplot as plt
import os
from skimage.feature import hog
from sklearn.cluster import KMeans
import pickle

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
# classesFile = "coco.names";
classesFile = "data/all.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
# modelConfiguration = "yolov3.cfg";
# modelWeights = "yolov3.weights";

modelConfiguration = "data/valid_yolo_v2.cfg";
modelWeights = "data/train_yolo_v2_kinect_1.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Load kmeans data
saveModelsName = 'kmeans_models.sav'
kmeans = pickle.load(open(saveModelsName, 'rb'))
postureLabels = ['sitting', 'standing', 'lying']


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Calculate HOG feature on bounding box


def calculate_hog_feature(classId, conf, left, top, right, bottom):

    x1 = left
    x2 = right
    y1=top
    y2=bottom
    obj_w = int(x2 - x1)
    obj_h = int(y2 - y1)

    obj_x = int(x1)
    obj_y = int(y1)
    subImg = frame[obj_y: obj_y + obj_h, obj_x: obj_x + obj_w]

    newWidth = 0
    newHeight = 0
    fillArear = 0

    if (obj_w > obj_h):
        newWidth = 128
        newHeight = int(obj_h / obj_w * 128)
        fillArear = np.full((newWidth - newHeight, 128, 3), 0)
    else:
        newHeight = 128
        newWidth = int(obj_w / obj_h * 128)
        fillArear = np.full((128, newHeight - newWidth, 3), 0)

    subImg = cv.resize(frame, (newWidth, newHeight))

    if (newWidth > newHeight):
        subImg = np.concatenate((subImg, fillArear), axis=0)
    else:
        subImg = np.concatenate((subImg, fillArear), axis=1)

    feature, hog_image = hog(subImg, orientations=5, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=True, multichannel=True,
                             feature_vector=True, block_norm='L1')

    feature = np.float32(feature)
    return feature



# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    hogFeatures = calculate_hog_feature(classId, conf, left, top, right, bottom)
    postureLabel = kmeans.predict(np.asarray([hogFeatures]))

    postureLabelText = postureLabels[postureLabel[0]]

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    # Calculate feature




    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv.putText(frame, postureLabelText, (0, 470), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
# winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    # cv.imshow(winName, frame)


