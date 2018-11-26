import numpy as np
import cv2
from shutil import copyfile
from matplotlib import pyplot as plt
import os
from skimage.feature import hog
from sklearn.cluster import KMeans
import pickle


# NUM_CLUSTER = 2
NUM_CLUSTER = 3
# NUM_CLUSTER = 4
# NUM_CLUSTER = 5

KMEAN_LOOP = 5000
NUM_SAMPLE = 10
# NUM_SAMPLE = 100
RANDOM_STATE = 100

listImage = 'kinect_1.txt'
# rootDir = '/home/nguyenductrung/darknet/darknet-v2-b/darknet/'
# rootDir = '/Users/trungnd/pfiev/dataset/'
rootDir = '/media/trungnd/Data/'

resultDir = 'result_'+str(NUM_CLUSTER)+'/'
resultFile = 'result_'+str(NUM_CLUSTER)+'.txt'

def calculate_hog_feature(img_path):
    img_path = img_path.replace('\n', '')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    txt_path = img_path.replace('.jpg', '.txt')
    with open(txt_path, 'r') as groundTrust:
        line = groundTrust.readline()
        groundTrustData = line
        params = line.split(' ')

        yolo_x = float(params[1])
        yolo_y = float(params[2])
        yolo_w = float(params[3])
        yolo_h = float(params[4])
        img_w = float(640)
        img_h = float(480)

        obj_w = int(yolo_w * img_w)
        obj_h = int(yolo_h * img_h)

        obj_x = int(yolo_x * img_w - obj_w / 2)
        obj_y = int(yolo_y * img_h - obj_h / 2)
        subImg = img[obj_y: obj_y + obj_h, obj_x: obj_x + obj_w]

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

        subImg = cv2.resize(subImg, (newWidth, newHeight))

        if (newWidth > newHeight):
            subImg = np.concatenate((subImg, fillArear), axis=0)
        else:
            subImg = np.concatenate((subImg, fillArear), axis=1)

        feature, hog_image = hog(subImg, orientations=5, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualize=True, visualise=True, multichannel=True,
                                 feature_vector=True)
        return (feature, groundTrustData)

image_features = []

image_paths = []

count = 0
print('Loading data.....')

# Load kmeans data
saveModelsName = '../kmeans_models.sav'
kmeans = pickle.load(open(saveModelsName, 'rb'))

with open(listImage, 'r') as f:
    for line in f:
        image_paths.append(line)
        count += 1
        imgPath = rootDir + line
        feature, groundTrustData = calculate_hog_feature(imgPath)
        feature = np.float32(feature)

        labels = kmeans.predict(np.asarray([feature]))
        label = labels[0]

        print(line, groundTrustData)
        # Save image with labels

# # SHOW IMAGE
# print('Save results ..... ')
# with open(resultFile, 'a') as result:
#     for index, x in enumerate(labels.tolist()):
#         print('Kmean', index, x)
#         # result.write(image_paths[index] + 'Label: ' + str(x[0]) + '\n')
#         result.write(image_paths[index] + 'Label: ' + str(x) + '\n')
#         img_path = image_paths[index]
#         img_path = img_path.replace('\n', '')
#         img_path = rootDir+img_path
#
#         # directory = resultDir + str(x[0])
#         directory = resultDir + str(x)
#         if not os.path.exists(directory):
#             print('Create directory ', directory)
#             os.makedirs(directory)
#         copyfile(img_path, directory + '/' + str(index) + '.jpg')
