import numpy as np
import cv2
from shutil import copyfile
from matplotlib import pyplot as plt
import os
from skimage.feature import hog


# NUM_CLUSTER = 2
NUM_CLUSTER = 4
# NUM_CLUSTER = 4
# NUM_CLUSTER = 5

KMEAN_LOOP = 5000
NUM_SAMPLE = 2040
# NUM_SAMPLE = 100

listImage = 'server_Kinect_1.txt'
rootDir = '/home/nguyenductrung/darknet/darknet-v2-b/darknet/'
resultDir = 'result_'+str(NUM_CLUSTER)+'/'
resultFile = 'result_'+str(NUM_CLUSTER)+'.txt'

def calculate_hog_feature(img_path):
    img_path = img_path.replace('\n', '')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    txt_path = img_path.replace('.jpg', '.txt')
    with open(txt_path, 'r') as groundTrust:
        line = groundTrust.readline()
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
        return feature

image_features = []

image_paths = []

count = 0
print('Loading data.....')
with open(listImage, 'r') as f:
    for line in f:
        image_paths.append(line)
        count += 1
        imgPath = rootDir + line
        feature = calculate_hog_feature(imgPath)
        feature = np.float32(feature)
        # np.append(image_features, feature)
        image_features.append(feature)
        if (count == NUM_SAMPLE): break

image_features = np.asarray(image_features)

term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)

print('Calculating Kmeans.....')

ret, labels, centers = cv2.kmeans(image_features, NUM_CLUSTER, None, term_crit, KMEAN_LOOP, cv2.KMEANS_RANDOM_CENTERS)


# SHOW IMAGE
print('Save results ..... ')
with open(resultFile, 'a') as result:
    for index, x in enumerate(labels):
        result.write(image_paths[index] + 'Label: ' + str(x[0]) + '\n')
        img_path = image_paths[index]
        img_path = img_path.replace('\n', '')
        img_path = rootDir+img_path

        directory = resultDir + str(x[0])
        if not os.path.exists(directory):
            print('Create directory ', directory)
            os.makedirs(directory)
        copyfile(img_path, directory + '/' + str(index) + '.jpg')
print('Finish ..... ')
#
# # Now separate the data, Note the flatten()
# A = image_features[labels.ravel()==0]
# B = image_features[labels.ravel()==1]
# C = image_features[labels.ravel()==2]
# D = image_features[labels.ravel()==3]
#
# # Plot the data
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(C[:,0],C[:,1],c = 'g')
# plt.scatter(D[:,0],D[:,1],c = 'b')
#
# plt.scatter(centers[:,0],centers[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

# In duoc tam cua cac cluster ra