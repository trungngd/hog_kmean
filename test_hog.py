import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, exposure
import numpy as np

# Dung
# img_path = 'dataset/300.jpg'
# result_name = 'dung'
# Dang nga
# img_path = 'dataset/3450.jpg'
# result_name = 'dang_nga'
# Nam ngang
# img_path = 'dataset/4600.jpg'
# result_name = 'nam_ngang'
# Ngoi
# img_path = 'dataset/6500.jpg'
# result_name = 'ngoi'
# Nam thang dung
img_path = 'dataset/9050.jpg'
result_name = 'nam_dung'

result_path = 'result_hinh_trang_hog/'+result_name+'.jpg'


# image = data.astronaut()

image = cv2.imread(img_path, cv2.IMREAD_COLOR)

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
                                 feature_vector=True, block_norm='L2')
        return hog_image

hog_img = calculate_hog_feature(img_path)

# cv2.imwrite(result_path, hog_img)
imgplot = plt.imshow(hog_img, cmap=plt.cm.gray)
plt.show()