# The Script written by:
#  SHIVAM GUPTA(SXG190040), KAVIN KUPPUSAMY(KXK190026), PRACHI VATS(PXV180021), BHAVYA SREE BOMBAY(BXB180036)

# Importing all the required libraries
import glob 
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as mp
import glob
from cnn_model.model_network import CNN_NET
from skimage import io
import numpy as np
from skimage.transform import rescale, resize


print('Loading and Preparing Training Image data......')

# Loading the Training Images data
train_img = []
training_img_dir = r"training_datasets"
for img in glob.glob( training_img_dir + '/*' + '/*.jpg'):
#     cv_img.append(resize(io.imread(img, as_grey = True)/255, (100, 100)))
    # Image Preprocessing
    train_img.append(resize(mpimg.imread(img)/255, (100, 100, 3)))
    train_imgs = np.array(train_img)
train_imgs -= int(np.mean(train_imgs))
train_imgs.shape

print('Preparing Training Image data Labels......')

# Creating the Training Images Class Labels with One hot Coding
Class_num = 6
class_lab = []
classes = glob.glob( training_img_dir + '/*')
count = 0
for clas in classes:
    c_len = len(glob.glob( clas + '/*.jpg'))
    class_lab.extend([count] * c_len)
    count += 1
class_labels = np.array(class_lab) 
training_labels = np.eye(Class_num)[class_labels]

print('Loading and Preparing Testing Image data......')

# Loading the Testing Images data
testing_img_dir = r"testing_datasets"
test_img = []
for img in glob.glob( testing_img_dir + '/*' + '/*.jpg'):
#     cv_img.append(resize(io.imread(img, as_grey = True)/255, (100, 100)))
    # Image Preprocessing
    test_img.append(resize(mpimg.imread(img)/255, (100, 100, 3)))
    test_imgs = np.array(test_img)
test_imgs -= int(np.mean(test_imgs))


print('Preparing Testing Image data Labels......')

# Creating the Testing Images Class Labels with One hot Coding
Class_num = 6
class_lab = []
classes = glob.glob( testing_img_dir + '/*')
count = 0
for clas in classes:
    c_len = len(glob.glob( clas + '/*.jpg'))
    class_lab.extend([count] * c_len)
    count += 1
class_labels = np.array(class_lab) 
testing_labels = np.eye(Class_num)[class_labels]


test_imgs_count = 1500
batch_size = 30
Epochs = 5
CNN = CNN_NET()
print('Training CNN......')
#  Training the CNN Model
CNN.training(train_imgs, training_labels, batch_size, Epochs, 'cnn_model_weights.pkl')
print('Testing CNN......')
#  Testing the CNN Model
CNN.testing(test_imgs, testing_labels, test_imgs_count)


