# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:35:34 2019

Using MNIST neural network classification result to achieve image figure recognize

@author: Ultimedical
"""

from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
import tensorflow as tf

# =============================================================================
# Helper Function
# =============================================================================
rect_size = 28
#Helper Funtion
def GetRectROI(im, roi_num):
# return A SET OF (left, up, c, r)
    (width, height) = im.shape
    print(im.shape)
    RectROI = []
    for n in range(1, roi_num):
        left = width
        right = 0
        down = 0
        up = height
        for i in range(width):
            for j in range(height):
                if(im[i, j] == n):
                    if(i < left):
                        left = i
                    if(i > right):
                        right = i
                    if(j > down):
                        down = j
                    if(j < up):
                        up = j
        r = down - up
        c = right - left
        RectROI.append((left - 1, up - 1, c + 2, r + 2))
    return RectROI
  
def ResultTransfer(test):
# VECTOR Result transfer to NUMBER Result
    result = []
    (height, width) = test.shape
    for i in range(height):
        for j in range(width):
            if test[i, j] == 1:
                result.append(j)
    return result
    
    
    
# Image Load
im = Image.open('mnist_test.jpg')

# Image Pre-process
# Image Invert
inverted_image = ImageOps.invert(im)
r, g, b = inverted_image.split()
rdata = np.array(r)
# otsu threshold
th, bwdata = cv.threshold(rdata, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Labels
retval, labels = cv.connectedComponents(bwdata)
# Seperate as individal part
RectROIInfo = GetRectROI(labels, retval)
RectROI = []
RectResize = []
# Individal part resize and prepare for classifcation
for i in range(retval - 1):
    RectROI.append(labels[RectROIInfo[i][0]: RectROIInfo[i][0] + RectROIInfo[i][2], RectROIInfo[i][1]:RectROIInfo[i][1] + RectROIInfo[i][3]])

for i in range(retval - 1):
    RectNow = RectROI[i]
    RectNow = np.array(RectNow, dtype = np.uint8)
    ret, RectNow  = cv.threshold(RectNow, 0, 255, cv.THRESH_BINARY)
    (height, width) = RectNow.shape
    
    if width * height < 100:
        continue

    if height > width:
        tempROI = np.zeros((height, height))
        NewBegin = round((height - width) / 2)
        for w in range(width):
            for h in range(height):
                tempROI[h, w + NewBegin] = RectNow[h, w]
    else:
        tempROI = np.zeros((width, width))

        NewBegin = round((width - height) / 2)
        for w in range(width):
            for h in range(height):
                tempROI[h + NewBegin, w] = RectNow[h, w] 
    resizeROI = cv.resize(tempROI, (rect_size, rect_size), interpolation=cv.INTER_CUBIC)
    maxresizeROI = np.max(resizeROI)
    resizeROI = resizeROI / maxresizeROI * 255.0
    resizeROI = np.array(resizeROI, dtype = int)
    resizeROI = resizeROI.tolist()
    RectResize.append(resizeROI)

RectResize = np.array(RectResize, dtype = np.uint8)
    
# Classification
# Model Load
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('./mnist/mnist_checkpoint')

# Prediction and Result Process
result = model.predict(RectResize)
result_show = ResultTransfer(result)

# Image Show
plt.figure(1)
plt.imshow(r, 'gray')
plt.figure(2)
plt.imshow(bwdata, 'gray')
plt.figure(3)
plt.imshow(RectResize[0], 'gray')
plt.figure(4)
plt.title('Figure Classification')
for i in range(len(RectResize)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(RectResize[i])
    plt.xlabel(str(result_show[i]))
    plt.xticks([])
    plt.yticks([])