import numpy as np
import matplotlib.pyplot as plt
import cv2

# image = cv2.imread('../Data/ClassificationCity/CleanedData/train/3_original.jpg', cv2.IMREAD_GRAYSCALE)
#
# plt.figure(figsize=(15, 10))
# plt.imshow(image, cmap='gray')
# plt.show()
#
# _, threshold = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
#
# plt.figure(figsize=(15, 10))
# plt.imshow(threshold, cmap='gray')
# plt.show()
#
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
# for cnt in contours:
#     if len(cnt) > 100:
#         print(len(cnt))
#         approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#         cv2.drawContours(image, [approx], 0, 0, 2)
#
#
# plt.figure(figsize=(15, 10))
# plt.imshow(image, cmap='gray')
# plt.show()


#
# image = cv2.imread('../Data/ClassificationCity/CleanedData/train/3_original.jpg', cv2.IMREAD_GRAYSCALE)
#
# img = cv2.medianBlur(image, 5)
# ret, th1 = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
#
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
#
#
import cv2 as cv

img = cv.imread('../Data/ClassificationCity/CleanedData/images/3_original.jpg', cv2.IMREAD_GRAYSCALE)
# global thresholding
ret1,th1 = cv.threshold(img,80,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,80,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(7,7),0)

ret3,th3 = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]


titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()