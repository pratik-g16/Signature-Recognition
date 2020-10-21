# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:21:05 2020

@author: HP
"""
import numpy as np
import cv2
image1 = cv2.imread("sign.jpeg")
# cv2.imshow("IMAGE",image1)

result1 = image1.copy()
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
mask1 = cv2.inRange(image1, lower, upper)
#image in black and white
# cv2.imshow("MASK1",mask1)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel1, iterations=1)
close1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel1, iterations=2)

cnts = cv2.findContours(close1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

boxes = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

boxes = np.asarray(boxes)
left = np.min(boxes[:,0])
top = np.min(boxes[:,1])
right = np.max(boxes[:,2])
bottom = np.max(boxes[:,3])

result1[close1==0] = (255,255,255)
ROI1= result1[top:bottom, left:right].copy()
cv2.rectangle(result1, (left,top), (right,bottom), (36, 255, 12), 2)

# cv2.imshow('result', result1)
# cv2.imshow('ROI', ROI1)
# cv2.imshow('close', close1)
cv2.imwrite('result.png', result1)
cv2.imwrite('ROI.png', ROI1)
cv2.waitKey()




image2 = cv2.imread("sign.jpeg")
# cv2.imshow("IMAGE",image2)

result2 = image2.copy()
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
mask2 = cv2.inRange(image2, lower, upper)
#image in black and white
# cv2.imshow("MASK2",mask2)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel2, iterations=1)
close2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel2, iterations=2)

cnts = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

boxes = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

boxes = np.asarray(boxes)
left = np.min(boxes[:,0])
top = np.min(boxes[:,1])
right = np.max(boxes[:,2])
bottom = np.max(boxes[:,3])

result2[close2==0] = (255,255,255)
ROI2 = result2[top:bottom, left:right].copy()
cv2.rectangle(result2, (left,top), (right,bottom), (36, 255, 12), 2)

# cv2.imshow('result2', result2)
# cv2.imshow('ROI2', ROI2)
# cv2.imshow('close2', close2)
cv2.imwrite('result2.png', result2)
cv2.imwrite('ROI2.png', ROI2)


if(ROI1.shape>ROI2.shape):
    smaller=ROI2
else:
    smaller=ROI1
width=smaller.shape[1]
height=smaller.shape[2]
imgResize1=cv2.resize(ROI1,(width,height))
imgResize2=cv2.resize(ROI2,(width,height))


imgResize1=cv2.resize(ROI1,(100,100))
imgResize2=cv2.resize(ROI2,(100,100))
cv2.imshow("RESIZE1",imgResize1)
cv2.imshow("RESIZE2",imgResize2)
cv2.waitKey()

