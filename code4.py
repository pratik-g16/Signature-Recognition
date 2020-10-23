# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:44:40 2020

@author: HP
"""
# -- coding: utf-8 --
"""
Created on Tue Oct 20 22:21:05 2020

@author: HP
"""

import numpy as np
import cv2
# import pillow
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

cv2.imshow("1",ROI1)
cv2.imshow("2",ROI2)
#Compression of image
imgResize1=cv2.resize(ROI1,(100,100))
imgResize2=cv2.resize(ROI2,(100,100))
cv2.imshow("1",imgResize1)
cv2.imshow("2",imgResize2)
# print(ROI1.shape)
# print(ROI2.shape)

scale_percent=0.05
width=int(ROI1.shape[1]*scale_percent)
height=int(ROI1.shape[0]*scale_percent)
dimention=(width,height)
resized1=cv2.resize(imgResize1,dimention,interpolation=cv2.INTER_AREA)
resized2=cv2.resize(imgResize2,dimention,interpolation=cv2.INTER_AREA)

cv2.imshow("1",resized1)
cv2.imshow("2",resized2)
# print(resized1.shape)
# cv2.imwrite('Small.jpg',resized1)

# cv2.imshow('SMALL2',resized1)


# if(ROI1.shape>ROI2.shape):
#     smaller=ROI2
# else:
#     smaller=ROI1
# width=smaller.shape[1]
# height=smaller.shape[0]
# print(height,width)
# imgResize1=ROI1.resize((width,height),PIL.Image.ANTIALIAS)
# imgResize2=ROI2.resize((width,height),PIL.Image.ANTIALIAS)
# imgResize1.save("RESIZE1.jpeg")
# imgResize1.save("RESIZE2.jpeg")



# imgResize1=cv2.resize(ROI1,(100,100))
# imgResize2=cv2.resize(ROI2,(100,100))
# cv2.imshow("RESIZE1",imgResize1)
# cv2.imshow("RESIZE2",imgResize2)

cv2.waitKey()

 
# #compression
# print("Starting compresion")
# ROI1f1=ROI1("Compressed_1",optimize=True,quality=10)
# ROI2f2=ROI2("Compressed_2",optimize=True,quality=10)
#ROI1.save("Compressed_1","JPEG",optimize = True,quality = 10) 

# #LCS CODE
#converting image1 to a flat array
img1=resized1
arr1=np.array(img1)
shape1=arr1.shape
flat_arr1=arr1.ravel()

#converting image 2 to a flat array
img2=resized2
arr2=np.array(img2)
shape2=arr2.shape
flat_arr2=arr2.ravel()

#print(flat_arr1)
#print(flat_arr2)

#vector1=np.matrix(flat_arr1)
#vector2=np.matrix(flat_arr2)

#print(vector1)
#print(vector1.shape)
#print(vector2)
#print(vector2.shape)

print("Length of image 1 in array is ",str(len(flat_arr1)))
print("Length of image 2 in array is ",str(len(flat_arr2)))

#print("Length of image 1 in vector is ",str(len(vector1)))
#print("Length of image 2 in vector is ",str(len(vector2)))

def lcs_length_calculation(x,y):
    m=len(x)+1
    n=len(y)+1
    c=[[0]*n]*m
    b=[['' for i in range(n)] for j in range(m)]
    count=0
    for i in range(m):
        c[i][0]=0
        b[i][0]='H'
    for j in range(n):
        c[0][j]=0
        b[0][j]='H'
    x.insert(0,0)
    y.insert(0,0)
    
    for i in range(1,m):
        for j in range(1,n):
            if x[i]==y[j]:
                c[i][j]=c[i-1][j-1]+1
                b[i][j]='D'
                count=count+1
            else:
                if c[i-1][j] >= c[i][j-1] :
                    c[i][j]=c[i-1][j]
                    b[i][j]='U'
                else :
                    c[i][j]=c[i][j-1]
                    b[i][j]='L'
                    
    return (c,b,count)

def lcs_print_seq(b,x,i,j,l):
    if i==0 or j==0:
        return
    if b[i][j]=='D':
        lcs_print_seq(b,x,i-1,j-1,l)
        l.append(x[i])
    elif b[i][j]=='U':
        lcs_print_seq(b,x,i-1,j,l)
    else:
        lcs_print_seq(b,x,i,j-1,l)
       
list_arr1=flat_arr1.tolist()
list_arr2=flat_arr2.tolist()
c,b,count=lcs_length_calculation(list_arr1,list_arr2);
#print(c)
#print(b)
l=[]
lcs_print_seq(b,list_arr1,len(list_arr1)-1,len(list_arr2)-1,l)
print()
# print("longest Common Subsequence is : ",l)
print()
print("Length of longest Common Subsequence is : ",len(l))

print("The 2 images are "+str(    len(l)/len(flat_arr1) *100    )+"% same" )

"""
x=[1,2,3,4,5]
y=[1,3,5,7]
c,b,count=lcs_length_calculation(x,y);
#print(c)
#print(b)
l2=[]
lcs_print_seq(b,x,len(x)-1,len(y)-1,l2)
print(l2)

"""