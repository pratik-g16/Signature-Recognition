# -- coding: utf-8 --
"""
Created on Tue Oct 20 22:21:05 2020

@author: HP
"""

import numpy as np
import cv2


#lets us print the complete array without truncation
#np.set_printoptions(threshold=sys.maxsize)

#---------------------------input----------------------------------

#taking an image
image1 = cv2.imread("niraj1.jpeg")
result1 = image1.copy()

#--------------------detecting and cropping signature 1---------------------

#converting image from bgr to hsv 
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
#converting the hsv image to a mask(threshold)
mask1 = cv2.inRange(image1, lower, upper)

#removing noise
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#using opening
opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel1, iterations=1)
#using closing
close1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel1, iterations=2)

#finding a curve joining all continuous points
cnts = cv2.findContours(close1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

boxes = []
for c in cnts:
    #using cv2.boundingRect to highlight the area of use from contours
    (x, y, w, h) = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

#finding top,left,right and bottom most coordinates of the signature
boxes = np.asarray(boxes)
left = np.min(boxes[:,0])
top = np.min(boxes[:,1])
right = np.max(boxes[:,2])
bottom = np.max(boxes[:,3])

#cropping image
result1[close1==0] = (255,255,255)
ROI1= result1[top:bottom, left:right].copy()
cv2.rectangle(result1, (left,top), (right,bottom), (36, 255, 12), 2)
cv2.waitKey()

#--------------------detecting and cropping signature 2---------------------

#image 2
image2 = cv2.imread("niraj2.jpeg")
result2 = image2.copy()
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
mask2 = cv2.inRange(image2, lower, upper)

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

#convert bgr to gray for smaller and more accurate array
ROI1=cv2.cvtColor(ROI1,cv2.COLOR_BGR2GRAY)
ROI2=cv2.cvtColor(ROI2,cv2.COLOR_BGR2GRAY)


#-------------------------Image printing------------------------------
#mask
cv2.imshow("MASK1",mask1)
cv2.imshow("MASK2",mask2)

#image while cropping
cv2.imshow("RESULT1",result1)
cv2.imshow("RESULT2",result2)

#cropped image
cv2.imshow("ROI1",ROI1)
cv2.imshow("ROI2",ROI2)
#----------------------------Compressing image-------------------------------

#Resize to same height and width first
imgResize1=cv2.resize(ROI1,(100,100))
imgResize2=cv2.resize(ROI2,(100,100))

#changing image quality according to scale_percent
scale_percent=0.05
#from shape array get height and width
width=int(ROI1.shape[1]*scale_percent)
height=int(ROI1.shape[0]*scale_percent)
dimention=(width,height)
#using interpolation technique-interarea to resize
resized1=cv2.resize(imgResize1,dimention,interpolation=cv2.INTER_AREA)
resized2=cv2.resize(imgResize2,dimention,interpolation=cv2.INTER_AREA)

cv2.waitKey()


#-----------------------------image to array-----------------------------
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


#--------------------------1-d array to pixel array--------------------------

#taking all the indices for gray and black pixels from flat_arr1,flat_arr2

a=[]
for i in range(0,len(flat_arr1)):
    if(flat_arr1[i]!=255):
        a.append(i)
    
b=[]
for i in range(0,len(flat_arr2)):
    if(flat_arr2[i]!=255):
        b.append(i)
flat_arr1=a
flat_arr2=b
# print(a)
# print(b)

#---------------------------------LCS Code-------------------------------------

# print("Length of image 1 in array is ",str(len(flat_arr1)))
# print("Length of image 2 in array is ",str(len(flat_arr2)))

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
list_arr1=a
list_arr2=b
c,b,count=lcs_length_calculation(list_arr1,list_arr2);
l=[]
lcs_print_seq(b,list_arr1,len(list_arr1)-1,len(list_arr2)-1,l)
print()
print()

#----------------------------printing output--------------------------------

# print("Length of longest Common Subsequence is : ",len(l))
if(len(flat_arr1)== len(flat_arr2)):
    print("The 2 images are "+str(    len(l)/len(flat_arr1) *100    )+"% same" )
else:
    print("The 2 images are "+str(  "{:.2f}".format(  2*len(l)/(len(flat_arr1)+len(flat_arr2)) *100   ) )+"% same" )