# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:57:38 2020

@author: HP
"""

import cv2
import numpy as np

img=cv2.imread("sign.jpeg")
hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_range=np.array([110,50,50])
upper_range=np.array([130,255,255])

mask=cv2.inRange(hsv,lower_range,upper_range)

cv2.imshow("IMAGE",img)
cv2.imshow("MASK",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()