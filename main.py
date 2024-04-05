import cv2
import numpy as np

balls2=cv2.imread("Balls2.jpg")
ghost2=cv2.imread("Ghost2.png")

'''
gray=cv2.cvtColor(ghost2,cv2.COLOR_BGR2GRAY)

blur=cv2.blur(gray,(3,3))

print(blur.shape)

if blur is None:
    print("Error: Image is Empyt")
else:
    detect=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,10,param1=20,param2=40,minRadius=10,maxRadius=300)
    print(detect)

if detect is not None:
    detect=np.uint16(np.around(detect))
    for point in detect[0,:]:
        x,y,r= point[0], point[1], point[2]
        cv2.circle(ghost2,(x,y),r,(0,255,0),2)
        cv2.imshow("balls",ghost2)
        cv2.waitKey(10)     
'''

params=cv2.SimpleBlobDetector_Params() 
params.filterByArea=True
params.minArea=100
params.filterByCircularity=True
params.minCircularity=0.7
params.filterByConvexity=True
params.minConvexity=0.1
params.filterByInertia=True
params.minInertiaRatio=0.01


detector=cv2.SimpleBlobDetector_create(params)
keypoints=detector.detect(balls2)
print(keypoints)

blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(balls2, keypoints, blank, (0,255,0), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv2.imshow('blobs',blobs)
cv2.waitKey(10000)