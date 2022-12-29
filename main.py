#!/usr/bin/env python
# coding: utf-8

# In[143]:


import cv2
import sys
import matplotlib.pyplot as plt
import time


# In[144]:


class FaceDetection():
    def __init__(self, img, color=(0, 0, 255), thickness=2):
        self.img = img.copy()
        self.color = color
        self.thickness = thickness
        
    def detect(self, show_result=True):
        # convert image to gray scale image
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # making face detector model
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # detecting faces
        face_cordinates = face_detector.detectMultiScale(
                                      gray_img,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(30,30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
        # drawing a rectangle around faces
        if len(face_cordinates) > 0:
            print(str(len(face_cordinates)) + 'Face Detected')
        else:
            print('No Face Detected.')
        for (x, y, w, h) in face_cordinates:
            start_point = (x, y)
            end_point = (x+w, y+h)
            cv2.rectangle(self.img, start_point, end_point, self.color, self.thickness)
        # showing image
        if show_result:
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.show()
        return self.img


# In[145]:


video_capturer = cv2.VideoCapture(0)
while True:
    ret, frame = video_capturer.read()
    face_detection = FaceDetection(frame)
    result_img = face_detection.detect()

