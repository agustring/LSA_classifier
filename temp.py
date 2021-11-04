# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:18:39 2021

@author: agustring
"""


import math
import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import numpy as np

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()

# For static images:
def read_frames(label, user, sample, input_folder):
    """
    Reads the frames for the given sample
    """
    vid = []
    path = os.path.join(input_folder, '{}_{}_{}.mp4'.format(label, user, sample))
    # path = os.path.join('Nuevis', 'IMG_8787.MOV')
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frames = cap.read()
        if ret:
          vid.append(frames)
        else:
          break
    return vid

width = 1280
hieght = 720
channel = 3
 
fps = 29
# sec = 5

for i in range(1,16):
    if i>9:
        b = '0'+str(i)
    else:
        b = '00'+str(i)
    frames = read_frames('008',b,'004','theRealDataset\edit data')    
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    
    video = cv2.VideoWriter('testing8{}.mp4'.format(i), fourcc, float(fps), (width, hieght))
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8) as hands:
        
      for i in range(len(frames)):
        image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if not results.multi_hand_landmarks:
          continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              annotated_image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          im = cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1)
          video.write(im)
    
    video.release()

# frames = read_frames('001','004','004','theRealDataset\edit data')    
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# i=1
# video = cv2.VideoWriter('testing1{}.mp4'.format(i), fourcc, float(fps), (width, hieght))

# with mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7) as hands:
    
#   for i in range(len(frames)):
#     image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
#     results = hands.process(image)
    
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#       im = cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1)
#       video.write(im)

# video.release()

                # for j in range(len(results.multi_hand_landmarks)):
                #     if str(results.multi_handedness[j]).find("Right")>1 and len(results.multi_hand_landmarks) == 1:
                #         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                #             data_right[str(i)].append({
                #                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                #                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                #                 'z': results.multi_hand_landmarks[j].landmark[i].z
                #             })
                #             data_left[str(i)].append({
                #                 'x': 0.0, 
                #                 'y': 0.0, 
                #                 'z': 0.0
                #             })
                #     elif str(results.multi_handedness[j]).find("Left")>1 and len(results.multi_hand_landmarks) == 1:
                #         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                #             data_left[str(i)].append({
                #                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                #                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                #                 'z': results.multi_hand_landmarks[j].landmark[i].z
                #             })
                #             data_right[str(i)].append({
                #                 'x': 0.0, 
                #                 'y': 0.0, 
                #                 'z': 0.0
                #             })
                #     elif str(results.multi_handedness[j]).find("Right")>1 and len(results.multi_hand_landmarks) == 2:
                #         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                #             data_right[str(i)].append({
                #                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                #                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                #                 'z': results.multi_hand_landmarks[j].landmark[i].z
                #             })
                #     elif str(results.multi_handedness[j]).find("Left")>1 and len(results.multi_hand_landmarks) == 2:
                #         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                #             data_left[str(i)].append({
                #                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                #                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                #                 'z': results.multi_hand_landmarks[j].landmark[i].z
                #             })