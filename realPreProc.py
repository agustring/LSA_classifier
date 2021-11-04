# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:33:25 2021

@author: agustring
"""
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

input_folder = 'theRealDataset\edit data'

def read_frames(label, user, sample, input_folder):
    vid = []
    path = os.path.join(input_folder, '{}_{}_{}.mp4'.format(label, user, sample))
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frames = cap.read()
        if ret:
          vid.append(frames)
        else:
          break
    return vid

def neck_video(label, user, sample):
    data_neck = []
    
    frames = read_frames(label, user, sample,input_folder)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8) as pose:
        for i in range(len(frames)):
            image = cv2.flip(frames[i], 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                neck_x = np.nan
                neck_y = np.nan
            else:
                # 12: mp_pose.PoseLandmark.RIGHT_SHOULDER
                # 11: mp_pose.PoseLandmark.LEFT_SHOULDER
                neck_x = (results.pose_landmarks.landmark[12].x + results.pose_landmarks.landmark[11].x)/2
                neck_y = (results.pose_landmarks.landmark[12].y + results.pose_landmarks.landmark[11].y)/2
            data_neck.append({
                'x': neck_x,
                'y': neck_y
                })

        return data_neck
    
def hands_video(label, user, sample):
    
    frames = read_frames(label, user, sample,input_folder)
    
    data_right = {}
    data_left = {}
    for i in range(21):
        data_right[str(i)] = []
        data_left[str(i)] = []
    
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7) as hands:
        for i in range(len(frames)):
            im2 = frames[i]        
            # Convert the BGR image to RGB before processing.
            image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                pass
            else:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    if str(results.multi_handedness[0]).find("Right")>1:
                        for i in range(len(hand_landmarks.landmark)):
                            data_right[str(i)].append({
                                'x': hand_landmarks.landmark[i].x,
                                'y': hand_landmarks.landmark[i].y,
                                'z': hand_landmarks.landmark[i].z
                            })  
                    else:
                        for i in range(len(hand_landmarks.landmark)):
                            data_left[str(i)].append({
                                'x': hand_landmarks.landmark[i].x,
                                'y': hand_landmarks.landmark[i].y,
                                'z': hand_landmarks.landmark[i].z
                            })
                            
                    if len(results.multi_hand_landmarks)==2:
                        if str(results.multi_handedness[1]).find("Left")>1:
                            hand_landmarks = results.multi_hand_landmarks[1]
                            for i in range(len(hand_landmarks.landmark)):
                                data_left[str(i)].append({
                                    'x': hand_landmarks.landmark[i].x,
                                    'y': hand_landmarks.landmark[i].y,
                                    'z': hand_landmarks.landmark[i].z
                                })
                        else:
                            for i in range(len(hand_landmarks.landmark)):
                                data_right[str(i)].append({
                                    'x': hand_landmarks.landmark[i].x,
                                    'y': hand_landmarks.landmark[i].y,
                                    'z': hand_landmarks.landmark[i].z
                                }) 
                                
        return data_right,data_left
    
import json

def procPerson(label):
    import time
    t1 = time.time()
    for user in range(1,16):
        u = '00'+str(user)
        if user>9:
            u = '0'+str(user)
        for sample in range(1,6):
            t2 = time.time()
            s = '00'+str(sample)
            R,L = hands_video(label,u,s)
            N = neck_video(label,u,s)
            with open('theRealDataset\preproc data\{}_{}_{}_R.txt'.format(label, u, s), 'w') as outfile:
                json.dump(R, outfile)
            with open('theRealDataset\preproc data\{}_{}_{}_L.txt'.format(label, u, s), 'w') as outfile:
                json.dump(L, outfile)
            with open('theRealDataset\preproc data\{}_{}_{}_N.txt'.format(label, u, s), 'w') as outfile:
                json.dump(N, outfile)
            print('It en segundos: ',(time.time()-t2))
    print('Todo en minutos: ',(time.time()-t1)/60)

procPerson('006')          