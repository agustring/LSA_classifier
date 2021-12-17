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
    frames = read_frames(label, user, sample,input_folder)
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8) as pose:
        
            data_neck = np.zeros(len(frames),dtype=object)
            
            insider = np.zeros(2,dtype=float)
            
            for i in range(len(frames)):
                image = cv2.flip(frames[i], 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if not results.pose_landmarks:
                    pass
                else:
                    # 12: mp_pose.PoseLandmark.RIGHT_SHOULDER
                    # 11: mp_pose.PoseLandmark.LEFT_SHOULDER
                    neck_x = (results.pose_landmarks.landmark[12].x + results.pose_landmarks.landmark[11].x)/2
                    neck_y = (results.pose_landmarks.landmark[12].y + results.pose_landmarks.landmark[11].y)/2
                    # neck_z = (results.pose_landmarks.landmark[12].z + results.pose_landmarks.landmark[11].z)/2
                    insider[:] = [neck_x,neck_y]
                    
                data_neck[i] = insider

    return data_neck
    
def hands_video(label, user, sample):
    
    frames = read_frames(label, user, sample,input_folder)
    
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9) as hands:
        
        data = np.zeros(len(frames),dtype=object)
        right = np.zeros((21,2))
        left = np.zeros((21,2))
        insider = np.zeros(2,dtype=float)
        
        for i in range(len(frames)):
            im2 = frames[i]        
            # Convert the BGR image to RGB before processing.
            image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
            results = hands.process(image)
            # insider[:] = np.nan
            
            if not results.multi_hand_landmarks:

                pass
            else:
                a = len(results.multi_hand_landmarks)
                if a > 2:
                    a = 2 
                for j in range(a):
                 if str(results.multi_handedness[j]).find("Right")>1:
                     for k in range(21):
                         insider[:] = [results.multi_hand_landmarks[j].landmark[k].x,
                                       results.multi_hand_landmarks[j].landmark[k].y]
                                       # results.multi_hand_landmarks[j].landmark[k].z]
                         right[k,:] = insider[:]
                 elif str(results.multi_handedness[j]).find("Left")>1:
                     for k in range(21):
                         insider[:] = [results.multi_hand_landmarks[j].landmark[k].x,
                                       results.multi_hand_landmarks[j].landmark[k].y]
                                       # results.multi_hand_landmarks[j].landmark[k].z]
                         left[k,:] = insider[:]
                         
            data[i] = np.concatenate((left,right))
    return data   
 
import codecs, json 
import time

def procPerson(label):
    t1 = time.time()
    x=0
    for user in range(1,16):
        
        u = '00'+str(user)
        
        if user>9:
            u = '0'+str(user)
            
        for sample in range(1,6):
            t2 = time.time()
            s = '00'+str(sample)
            
            H = hands_video(label,u,s)
            N = neck_video(label,u,s)
            
            data = np.zeros(len(H),dtype=object)
            data[:][:len(data)] = H
            
            for i in range(len(data)):
                for j in range(len(data[i])):
                    data[i][j][0] = float(data[i][j][0] - N[i][0])
                    data[i][j][1] = float(data[i][j][1] - N[i][1])
                    data[i][j] = data[i][j].tolist()
                data[i] = data[i].tolist()
                
            real_list = data.tolist()
            file_path = 'theRealDataset\preproc data\{}_{}_{}_pre.json'.format(label, u, s)
            json.dump(real_list, codecs.open(file_path, 'w', encoding='utf-8'), 
                      separators=(',', ':'), sort_keys=True, indent=4)
            
            print('Video numero ',x,' en segundos: ',(time.time()-t2))
            x+=1
    print('Todo en minutos: ',(time.time()-t1)/60)

#For preprocesing the whole data 
       
for user in range(6,10):
    u = '00'+str(user)
    if user>9:
        u = '0'+str(user)
    procPerson(u)  
        
#UNJSONIFY

# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)