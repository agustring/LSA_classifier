# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:24:59 2021

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

def a_uint(ima):
  for i in range(3):
    ima[:,:,i] = 255*(ima[:,:,i]-ima[:,:,i].min())/ima[:,:,i].max()
  return ima.astype(np.uint8)

def change_cha(ima,alpha,beta,ch):
    yiq = (ima)
    # c = 2
    for x in range(len(ima[:,1,ch])):
        for y in range(len(ima[1,:,ch])):
            if yiq[x,y,ch]>yiq[x,y,ch].max()*beta:
                yiq[x,y,ch] = yiq[x,y,ch] *alpha
            #if yiq[x,y,2]>yiq[x,y,2].max()*beta:
            #    yiq[x,y,2] = yiq[x,y,2] *alpha
    rgb = (yiq)
    return rgb

def box(r):
    se = np.ones((r*2+1,r*2+1),dtype=np.bool)
    return se

def circle(r, threshold = 0.3):
    vec = np.linspace(-r, r, r*2+1)
    [x,y] = np.meshgrid(vec,vec) 
    se = (x**2 + y**2)**0.5 < (r + threshold)
    return se

se = circle(9)

MAT_YIQ = np.array([[0.299, 0.595716, 0.211456],[0.587, -0.274453, -0.522591],[0.114, -0.321263, 0.311135]])

def rgb2yiq(_im):
    return (_im.reshape((-1, 3)) @ MAT_YIQ).reshape(_im.shape)

def yiq2rgb(_yiq):
    return (_yiq.reshape((-1, 3)) @ np.linalg.inv(MAT_YIQ)).reshape(_yiq.shape)

def _morph_multiband(im, se, op):
    result = im
    offset = (np.array(se.shape)-1)//2
    im = np.pad(im,[(offset[0],offset[0]),(offset[1],offset[1]),(0,0)],'edge')
    for y, x in np.ndindex(result.shape[:2]):
        pixels = im[y:y+se.shape[0], x:x+se.shape[1]][se]
        result[y, x, 0] = op(pixels[:,0])
    return result

def _morph_color(im, se, op):
    im2 = im
    result = _morph_multiband(im2, se, op)
    return result

def im_dilate(im, se):
    if im.ndim == 3:
        return _morph_color(im, se, np.max)
    else:
        return _morph_gray(im, se, np.max)
    
def im_erode(im, se):
    if im.ndim == 3:
        return _morph_color(im, se, np.min)
    else:
        return _morph_gray(im, se, np.min)

def im_median(im, se):
    if im.ndim == 3:
        return _morph_color(im, se, lambda pixels: np.argsort(pixels)[len(pixels)//2])
    else:
        return _morph_gray(im, se, np.median)

def im_border_ext(im, se):
    return im_dilate(im, se) - im

def im_border_int(im, se):
    return im - im_erode(im, se)

def im_gradient(im, se):
    return im_dilate(im,se) - im_erode(im,se)

def im_open(im, se):
    return im_dilate(im_erode(im, se), se)

def im_close(im, se):
    return im_erode(im_dilate(im, se), se)

def im_tophat(im,se):
    return im - im_open(im,se)

def im_bottomhat(im,se):
    return im_close(im,se) - im

# im = imageio.imread('Untitled.png')
# im2 = change_red(im[400:700,650:1000,:],.7,.7)  #LA DETECTO!!!!!!!!!
# im2 = im_open(im2,se) 


# plt.imshow(im2[:,:,0:3],'hsv')
# plt.show()


def read_frames(label, user, sample, input_folder):
    """
    Reads the frames for the given sample
    """
    vid = []
    path = os.path.join(input_folder, '{}_{}_{}.mp4'.format(label, user, sample))
    # path = os.path.join('Nuevis', 'Untitled.mp4')
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frames = cap.read()
        if ret:
          vid.append(frames)
        else:
          break
    return vid

frames = read_frames('001','004','004','theRealDataset\edit data')

# plt.imshow(frames[30])
# plt.show()

def lin_reg(data):
    #Bordes
    if data[0] is np.nan:
        i=0
        while data[i] is np.nan:
            i+=1
        data[:i]=np.full((len(data[:i]),1),data[i])[0]
    if data[len(data)-1] is np.nan:
        i=len(data)-1
        while data[i] is np.nan:
            i-=1
        data[i:] = np.full((len(data[i:]),1),data[i])[0]
    #real data
    for i in range(len(data)):
        j = i
        while data[i] is np.nan:
            i+=1
        if (i-j) != 0:
            data[j:i] = np.arange(data[j-1], data[i], (data[i]-data[j-1])/(i-j))
    return data
        

def nariz_video(label, user, sample):
    data_neck_x = []
    data_neck_y = []
    frames = read_frames(label, user, sample,'theRealDataset\edit data')
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8) as pose:
        
        for i in range(len(frames)):
            # Make detection
            image = cv2.flip(frames[i], 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # plt.imshow(image)
            results = pose.process(image)
            if not results.pose_landmarks:
                neck_x = np.nan
                neck_y = np.nan
            else:
                neck_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)/2            
                neck_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)/2
            data_neck_x.append(neck_x)
            data_neck_y.append(neck_y)
        
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.flip(frames[30], 1)
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # plt.imshow(annotated_image)

        return data_neck_x, data_neck_y

neck_x, neck_y= nariz_video('001','004','004')

# def trim_frame(frame, hand_x, hand_y,scale):
#     x = int((hand_x) * frame.shape[0]-scale/2)
#     y = int((hand_y) * frame.shape[0]-scale)
    
#     if x < 0: 
#         x = 0
#     elif x + scale > frame.shape[0]:
#         x = frame.shape[0] - scale
#     if y < 0: 
#         y = 0
#     elif y + scale > frame.shape[1]:
#         y = frame.shape[1] - scale
#     plt.imshow(frame[x:x+scale,y:y+scale,:])
#     return x,y

i = 30
# sca = 400
im2 = frames[i]
# dx,dy = trim_frame(ima,hand_x[i],hand_y[i],sca)
# im2 = ima[dx:dx+sca,dy:dy+sca,:]
# im2 = change_cha(im2,.7,.7, 1)  
# im2 = change_cha(im2,.7,.7, 0)
# im2 = im_open(im2,se) 
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
# plt.imshow(im2)
#No la detecta si esta todo el cuerpo

# def noisy(noise_typ,image):
#    if noise_typ == "gauss":
#       row,col,ch= image.shape
#       mean = 0
#       var = 0.001
#       sigma = var**0.5
#       gauss = np.random.normal(mean,sigma,(row,col,ch))
#       gauss = gauss.reshape(row,col,ch)
#       noisy = image + gauss
#       return noisy
#    elif noise_typ == "s&p":
#       row,col,ch = image.shape
#       s_vs_p = 0.5
#       amount = 0.004
#       out = np.copy(image)
#       # Salt mode
#       num_salt = np.ceil(amount * image.size * s_vs_p)
#       coords = [np.random.randint(0, i - 1, int(num_salt))
#               for i in image.shape]
#       out[coords] = 1

#       # Pepper mode
#       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
#       coords = [np.random.randint(0, i - 1, int(num_pepper))
#               for i in image.shape]
#       out[coords] = 0
#       return out
#    elif noise_typ == "poisson":
#       vals = len(np.unique(image))
#       vals = 2 ** np.ceil(np.log2(vals))
#       noisy = np.random.poisson(image * vals) / float(vals)
#       return noisy
#    elif noise_typ =="speckle":
#       row,col,ch = image.shape
#       gauss = np.random.randn(row,col,ch)
#       gauss = gauss.reshape(row,col,ch)        
#       noisy = image + image * gauss
#       return noisy

# im3 = noisy("gauss",im2)   
# plt.imshow(im3)
 
# with mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:

#     image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
#     plt.imshow(image)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(image)
    
#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     # if not results.multi_hand_landmarks:
#     #   continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       # print(
#       #     f'Index finger tip coordinates: (',
#       #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#       #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       # )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     # cv2.imwrite(
#         # '/tmp/annotated_image' + str(2) + '.png', cv2.flip(annotated_image, 1))
#     im2 = cv2.flip(annotated_image, 1)
#     plt.imshow(im2)

def manos_video3(frames):
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9) as hands:
        
        data = np.zeros(len(frames),dtype=object)
        right = np.zeros((21,3))
        left = np.zeros((21,3))
        insider = np.zeros(3,dtype=float)
        
        for i in range(len(frames)):
            im2 = frames[i]        
            # Convert the BGR image to RGB before processing.
            image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
            results = hands.process(image)
            insider[:] = np.nan
            
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
                                       results.multi_hand_landmarks[j].landmark[k].y,
                                       results.multi_hand_landmarks[j].landmark[k].z]
                         right[k,:] = insider[:]
                 elif str(results.multi_handedness[j]).find("Left")>1:
                     for k in range(21):
                         insider[:] = [results.multi_hand_landmarks[j].landmark[k].x,
                                       results.multi_hand_landmarks[j].landmark[k].y,
                                       results.multi_hand_landmarks[j].landmark[k].z]
                         left[k,:] = insider[:]
            data[i] = np.concatenate((left,right))
    return data


        
def manos_video2(frames):    
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9) as hands:
        
        data = np.zeros(len(frames),dtype=dict)
        right = np.zeros(21,dtype=dict)
        left = np.zeros(21,dtype=dict)
        
        for i in range(len(frames)):
            im2 = frames[i]        
            # Convert the BGR image to RGB before processing.
            image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
            results = hands.process(image)
            right = np.zeros(21,dtype=dict)
            left = np.zeros(21,dtype=dict)
            # right[:] = ({
            #                  'x': np.nan,
            #                  'y': np.nan,
            #                  'z': np.nan,
            #                  'd': True
            #              })
            if not results.multi_hand_landmarks:

                pass
            else:
                a = len(results.multi_hand_landmarks)
                if a > 2:
                    a = 2 
                for j in range(a):
                 if str(results.multi_handedness[j]).find("Right")>1:
                     for k in range(21):
                         right[k] = ({
                             'x': results.multi_hand_landmarks[j].landmark[k].x,
                             'y': results.multi_hand_landmarks[j].landmark[k].y,
                             'z': results.multi_hand_landmarks[j].landmark[k].z,
                             'd': True
                         })
                 elif str(results.multi_handedness[j]).find("Left")>1:
                     for k in range(21):
                         left[k] = ({
                             'x': results.multi_hand_landmarks[j].landmark[k].x,
                             'y': results.multi_hand_landmarks[j].landmark[k].y,
                             'z': results.multi_hand_landmarks[j].landmark[k].z,
                             'd': True
                         })
            data[i] = ({
                'R': right,
                'L': left
                })
    return data
    
def manos_video(frames):
    data = []
    data_right = []
    data_left = []
    
    with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
        for i in range(len(frames)):
            im2 = frames[i]        
            # Convert the BGR image to RGB before processing.
            image = cv2.flip(cv2.cvtColor(im2[:,:,0:3], cv2.COLOR_BGR2RGB), 1)
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                for i in range(21):
                            data_right.append({
                                'x': np.nan,
                                'y': np.nan,
                                'z': np.nan,
                                'd': False
                            }) 
                            data_left.append({
                                'x': np.nan,
                                'y': np.nan,
                                'z': np.nan,
                                'd': False
                            })
                data.append({
                    'R': data_right,
                    'L': data_left
                    })
                data_left = []
                data_right = [] 
               
            else:
                    if len(results.multi_hand_landmarks) > 2:
                        a = 2 
                    for j in range(a):
                     if str(results.multi_handedness[j]).find("Right")>1 and len(results.multi_hand_landmarks) == 1:
                         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                             data_right.append({
                                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                                 'z': results.multi_hand_landmarks[j].landmark[i].z,
                                 'd': True
                             })
                             data_left.append({
                                 'x': np.nan, 
                                 'y': np.nan, 
                                 'z': np.nan,
                                 'd': False
                             })
                     elif str(results.multi_handedness[j]).find("Left")>1 and len(results.multi_hand_landmarks) == 1:
                         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                             data_left.append({
                                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                                 'z': results.multi_hand_landmarks[j].landmark[i].z,
                                 'd': True
                             })
                             data_right.append({
                                 'x': np.nan, 
                                 'y': np.nan, 
                                 'z': np.nan,
                                 'd': False
                             })
                     elif str(results.multi_handedness[j]).find("Right")>1 and len(results.multi_hand_landmarks) == 2:
                         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                             data_right.append({
                                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                                 'z': results.multi_hand_landmarks[j].landmark[i].z,
                                 'd': True
                             })
                     elif str(results.multi_handedness[j]).find("Left")>1 and len(results.multi_hand_landmarks) == 2:
                         for i in range(len(results.multi_hand_landmarks[j].landmark)):
                             data_left.append({
                                 'x': results.multi_hand_landmarks[j].landmark[i].x,
                                 'y': results.multi_hand_landmarks[j].landmark[i].y,
                                 'z': results.multi_hand_landmarks[j].landmark[i].z,
                                 'd': True
                             })
                    if len(data_left)==0:
                        for i in range(len(data_left),21):
                            data_left.append({
                                'x': np.nan,
                                'y': np.nan,
                                'z': np.nan,
                                'd': False
                            })
                    if len(data_right)==0:
                        for i in range(len(data_right),21):
                            data_right.append({
                                'x': np.nan,
                                'y': np.nan,
                                'z': np.nan,
                                'd': False
                            })
                    data.append({
                        'R': data_right,
                        'L': data_left
                        })
                    data_left = []
                    data_right = []                            
        return data
    
datos = manos_video3(read_frames('001','004','004','theRealDataset\edit data'))

def neck_video(label, user, sample):
    
    frames = read_frames(label, user, sample,'theRealDataset\edit data')
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8) as pose:
        
            data_neck = np.zeros(len(frames),dtype=object)
            
            insider = np.zeros(3,dtype=float)
            
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
                    neck_z = (results.pose_landmarks.landmark[12].z + results.pose_landmarks.landmark[11].z)/2
                    insider[:] = [neck_x,neck_y,neck_z]
                    
                data_neck[i] = insider[:]

    return data_neck

neckkk = neck_video('001','004','004')

# def para_lin_data(data):
#     d = np.zeros(len(data))
    
#     ejes = ['x','y','z']
#     manos = ['R','L']
    
#     for manos in manos:
#         for ejes in ejes:    
#             for j in range(21):
#                 for i in range(len(data)):
#                     d[i] = data[i][manos][j][ejes]
#                 n = lin_reg(d)
#                 for i in range(len(data)):
#                     data[i][manos][j][ejes] = d[i]
#     return data

# b = para_lin_data(datos)

# for i in range(len(b)):
#     x = b[i]['R']

# plt.plot(y,x)

import json

# with open('data1.txt', 'w') as outfile:
    # json.dump(data, outfile)
