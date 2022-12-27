from __future__ import print_function


import cv2
import numpy as np
from scipy.spatial import distance
import cv2
import os
import glob
from numpy.lib.function_base import append
from sklearn.utils.linear_assignment_ import linear_assignment
import time

from tomlkit import boolean
np.random.seed(0)
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import distance
import argparse
from config_fusion import cfg
from ensemble_boxes import *

"""
Radar and lidar to camera transformation matrixes
"""
def RX(LidarToCamR):
    thetaX = np.deg2rad(LidarToCamR[0])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(thetaX), -np.sin(thetaX)],
                    [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float)
    return Rx

def RY(LidarToCamR):
    thetaY = np.deg2rad(LidarToCamR[1])
    Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                    [0, 1, 0],
                    [-np.sin(thetaY), 0, np.cos(thetaY)]])
    return Ry

def RZ(LidarToCamR):
    thetaZ = np.deg2rad(LidarToCamR[2])
    Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                    [np.sin(thetaZ), np.cos(thetaZ), 0],
                    [0, 0, 1]]).astype(np.float)
    return Rz

def transform(LidarToCamR, LidarToCamT):
    Rx = RX(LidarToCamR)
    Ry = RY(LidarToCamR)
    Rz = RZ(LidarToCamR)

    R = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]]).astype(np.float)
    R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

    LidarToCam = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                            [R[1, 0], R[1, 1], R[1, 2], 0.0],
                            [R[2, 0], R[2, 1], R[2, 2], 0.0],
                            [LidarToCamT[0], LidarToCamT[1], LidarToCamT[2], 1.0]]).T
    return LidarToCam

"""
Radar to camera transformation
"""
RadarToCameraT = cfg.RadarT - cfg.CameraT
RadarToCameraR = cfg.RadarR - cfg.CameraR
RadarToCamera = transform(
            RadarToCameraR, RadarToCameraT)

LidarToCameraT = cfg.LidarT - cfg.CameraT
LidarToCameraR = cfg.LidarR - cfg.CameraR
LidarToCamera = transform(
            RadarToCameraR, RadarToCameraT)
""""
Project Radar and Lidar data into camera reference
"""
def __get_projected_bbox(bb, rotation, cameraMatrix, extrinsic, obj_height=2):
    """get the projected boundinb box to some camera sensor
    """
    rotation = np.deg2rad(-rotation)
    res = 0.173611
    cx = bb[0] + bb[2] / 2
    cy = bb[1] + bb[3] / 2
    T = np.array([[cx], [cy]])
    pc = 0.2
    bb = [bb[0]+bb[2]*pc, bb[1]+bb[3]*pc, bb[2]-bb[2]*pc, bb[3]-bb[3]*pc]

    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]])

    points = np.array([[bb[0], bb[1]],
                        [bb[0] + bb[2], bb[1]],
                        [bb[0] + bb[2], bb[1] + bb[3]],
                        [bb[0], bb[1] + bb[3]],
                        [bb[0], bb[1]],
                        [bb[0] + bb[2], bb[1] + bb[3]]]).T

    points = points - T
    points = np.matmul(R, points) + T
    points = points.T

    points[:, 0] = points[:, 0] - 576
    points[:, 1] = 576 - points[:, 1]
    points = points * res

    points = np.append(points, np.ones(
        (points.shape[0], 1)) * -1.7, axis=1)
    p1 = points[0, :]
    p2 = points[1, :]
    p3 = points[2, :]
    p4 = points[3, :]

    p5 = np.array([p1[0], p1[1], p1[2] + obj_height])
    p6 = np.array([p2[0], p2[1], p2[2] + obj_height])
    p7 = np.array([p3[0], p3[1], p3[2] + obj_height])
    p8 = np.array([p4[0], p4[1], p4[2] + obj_height])
    points = np.array([p1, p2, p3, p4, p1, p5, p6, p2, p6,
                        p7, p3, p7, p8, p4, p8, p5, p4, p3, p2, p6, p3, p1])

    points = np.matmul(np.append(points, np.ones(
        (points.shape[0], 1)), axis=1), extrinsic.T)

    points = (points / points[:, 3, None])[:, 0:3]

    filtered_indices = []
    for i in range(points.shape[0]):
        if (points[i, 2] > 0 and points[i, 2] < 100):
            filtered_indices.append(i)

    points = points[filtered_indices]

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]

    xIm = np.round((fx * points[:, 0] / points[:, 2]) + cx).astype(np.int)
    yIm = np.round((fy * points[:, 1] / points[:, 2]) + cy).astype(np.int)

    proj_bbox_3d = []
    for ii in range(1, xIm.shape[0]):
        proj_bbox_3d.append([xIm[ii], yIm[ii]])
    proj_bbox_3d = np.array(proj_bbox_3d)

    return proj_bbox_3d

def project_3D_bboxes_to_camera(annotations, intrinsict, extrinsic):
    """method to project the bounding boxes to the camera

    :param annotations: the annotations for the current frame
    :type annotations: list
    :param intrinsict: intrisic camera parameters
    :type intrinsict: np.array
    :param extrinsic: extrinsic parameters
    :type extrinsic: np.array
    :return: dictionary with the list of bbounding boxes with camera coordinate frames
    :rtype: dict
    """
    bboxes_3d = []
    for object in annotations:
        obj = {}
        class_name = object['class_name']
        obj['class_name'] = class_name
        obj['id'] = (object['id'] if 'id' in object.keys() else 0)
        height = cfg.heights[class_name]
        bb = object['bbox']['position']
        rotation = object['bbox']['rotation']
        bbox_3d = __get_projected_bbox(
            bb, rotation, intrinsict, extrinsic, height)
        obj['bbox_3d'] = bbox_3d
        bboxes_3d.append(obj)

    return bboxes_3d
def projrct_2D_bbox_cam(image, bboxes_3d, scores, radar_labels, pc_size1=0.78, pc_size2=0.95):
    """diplay pseudo 2d bounding box from camera

    :param image: camera which the bounding box is going to be projected
    :type image: np.array
    :param bboxes_3d: list of bounding box information with pseudo-3d image coordinate frame
    :type bboxes_3d: dict
    :param pc_size: percentage of the size of the bounding box [0.0 1.0]
    :type pc_size: float
    :return: camera image with the correspondent bounding boxes
    :rtype: np.array
    """
    radar_boxe = []
    j=0
    final_label=[]
    final_score=[]
    for obj in bboxes_3d:
        bb = np.zeros((4))
        if obj['bbox_3d'].shape[0] > 0:
            bb[0] = np.min(obj['bbox_3d'][:, 0])
            bb[1] = np.min(obj['bbox_3d'][:, 1])
            bb[2] = np.max(obj['bbox_3d'][:, 0])
            bb[3] = np.max(obj['bbox_3d'][:, 1])
            wid = bb[2] - bb[0]
            hei = bb[3] - bb[1]
            bb[0] += wid*(1.0 - pc_size1-0.15)
            bb[2] -= wid*(1.0 - pc_size1+0.1)
            bb[1] += hei*(1.0 - pc_size2-0.1)
            bb[3] -= hei*(1.0 - pc_size2+0.1)
            bb = bb.astype(np.int)
            radar_boxe.append([bb[0], bb[1], bb[2], bb[3]])
            final_label.append(radar_labels[j])
            final_score.append(scores[j])
        j+=1
    return radar_boxe,final_label,final_score

class Extract_radar_LiDAR_data(object):
    """
    Represent a 2D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        #self.class_camera = int 
        self.Score_camera = data[1]
        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[2]  # left
        self.ymin = data[3]  # top
        self.xmax = data[4]  # right
        self.ymax = data[5]  # bottom
        self.box2d_camera = [self.xmin, self.ymin, self.xmax, self.ymax]

class Extract_camera(object):
    """
    Represent a 2D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        #self.class_camera = int 
        self.Score_camera = data[1]
        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[2]  # left
        self.ymin = data[5]  # top
        self.xmax = data[4]  # right
        self.ymax = data[3]  # bottom
        self.box2d_camera = [self.xmin, self.ymin, self.xmax, self.ymax]

def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size

         return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)   

def score_fusion (A,B):
    


    '''
    A : Camera score
    B : Lidar score
    Confidence score fusion using :
    Dempster-Shafer Sensor Fusion using Distance Function

    '''

    Score_camera_1 = A
    S_camera = 1-A
    Score_camera_2 = S_camera/2
    Score_camera_3 = S_camera/2

    Score_lidar_1 = B
    S_ldiar = 1 - B
    Score_lidar_2 = S_ldiar/2
    Score_ldiar_3 = S_ldiar/2

    '''
    We couldn't have the classifier vector, so we decided to put the same score if 
    the socre is greater than the socre and leave it as it is if it is different.
    
    '''

    if B == 0 and A >= 0.35 : 

        Score_lidar_1 = A
        S_ldiar = 1 - A
        Score_lidar_2 = S_ldiar/2
        Score_ldiar_3 = S_ldiar/2

    elif B == 0 and  A < 0.35 :
    
        Score_lidar_1 = B
        S_ldiar = 1 - B
        Score_lidar_2 = S_ldiar/2
        Score_ldiar_3 = S_ldiar/2


    camera = np.array((Score_camera_1,Score_camera_2,Score_camera_3),dtype=np.float16)
    lidar = np.array((Score_lidar_1,Score_lidar_2,Score_ldiar_3),dtype=np.float16)
    Info_mat = np.array((camera,lidar),dtype=np.float16)
    #step01
    Info_mat= np.array((camera,lidar),dtype=np.float16)


    #step02
    Alpha = 1

    #step03
    x,y= Info_mat.shape

    distt=np.zeros((2,2), dtype=float)

    dist = np.zeros_like(Info_mat,dtype=float)

    for i in range (0,2) : 
        for j in range ( 0,2) :
            #print('i,j',i,j)
            distt[i,j] = (distance.euclidean(Info_mat [i],Info_mat[j]))
 

    for i,trk in enumerate (Info_mat):
        for j,det in enumerate (Info_mat):       
            dist[i,j] = (distance.euclidean(Info_mat [i],Info_mat[j]))
        

    #step4 :Create similarity matrix

    SIM = (np.ones_like(distt,dtype=float)) - distt


    #Step 5: Create Supplementary matrix or vector.

    array = np.array(SIM)
    np.fill_diagonal(array, 0)
    Sup = array.sum(1)


    #Step 6: Create credibility matrix or vector.

    summe = Sup.sum(0)
    #print('somme',summe)
    Crd = Sup / summe

    #Step 7: Modify the original evidence.

    Row_0=[row[0] for row in Info_mat]
    Row_1=[row[1] for row in Info_mat]
    Row_2=[row[2] for row in Info_mat]
    A=sum(x * y for x, y in zip(Row_0, Crd))
    B=sum(x * y for x, y in zip(Row_1, Crd))
    C=sum(x * y for x, y in zip(Row_2, Crd))
  

    I= np.array((A,B,C),dtype=np.float16)
    
    K = 0
    for i in range (0,3) : 
        for j in range ( 0,3) : 
            if i != j : 
                K = K + (I[i]*I[j])
              
   
    S1=(I[0]*I[0])/(1-K)
    S2=(I[1]*I[1])/(1-K)
    S3=(I[2]*I[2])/(1-K)

    
    Final_score = round(S1, 2)
 
    Final_score = round(((I[0]*I[0])/(1-K)), 2)
    

    return  Final_score


def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    print(bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2])
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'x='+str((left+right)/2)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img  


def convertBack(x, y, w, h): 
    """
    # 2. Converts center coordinates to rectangle coordinates     
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def initialization_based(a): 
    
    '''
    Take the maximum value on each Row
    '''

    idx = a.argmax(axis=1)
    print(idx)
    out = np.zeros_like(a,dtype=float)
    print(np.arange(a.shape[0]),idx)
    out[np.arange(a.shape[0]), idx] = 1
    return out


def dist(boxA,boxB):

    '''
    Dsitance euclidean bitween 2 bbox
    '''
    x1 = (boxA[0] + boxA[2])/2.0
    y1 = (boxA[1] + boxA[3])/2.0
    x2 = (boxB[0] + boxB[2])/2.0
    y2 = (boxB[1] + boxB[3])/2.0
    a=[]
    b=[]
    a=(x1,y1)
    b=(x2,y2)
    #print('centroide',a,b)
    dist = distance.euclidean(a, b)
    return dist


def image_résolution (boxA,w,h):


    '''
    Resize the bbox 
    
    '''

    if  boxA[0] < 0 :
        boxA[0]  = 0
    else : boxA [0]

    if  boxA[2] > h :
        boxA[2]  = h
    else : boxA [2]

    if  boxA[1] < 0 :
        boxA[1]  = 0
    else : boxA [1]
   
    if  boxA[3] > w :
        boxA[3]  = w
    else : boxA [3]
        
    box=[boxA[0],boxA[1],boxA[2],boxA[3]]

    return box


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    print("xA",xA)
    print("xB",xB)
    print("yA",yA)
    print("yB",yB)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def union(boxA, boxB):
    
  '''
  Union bitween 2 bbox

  
  '''
  xA = min(boxA[0], boxB[0])
  yA = min(boxA[1], boxB[1])
  xB = max(boxA[2], boxB[2])
  yB = max(boxA[3], boxB[3])

 
  return xA,yA,xB,yB

def overlap(box1, box2):
 
 '''
 Overlap bitween 2 bbox

 '''
 x1 = max(box1[0], box2[0])
 y1 = max(box1[1], box2[1])
 x2 = min(box1[2], box2[2])
 y2 = min(box1[3], box2[3])
    
 return x1,y1,x2,y2


def Data_Association(Box_information_sensor1 , Box_information_sensor2, iou_thrd ,Dist):

        boxes_sensor1 = Box_information_sensor1[0]
        classIds_sensor1 = Box_information_sensor1[2]
        img = Box_information_sensor1[3]

        boxes_sensor2 = Box_information_sensor2[0]
        classIds_sensor2 = Box_information_sensor2[2]
        

        IOU_mat= np.zeros((len(boxes_sensor1),len(boxes_sensor2)),dtype=np.float32)
        img_width,img_height=img.shape[:2]
        print(img_height,img_width)
        print('boxes_sensor1',boxes_sensor1)
        print('Number of sensor1 detection', len(boxes_sensor1))
        
        print('classIds_sensor1',classIds_sensor1)
        print('boxes_sensor2',boxes_sensor2)
        print('Number of sensor2 detection', len(boxes_sensor2))
        
        print('classIds_LiDAR',classIds_sensor2)
       
        for t,trk in enumerate(boxes_sensor1):

            for d,det in enumerate(boxes_sensor2):
            
                trk=image_résolution(trk,img_width,img_height)#### #### Adjust the image size so that the image does not go outside the frame
                det=image_résolution(det,img_width,img_height) #### // // // 
                print('isammmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm',classIds_sensor2[d], classIds_sensor1[t])
                if classIds_sensor2[d] == classIds_sensor1[t] and dist(trk,det) < Dist:

                    '''
                    Buil the matrix of intersection_over_union : 
                    To optimize the calculations, we filter by class and by distance (the Euclidean distance ) 
                    between the centroids of each bbox

                    ''' 
                    print('interrrrrrrrr', bb_intersection_over_union(trk,det))  
                    IOU_mat[t,d] = bb_intersection_over_union(trk,det)  # Buil the matrix of intersection_over_union
                else : 
                    IOU_mat[t,d] = 0       
        Iou=(IOU_mat)
        print('IOU',Iou)
        if min(Iou.shape) > 0:
            a = (Iou > iou_thrd).astype(np.int32) #matrix a : put 1 if the score is greater than the threshold
            if a.sum(1).max() == 1 and a.sum(0).max() == 1 :
                '''
                 after having the matrix 'a' we will check the matches
                 for the two detection: we will check if each row and column
                 contains a single '1', calculating the sum over all rows and columns
                 and check their max == 1
                '''
                matched_idx = np.stack(np.where(a), axis=1) # If the condition is true, we take each box that contains the 1 for a match

            elif    a.sum(1).max() > 1 and a.sum(0).max() == 1 :
                    matched_idx = np.stack(np.where(a), axis=1) 
                    print('MESSAGE: There are overlapping detections for camera')
                    #print(initialization_based (IOU_mat))
                    '''
                    The condition [a.sum(1).max()] is not respected on the rows
                    '''
            else : 
                    a.sum(1).max() == 1 and a.sum(0).max() > 1
                    matched_idx = np.stack(np.where(a), axis=1) 
                    print('MESSAGE: There are overlapping detections for Lidar')  
                    '''
                    The condition [a.sum(1).max()] is not respected on the columns
                    '''     
        else:
            matched_idx = np.empty(shape=(0, 2))


        ##### find the unmatched_detections #######

        '''
        we check in the list of matches, the detection that is not present
        '''
        unmatched_detections_sensor2 = []
        for t, trk in enumerate(boxes_sensor2):
            if (t not in matched_idx[:, 1]):
                unmatched_detections_sensor2.append(t)
        print('unmatched_detections_sensor2',unmatched_detections_sensor2)
        
                
        unmatched_detections_sensor1 = []
        for d, det in enumerate(boxes_sensor1):
            if (d not in matched_idx[:, 0]):
                unmatched_detections_sensor1 .append(d)
        print('unmatched_detections_sensor1',unmatched_detections_sensor1)


        


        Assosciation = matched_idx, unmatched_detections_sensor1, unmatched_detections_sensor2, Iou

        return Assosciation 


def Bbox_Score_fusion (Update_class, Data1, Data2,Iou_score_fusion): 


    ###################################################################

                    ######## Fusion step ###########

    ###################################################################


        boxes_Data1 = Data1[0]
        confidences_score_Data1 = Data1[1]
        classIds_Data1 = Data1[2]
        img = Data1[3]
        img_width,img_height=img.shape[:2]

        boxes_Data2 = Data2[0]
        confidences_score_Data2 = Data2[1]
        classIds_Data2 = Data2[2]

        matched_idx = Update_class [0]
        unmatched_detections_Data1 = Update_class [1]
        unmatched_detections_Data2 = Update_class [2]
        Iou = Update_class [3]
        IOU_mat = Iou

        box_fusion = []
        
        Score_fusion = []

        class_fusion = []

        box_fusion_unmathed_Data1 = []
        score_fusion_unmathed_Data1 = []

        box_fusion_unmathed_Data2 = []
        score_fusion_unmathed_Data2 = []

################################################################Fuison of matched detections##############################################################

        if len(matched_idx) > 0 :
            for m in matched_idx:
                boxes_list = []; scores_list = []; labels_list = [] 
                boxes_rad = []; scores_rad = []; labels_rad = [] 
                boxes_cam = []; scores_cam = []; labels_cam = []   
                cam_idx = m[0]
                lid_idx = m[1]
                class_fusion.append(classIds_Data2[lid_idx])
                boxes_cam.append([boxes_Data1[cam_idx][0]/672,boxes_Data1[cam_idx][1]/376,boxes_Data1[cam_idx][2]/672,boxes_Data1[cam_idx][3]/376])
                scores_cam.append(confidences_score_Data1[cam_idx])
                labels_cam.append(1)
                boxes_list.append(boxes_cam)
                scores_list.append(scores_cam)
                labels_list.append(labels_cam)
                if bb_intersection_over_union(boxes_Data2[lid_idx], boxes_Data1[cam_idx])==0:
                    boxes_rad.append([boxes_Data2[lid_idx][0]/672,boxes_Data2[lid_idx][1]/376,boxes_Data2[lid_idx][2]/672,boxes_Data2[lid_idx][3]/376])
                    scores_rad.append(confidences_score_Data2[lid_idx])
                    labels_rad.append(1)
                    scores_list.append(scores_rad)
                    boxes_list.append(boxes_rad)
                    labels_list.append(labels_rad)

                    box_fusion_matched, _, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=0.2, skip_box_thr=0.01)
                    box_fusion.append([box_fusion_matched.tolist()[0][0]*672,box_fusion_matched.tolist()[0][1]*376,box_fusion_matched.tolist()[0][2]*672,box_fusion_matched.tolist()[0][3]*376])
                else :
                    box_fusion.append([boxes_cam[0][0]*672,boxes_cam[0][1]*376,boxes_cam[0][2]*672,boxes_cam[0][3]*376])                    
                Score_fusion.append(score_fusion(confidences_score_Data1[cam_idx],confidences_score_Data2[lid_idx]))

            print('matched', box_fusion)   
        else : 
            len(matched_idx) == 0
            print('No matched detection')

################################################################Add unmatched detections##############################################################
        for m in unmatched_detections_Data1 : 
            if classIds_Data1[m] == 'Car' or classIds_Data1[m] == 'Pedestrian' : 
                box_fusion.append(boxes_Data1[m]) 
                class_fusion.append(classIds_Data1[m])
                S_fusion_unmatched_camera = score_fusion(confidences_score_Data1[m],0) 
                Score_fusion.append(S_fusion_unmatched_camera) 
                box_fusion_unmathed_Data1.append(boxes_Data1[m]) 
                score_fusion_unmathed_Data1.append(S_fusion_unmatched_camera) 
        

            else : 
            
                box_fusion.append(boxes_Data1[m])
                class_fusion.append(classIds_Data1[m])
                Score_fusion.append(confidences_score_Data1[m])
                box_fusion_unmathed_Data1.append(boxes_Data1[m])
                score_fusion_unmathed_Data1.append(confidences_score_Data1[m])

        for m in unmatched_detections_Data2 : 
            add=True
            for t in unmatched_detections_Data1 : 
                if bb_intersection_over_union(boxes_Data2[m], boxes_Data1[t])>0:
                    add=False

            for t in matched_idx : 
                if bb_intersection_over_union(boxes_Data2[m], boxes_Data1[t[0]])>0:
                    add=False
            if add==True:
                if classIds_Data2[m] == 'Car' or classIds_Data2[m]== 'Pedestrian' : 
                    box_fusion.append(boxes_Data2[m])
                    class_fusion.append(classIds_Data2[m])
                    S_fusion_unmatched_lidar = score_fusion(confidences_score_Data2[m],0)
                    Score_fusion.append(S_fusion_unmatched_lidar)
                    box_fusion_unmathed_Data2.append(boxes_Data2[m])
                    score_fusion_unmathed_Data2.append(S_fusion_unmatched_lidar)
                else:
                    box_fusion.append(boxes_Data2[m])
                    class_fusion.append(classIds_Data2[m])
                    Score_fusion.append(confidences_score_Data2[m])
                    box_fusion_unmathed_Data2.append(boxes_Data2[m])
                    score_fusion_unmathed_Data2.append((confidences_score_Data2[m]))
                     
        print('unmatchedlidar', unmatched_detections_Data2)
        print('unmatchedcamera', unmatched_detections_Data1)

##############################################################################################################################################################
        #### Threshold ###
        score_fusion_after_threshold =[]
        box_fusion_after_threshold =[]
        class_fusion_after_threshold =[]
        if len(Score_fusion)> 0 :
                for idx, value in enumerate(box_fusion):
                    if Score_fusion[idx] > Iou_score_fusion :
                       print(idx)
                       score_fusion_after_threshold.append(Score_fusion[idx])
                       box_fusion_after_threshold.append(value)
                       class_fusion_after_threshold.append(class_fusion[idx])

        print('box_fusion',len(box_fusion))
        print('Score_fusion',len(Score_fusion))
        print('Class_fusion',len(class_fusion))
        print('score_fusion_after_threshold',len(score_fusion_after_threshold))
        print('box_fusion_after_threshold',len(box_fusion_after_threshold))
        print('class_fusion_after_threshold',len(class_fusion_after_threshold))


        

        Data_fusion = box_fusion_after_threshold, score_fusion_after_threshold, class_fusion_after_threshold, img

        
    
        return Data_fusion 



def radar_preprocessing (img,path_radar):

    '''
    Function to extract information from radar and prject them on camera

    '''
    # Load names of classes
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    sl = slice(0,-4) #extract a section from a string
    filenames = []
    filenames.append(img[sl])
    for image in filenames:
        textfilename = image+".txt"
        print('name of image',textfilename)
    
        img_path_radar = path_radar+image+".png"
        

        frame_orig_radar = cv2.imread(img_path_radar)
        

        radar_path = path_radar + textfilename
        

        img_height, img_width = frame_orig_radar.shape[:2]
        lines = [line.rstrip() for line in open(radar_path)]
        objects = [Extract_radar_LiDAR_data(line) for line in lines]
        box_radar =[]
        score_confiance_radar = []
        class_radar = []
        objs = []
        for obj in objects: 
            box_radar.append(obj.box2d_camera)
            bb, angle = obj.box2d_camera, 0
            objs.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})
            score_confiance_radar.append(obj.Score_camera)
            A=int(obj.type)
            class_radarr = classes[A]
            class_radar.append(class_radarr)
        bboxes_cam = project_3D_bboxes_to_camera(objs,cfg.Camera_mat,RadarToCamera)
        radar_boxe,final_label,final_score = projrct_2D_bbox_cam(frame_orig_radar, bboxes_cam, score_confiance_radar,class_radar)   

    Box_information_radar = radar_boxe,final_score, final_label, frame_orig_radar
    return Box_information_radar

def lidar_preprocessing (img,path_lidar):

    '''
    Function to extract information from radar and prject them on camera

    '''
    # Load names of classes
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    sl = slice(0,-4) #extract a section from a string
    filenames = []
    filenames.append(img[sl])
    for image in filenames:
        textfilename = image+".txt"
        print('name of image',textfilename)
    
        img_path_lidar = path_lidar+image+".png"
        

        frame_orig_lidar = cv2.imread(img_path_lidar)
        

        lidar_path = path_lidar + textfilename
        

        img_height, img_width = frame_orig_lidar.shape[:2]
        lines = [line.rstrip() for line in open(lidar_path)]
        objects = [Extract_radar_LiDAR_data(line) for line in lines]
        box_lidar =[]
        score_confiance_lidar = []
        class_lidar = []
        objs = []
        for obj in objects: 
            box_lidar.append(obj.box2d_camera)
            bb, angle = obj.box2d_camera, 0
            objs.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})
            score_confiance_lidar.append(obj.Score_camera)
            A=int(obj.type)
            class_lidarr = classes[A]
            class_lidar.append(class_lidarr)
        
        bboxes_cam = project_3D_bboxes_to_camera(objs,cfg.Camera_mat,LidarToCamera)
        lidar_boxe,final_label,final_score = projrct_2D_bbox_cam(frame_orig_lidar, bboxes_cam, score_confiance_lidar,class_lidar)   

    Box_information_lidar = lidar_boxe, final_score, final_label, frame_orig_lidar
    return Box_information_lidar



def camera_preprocessing (img,path_camera):

        '''
        Function to extract information from camera
    
        '''

        # Load names of classes
        classesFile = "coco.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        sl = slice(0,-4) #extract a section from a string
        filenames = []
        filenames.append(img[sl])
        for image in filenames:
            textfilename = image+".txt"
            print('name of image',textfilename)
        
            img_path_camera = path_camera+image+".png"
            

            frame_orig_camera = cv2.imread(img_path_camera)
            

            camera_path = path_camera + textfilename
           

            img_height, img_width = frame_orig_camera.shape[:2]
            lines = [line.rstrip() for line in open(camera_path)]
            objects = [Extract_camera(line) for line in lines]
            box_camera =[]
            score_confiance_camera = []
            class_camera = []
            for obj in objects: 
                box_camera.append(obj.box2d_camera)
                score_confiance_camera.append(obj.Score_camera)
                A=int(obj.type)
                class_cameraa = classes[A]
                class_camera.append(class_cameraa)
            

        Box_information_camera = box_camera,score_confiance_camera,class_camera,frame_orig_camera


        return Box_information_camera 

def fuse_class_Cycliste (Assosciation, Data1, Data2, Dist_1, Iou):


    #Fuse Cyclist with Pedestrain + Bicycle 


    '''
         As we know, we just have two classes in common, so we came up with the idea
         to merge the two classes of the Pedestrain + Bicycle camera with the class
         cyclist so when we have the two detection for the camera we take them for
         cyslite and all that using 3 conditions : 

         Euclidean distance : boxes_camera [ Pedestrian]  / boxes_lidar [Cyclist] > Dist_1
         Intersection over union : boxes_camera [Bicycle] / boxes_camera [Pedestrian] > Iou 
         Intersection over union : boxes_camera [Pedestrian] / boxes_lidar [Cyclist] >  Iou

    '''    

    matched_idx = Assosciation [0]
    unmatched_detections_senosr1 = Assosciation [1]
    unmatched_detections_sensor2 = Assosciation [2]
    Iou = Assosciation [3]

    
    boxes_sensor1 = Data1[0]
    score_confiance_sensor1 = Data1[1]
    classIds_sensor1 = Data1[2]
    img = Data1[3]

    boxes_sensor2 = Data2[0]
    score_confiance_sensor2 = Data2[1]
    classIds_sensor2 = Data2[2]

    list_to_remove = []

    if 'Cyclist' in classIds_sensor1 : 
            print('trueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            print('detection_cyclist',classIds_sensor2)
            for x,xx in enumerate(unmatched_detections_senosr1):
                for z,zz in enumerate (unmatched_detections_senosr1):
                    
                    if classIds_sensor1[xx] == 'Cyclist' and classIds_sensor1[zz] == 'Pedestrian' and  dist(boxes_sensor1[zz],boxes_sensor1[xx]) < Dist_1 and bb_intersection_over_union(boxes_sensor1[xx],boxes_sensor1[zz]) > 0.35 : 
                        list_to_remove.append(xx)
                        #list_to_remove.append(zz)
            unmatched_detections_senosr1 = list(set(unmatched_detections_senosr1) - set(list_to_remove))
            print(list_to_remove)                
            print('flitrage_for_Cyclist',unmatched_detections_senosr1)

    print('match',matched_idx)
    
    Update_class_sensor2_sensor1  = matched_idx,unmatched_detections_senosr1,unmatched_detections_sensor2,Iou

    
    return Update_class_sensor2_sensor1

def draw_bbox (Data_fusion, path): 
    
    boxes_camera = Box_information_camera[0]
    confidences_score_camera = Box_information_camera[1]
    classIds_camera = Box_information_camera[2]

    boxes_lidar = Box_information_lidar[0]
    confidences_score_lidar = Box_information_lidar[1]
    classIds_lidar = Box_information_lidar[2]
        

    boxes_radar = Box_information_radar[0]
    confidences_score_radar= Box_information_radar[1]
    classIds_radar = Box_information_radar[2]
    


    box_fusion = Data_fusion [0]
    Score_fusion = Data_fusion [1]
    img = Data_fusion [3]


    org = img.copy()
    
    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 2

    for (start_x, start_y, end_x, end_y), confidence in zip(boxes_camera, confidences_score_camera):
                    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
                    cv2.rectangle(img, (int(start_x), int(start_y) - (2 * baseline + 5)), (int(start_x) + w, int(start_y)), (255, 255, 0), -1)
                    cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 0,0), 2)
                    cv2.putText(img, str(confidence), (int(start_x), int(start_y)), font, font_scale, (0, 0, 0), thickness)

    for (start_x, start_y, end_x, end_y), confidence in zip(boxes_radar, confidences_score_radar):
                    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
                    cv2.rectangle(img, (int(start_x), int(start_y) - (2 * baseline + 5)), (int(start_x) + w, int(start_y)), (255, 255, 0), -1)
                    cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 0,255), 2)
                    cv2.putText(img, str(confidence), (int(start_x), int(start_y)), font, font_scale, (0, 0, 0), thickness)
    
    for (start_x, start_y, end_x, end_y), confidence in zip(boxes_lidar, confidences_score_lidar):

                    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
                    cv2.rectangle(img, (int(start_x), int(start_y) - (2 * baseline + 5)), (int(start_x) + w, int(start_y)), (255, 255, 0), -1)
                    cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255,0), 2)
                    cv2.putText(img, str(confidence), (int(start_x), int(start_y)), font, font_scale, (0, 0, 0), thickness)

    for (start_x, start_y, end_x, end_y), confidence in zip(box_fusion, Score_fusion):
                    
                    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
                    cv2.rectangle(org, (int(start_x), int(start_y) - (2 * baseline + 5)), (int(start_x) + w, int(start_y)), (255, 255, 0), -1)
                    cv2.rectangle(org, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (100, 255,255), 2)
                    cv2.putText(org, str(confidence), (int(start_x), int(start_y)), font, font_scale, (0, 0, 0), thickness)



    cv2.imshow('Image_before_fusion',img )
    cv2.waitKey(0)   
    cv2.imshow('Image_after_fusion', org)
    cv2.waitKey(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path_camera', help='Path to camera')
    parser.add_argument('path_radar',help='Path to radar')
    parser.add_argument('path_lidar',help='Path to lidar')
    parser.add_argument('-b','--draw_bbox', help='Draw bbox / optional')
    parser.add_argument('Camera', type=bool, default=False, help='Use Camera')
    parser.add_argument('Radar', type=bool, default=False, help='Use Radar')
    parser.add_argument('LiDAR', type=bool, default=False, help='Use LiDAR')
    args = parser.parse_args()

   
    print('Processsing input(s)...')
    args.Radar = True
    print(args.Camera)
    print(args.Radar)
    print(args.LiDAR)

    direcc = args.path_camera + "*.png"
    imgs = glob.glob(direcc)
    
    for img in imgs:   
        img = img[img.rfind('/'):]
        Path = img[1:-3]
        Box_information_camera = camera_preprocessing (img,args.path_camera) #Function to extract information from camera
        Box_information_radar = radar_preprocessing(img,args.path_radar) #Function to extract information from radar
        Box_information_lidar = lidar_preprocessing(img,args.path_lidar) #Function to extract information from lidar
        
        if args.Camera and args.LiDAR and args.Radar:
            """
            Camra radar fusion
            """
            Assosciation = Data_Association(Box_information_camera ,Box_information_radar, cfg.Iou_1 , cfg.Dist_1)
            Data_fusion = Bbox_Score_fusion (Assosciation, Box_information_camera, Box_information_radar,cfg.Iou_3)
            
            """
            Fuse the lidar with the fusion result
            """
            
            Assosciation = Data_Association(Data_fusion ,Box_information_radar, cfg.Iou_1 , cfg.Dist_1)
            Update_class = fuse_class_Cycliste (Assosciation, Data_fusion,Box_information_radar,cfg.Dist_2,cfg.Iou_2)
            Data_fusion = Bbox_Score_fusion (Update_class, Data_fusion, Box_information_radar,cfg.Iou_3)

        elif args.LiDAR and args.Radar:
            """
            LiDAR radar fusion
            """
            Assosciation = Data_Association(Box_information_lidar ,Box_information_radar, cfg.Iou_1 , cfg.Dist_1)
            Data_fusion = Bbox_Score_fusion (Assosciation, Box_information_lidar, Box_information_radar,cfg.Iou_3)

        elif args.Camera and args.Radar:
            """
            Camra radar fusion
            """
            Assosciation = Data_Association(Box_information_camera ,Box_information_radar, cfg.Iou_1 , cfg.Dist_1)
            Data_fusion = Bbox_Score_fusion (Assosciation, Box_information_camera, Box_information_radar,cfg.Iou_3)

        elif args.LiDAR and args.Camera:
            """
            Camra LiDAR fusion
            """
            Assosciation = Data_Association(Box_information_camera ,Box_information_lidar, cfg.Iou_1 , cfg.Dist_1)
            Data_fusion = Bbox_Score_fusion (Assosciation, Box_information_camera, Box_information_lidar,cfg.Iou_3)
        else:
            print("No fusion")

        if args.draw_bbox == 'Draw_bbox' : 
            print(Path)
            draw_bbox(Data_fusion, img[1:-3])
