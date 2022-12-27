import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import math




__C = edict()
# We can get config by:
#    import config as cfg
cfg = __C

__C.Iou_1= 0.3
__C.Dist_1= 30
__C.Dist_2= 30
__C.Iou_2 = 0.3
__C.Iou_3 = 0.3
__C.Iou_s = 0.8
__C.Iou_i = 0.3

__C.heights = {'car': 1.5,
                        'bus': 3,
                        'truck': 2.5,
                        'pedestrian': 1.8,
                        'van': 2,
                        'group_of_pedestrians': 1.8,
                        'motorbike': 1.5,
                        'bicycle': 1.5,
                        'vehicle': 1.5
                        }
__C.colors = {'car': (1, 0, 0),
                       'bus': (0, 1, 0),
                       'truck': (0, 0, 1),
                       'pedestrian': (1.0, 1.0, 0.0),
                       'van': (1.0, 0.3, 0.0),
                       'group_of_pedestrians': (1.0, 1.0, 0.3),
                       'motorbike': (0.0, 1.0, 1.0),
                       'bicycle': (0.3, 1.0, 1.0),
                       'vehicle': (1.0, 0.0, 0.0)
                       }


"""
Radar, camera and LiDar extrinsic parameters
"""
__C.RadarT = np.array([0.0, 0.0, 0.0])
__C.RadarR = np.array([0.0, 0.0, 0.0])
__C.LidarT = np.array([0.0, 0.0, 0.0])
__C.LidarR = np.array([0.0, 0.0, 0.0])  
#Radar and Lidar have the same coordinate system "the reference for the three sensors" 
__C.CameraT = np.array([0.4593822, -0.0600343, 0.287433309324])
__C.CameraR = np.array([0.8493049332, 0.37113944, 0.000076230])

"""
Camera intrinsic parameters
"""
fxr = 337.873451599077
cxr = 338.530902554779
fyr = 329.137695760749
cyr = 186.166590759716
__C.Camera_mat = np.array([[fxr, 0, fyr],
                                    [0, cxr, cyr],
                                    [0,  0,  1]])

'''

   

    Ioui_1 : The threshold to create the intersection matrix
    Dist_1 : The eucledian distance to filter what we need to calculate for the intersection matrix
    Ioui_2 : The threshold of the intersection to mix the person and biycle classes with cyclist classes 
    Dist_2 : The distance to mix the person and biycle classes with cyclist classes 
    Iou_3  : The threshold to filter the confidence scores after their fusion
    Iou_s  : The upper threshold to fuse the bbox
    Iou_i  : The lower threshold to fuse the bbox


     Parametre = { 'Iou_1' : 0.3, 
                  'Dist_1' : 30,
                  'Dist_2' : 30,
                  'Iou_2' : 0.3, 
                  'Iou_3'  : 0.35,
                  'Iou_s'  : 0.8,
                  'Iou_i'  : 0.3 }
    
    '''