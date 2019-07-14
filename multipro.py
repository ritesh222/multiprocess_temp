#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:56:22 2019

@author: riteshkumar
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import multiprocessing
import time
import visualization_utils_color as vis_util
import six.moves.urllib as urllib
from io import StringIO

from worker1 import worker1
from worker2 import worker2
import tarfile


# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

category_index = np.load('config.npy',allow_pickle=True).item()

PATH_TO_CKPT_2='frozen_inference_graph_face.pb'
category_index_2 = {1: {'id': 2, 'name': 'face'}, 2: {'id': 1, 'name': 'background'}}


print("ID of main process: {}".format(os.getpid())) 

# creating processes 
p1 = multiprocessing.Process(target=worker1,args = [PATH_TO_FROZEN_GRAPH,category_index]) 
p2 = multiprocessing.Process(target=worker2,args = [PATH_TO_CKPT_2,category_index_2]) 

# starting processes 
p1.start() 
p2.start() 

# process IDs 
print("ID of process p1: {}".format(p1.pid)) 
print("ID of process p2: {}".format(p2.pid)) 

# wait until processes are finished 
p1.join() 
p2.join() 

# both processes finished 
print("Both processes finished execution!") 
  
    # check if processes are alive 
print("Process p1 is alive: {}".format(p1.is_alive())) 
print("Process p2 is alive: {}".format(p2.is_alive()))

