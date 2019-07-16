#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:23:32 2019

@author: riteshkumar
"""
import os
import sys

import numpy as np
import cv2
import multiprocessing
import time
import visualization_utils_color as vis_util
import six.moves.urllib as urllib
from io import StringIO

def worker2(PATH_TO_CKPT_2,category_index_2): 
    # printing process id 
    print("ID of process running worker2: {}".format(os.getpid()))
    import tensorflow as tf
    cap = cv2.VideoCapture("Roller.mp4")
    if cap is None:
        print('Video file not found')
        
    out = None

    detection_graph_2 = tf.Graph()
    with detection_graph_2.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_2, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    config2 = tf.ConfigProto()
    #config2.gpu_options.allow_growth = True
    config2.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess2= tf.Session(graph=detection_graph_2, config=config2)
    frame_num = 1490;
    while frame_num:
        frame_num -= 1
        ret, image = cap.read()
        if ret == 0:
            break
    
        if out is None:
            [h, w] = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter("test_out_m_2.avi", fourcc, 25.0, (w, h))
    
    
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph_2.get_tensor_by_name('image_tensor:0')
                  # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph_2.get_tensor_by_name('detection_boxes:0')
                  # Each score represent how level of confidence for each of the objects.
                  # Score is shown on the result image, together with the class label.
        scores = detection_graph_2.get_tensor_by_name('detection_scores:0')
        classes = detection_graph_2.get_tensor_by_name('detection_classes:0')
        #print(np.squeeze(classes))
        num_detections = detection_graph_2.get_tensor_by_name('num_detections:0')
                  # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess2.run(
                      [boxes, scores, classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                  
                  #print('inference time cost: {}'.format(elapsed_time))
                  #print(boxes.shape, boxes)
                  #print(scores.shape,scores)
                  #print(classes.shape,classes)
                  #print(num_detections)
                  # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                #          image_np,
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index_2,
                      use_normalized_coordinates=True,
                      line_thickness=4,
                      )
                      
        elapsed_time = time.time() - start_time
        print('pid: {} inference time cost: {}'.format(os.getpid(),elapsed_time))
              
        out.write(image)

    print("Done")
    cap.release()
    out.release()
    sess2.close()
