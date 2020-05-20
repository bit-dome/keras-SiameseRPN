from __future__ import division, print_function, absolute_import

import cv2
import os
import numpy as np
import time
import argparse

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from build import Siamese_RPN
from config import Tracker_config





class Myconfig(Tracker_config):
    
    instance_size = (271,271,3)
    template_size = (127,127,3)
    
    batch_size = 1
    
config = Myconfig()

cv2.namedWindow('webcam' , cv2.WINDOW_NORMAL)
cv2.resizeWindow('webcam', 640, 480)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)


FPS_all = []

def init(img, xy, wh):
    config.outputs['target_xy'] = np.array(xy)
    config.outputs['target_wh'] = np.array(wh)
  
    network1 = Siamese_RPN(mode='inference_init', config=config, model_dir=None)        
    network1.keras_model.load_weights('pretrained/baseline.h5', by_name=True)
    
    network1.inference_init(img)
    
    del network1
    
   
    network2 = Siamese_RPN(mode = 'inference',config = config, model_dir = None)
    network2.keras_model.load_weights('pretrained/baseline.h5', by_name=True)
    return network2

def main():
    
    # Select ROI.
    ret, img = cap.read()
  
    x, y, w, h = cv2.selectROI('webcam', img, False, False) 
    
           
    # init model and first frame. 
    net = init(img, xy=[x+w//2, y+h//2], wh=[w, h])
    


    while True:
        ret, img = cap.read()
    
    
        t1 = time.time()
        xy, wh, scores = net.predict(img)        
        t2 = time.time() - t1
        
        
        
        x1  = xy[0] - wh[0]//2
        y1 = xy[1] - wh[1]//2
        x2 = x1 + wh[0]
        y2 = y1 + wh[1]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        # print('Outputs:',(int(x1),int(y1)),(int(x2),int(y2)),scores)
        
        FPS_all.append(1/t2)
        FPS = sum(FPS_all)/len(FPS_all)
        cv2.putText(img,"FPS: {:.0f}".format(FPS),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),1)
        cv2.imshow('webcam',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
        if len(FPS_all) >= 50:
            FPS_all.pop(0)
        
        
        
if __name__=='__main__':
    main()



