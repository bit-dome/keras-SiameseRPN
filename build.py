from __future__ import division, print_function, absolute_import

from models.backbone import build_encoder
from models.RPN import CONV
from anchors import generate_anchors
#from models.head import loss_head
from models.eval_graph import eval_graph
from models.predict_func import parse_outputs
from utilis.data_format import data_crop
import tensorflow as tf

#import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


stray_weights = {'cls1_kernel':[np.load('constant/conv_cls1_kernel_0.npy'), np.load('constant/conv_cls1_bias_0.npy')],
               'r1_kernel':[np.load('constant/conv_r1_kernel_0.npy'), np.load('constant/conv_r1_bias_0.npy')],
               'cls2_kernel':[np.load('constant/conv_cls2_kernel_0.npy'), np.load('constant/conv_cls2_bias_0.npy')],
               'r2_kernel':[np.load('constant/conv_r2_kernel_0.npy'), np.load('constant/conv_r2_bias_0.npy')],
               'regress_adjust': [np.load('constant/regress_adjust_kernel_0.npy'), np.load('constant/regress_adjust_bias_0.npy')],
               'anchors': np.load('constant/anchors.npy')}
  




class Siamese_RPN():
    def __init__(self, mode, config, model_dir='./log'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference','inference_init']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        if model_dir != None:
            self.set_log_dir()
        self.keras_model = self.build()
    def build(self):
        ##############
        # Set Inputs
        ##############
        
        
        if self.mode == 'inference_init':
            # Input template's batch size is nailed to 1. 
            inp_template = Input(batch_shape = (1,)+self.config.template_size, name='inp_template')
        
        
        
        elif self.mode == 'inference':
            # When evaluating batch size must be 1.!!!!!!
            assert self.config.batch_size == 1
            
            inp_img = Input(shape=self.config.instance_size,name='inp_img')
            # Generate anchors for every batch,
            anchors = generate_anchors(self.config.total_stride,
                                       self.config.scales,self.config.ratios,self.config.score_size)
            anchors = np.broadcast_to(anchors, (self.config.batch_size,)+anchors.shape) #shape=(1, 19, 19, 5, 4)
      
    
           
            
            
            
            
            
        ###########################
        # Set Backbone
        ###########################
        
        self.encoder = build_encoder()
        
        
        if self.mode == 'inference':
            encoded_img= self.encoder(inp_img)  
            model = Model([inp_img], outputs=encoded_img, name='bb_alex_large')
            return model
        
        
               
        elif self.mode == 'inference_init':
     
            cls_filters = 2*self.config.num_anchors*self.config.encoder_out_filter #5120
            bbox_filters = 4*self.config.num_anchors*self.config.encoder_out_filter #10240            
            encoded_template = self.encoder(inp_template)           
            
            model = Model([inp_template], encoded_template, name = 'bb_alex_small')        
                           
          
            return model
            
  
    @tf.function
    def compute_cnn_head(self, bb_outputs, cls_side_params, bbox_side_params):
        '''compute CNN head of template model'''

        cls_head_out = tf.nn.conv2d(bb_outputs, cls_side_params[0], strides=(1,1,1,1), padding='VALID') + cls_side_params[1]
        bbox_head_out = tf.nn.conv2d(bb_outputs, bbox_side_params[0], strides=(1,1,1,1), padding='VALID') + bbox_side_params[1]
        
        return cls_head_out, bbox_head_out
    
    

    @tf.function
    def compute_RPN_head(self, cnn_head_outs):         
    

        cls_out = tf.nn.conv2d(cnn_head_outs[0], self.config.cls_template, strides=(1,1,1,1), padding='VALID')          
        bbox_out = tf.nn.conv2d(cnn_head_outs[1], self.config.bbox_template, strides=(1,1,1,1), padding='VALID')
        
        bbox_out = tf.nn.conv2d(bbox_out, stray_weights['regress_adjust'][0], strides=(1,1,1,1), padding='VALID') + stray_weights['regress_adjust'][1]
        
        boxes, scores = eval_graph(bbox_out, cls_out, stray_weights['anchors'])
        
        return boxes, scores
    
    
    
    
    def reshape_template(self, template):
        '''reshape template to fit with the inference part [4 4 512 10||20]'''        
        template = tf.squeeze(template, axis=0)
        template = tf.reshape(template, (template.shape[0], template.shape[1], -1, self.config.encoder_out_filter))
        template = tf.transpose(template, (0,1,3,2))
        return template
        
        
    
    
    # ─────────────────────────────────────────────────────────────────


    def inference_init(self,img):
        input_template = data_crop(img, self.config, mode = 'template')
        input_template = np.expand_dims(input_template, axis = 0)
        bb_outputs = self.keras_model.predict(input_template)        
        
            
        
        cls_template, bbox_template = self.compute_cnn_head(bb_outputs, stray_weights['cls1_kernel'], stray_weights['r1_kernel'])        
        
        
        self.config.cls_template = self.reshape_template(cls_template)
        self.config.bbox_template = self.reshape_template(bbox_template)
        print('[INFO] Store template feature_map')
       
      
        
        
        
    def predict(self, img):
        input_img, scale_size = data_crop(img, self.config, mode = 'instance')
        input_img = np.expand_dims(input_img, axis = 0)
        
        
        bb_outputs = self.keras_model.predict(input_img)        
        cnn_head_outs = self.compute_cnn_head(bb_outputs, stray_weights['cls2_kernel'], stray_weights['r2_kernel'])               
        boxes, scores = self.compute_RPN_head(cnn_head_outs)               
        boxes = np.squeeze(boxes, axis = 0)
        scores = np.squeeze(scores, axis = 0)
        

        xy,wh,score = parse_outputs(boxes, scores, scale_size, self.config)
        # Update outputs
        self.config.outputs['target_xy'] = xy
        self.config.outputs['target_wh'] = wh
        return xy,wh,score


