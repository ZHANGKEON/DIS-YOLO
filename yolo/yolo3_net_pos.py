# -*- coding: utf-8 -*-
"""
Created on Fri May 24 9:25:12 2019

@author: Chaobo
"""

import tensorflow as tf
import yolo.config as cfg
slim = tf.contrib.slim

class YOLONet(object):
    def __init__(self, training=False):
        
        # 1. Initialize parameters
        # yolo parameters
        self.batchsize    = cfg.BATCH_SIZE 
        self.classes      = cfg.CLASSES       
        self.num_class    = len(self.classes)
        self.anchors      = cfg.ANCHORS
        self.num_anchor   = 3
        self.output_depth = (self.num_class + 5) * self.num_anchor
        max_grid_h, max_grid_w = [int(cfg.BASE_GRID * 4), int(cfg.BASE_GRID * 4)]
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.offset = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, self.num_anchor, 1])
        
        # mask subnet parameters
        self.k        = cfg.K_MAP
        self.k_mapout = self.k * self.k
        
        # training loss parameter 
        self.object_scale   = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale    = cfg.CLASS_SCALE
        self.coord_scale    = cfg.COORD_SCALE
        self.mask_scale     = cfg.MASK_SCALE
        self.regularizer    = tf.contrib.layers.l2_regularizer(0.0001)
        
        # input training parameters
        self.is_training = tf.placeholder(tf.bool, name='training')
        self.det_thresh  = tf.placeholder(tf.float32, [1], name='object_threshold')
        self.clip_window = tf.placeholder(tf.float32, [None, 4], name='clip_window')
        self.images      = tf.placeholder(tf.float32, [None, None, None, 3], name='images')

        # 2. Network output = [predicted logits, box detection output, position-sensitive score maps]
        self.logits = self.build_network(images=self.images, depth_outputs=self.output_depth, alpha=cfg.ALPHA, 
                                         is_training=self.is_training)

        # 3. Training loss
        if training:
            self.yolo1 = tf.placeholder(tf.float32, [None, None, None, self.num_anchor, 5 + self.num_class])
            self.yolo2 = tf.placeholder(tf.float32, [None, None, None, self.num_anchor, 5 + self.num_class])
            self.yolo3 = tf.placeholder(tf.float32, [None, None, None, self.num_anchor, 5 + self.num_class])
            self.labels_value = [self.yolo3, self.yolo2, self.yolo1]
            self.true_boxes   = tf.placeholder(tf.float32, [None, 1, 1, 1, cfg.MAX_BOX_PER_IMAGE, 5])
            self.true_masks   = tf.placeholder(tf.bool, [None, cfg.MAX_BOX_PER_IMAGE, None, None])
            
            self.loss_yolo(self.logits[0], self.true_boxes,  self.labels_value)
            self.loss_mask(self.logits[1], self.logits[2], self.true_boxes, self.true_masks, iou_threshold=0.5)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
        
        # 4. used in evaluation to output final predictions for validation and testing stage
        self.evaluation = self.val_test(self.logits[1], self.logits[2])
        

    def leaky_relu(self, inputs, alpha):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu') 
    
    def batch_norm(self, inputs, lock, istraining):
        with tf.variable_scope('BatchNorm'):
            is_conv_out = True
            decay = 0.997
            epsilon = 1e-5
            if lock:
                gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='gamma', trainable=False)
                beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='beta', trainable=False)
                moving_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='moving_mean', trainable=False)
                moving_variance = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='moving_variance', trainable=False)
                out = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
            else:
                gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='gamma')     # the size of depth
                beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='beta')
                moving_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='moving_mean', trainable=False)
                moving_variance = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='moving_variance', trainable=False)
    
                def f1():
                    if is_conv_out:
                        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
                    else:
                        batch_mean, batch_var = tf.nn.moments(inputs,[0])  
                    train_mean = tf.assign(moving_mean,
                                       moving_mean * decay + batch_mean * (1 - decay))
                    train_var = tf.assign(moving_variance,
                                      moving_variance * decay + batch_var * (1 - decay))
                    with tf.control_dependencies([train_mean, train_var]):
                        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
                
                def f2():
                    return tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
                
                out = tf.cond(istraining, f1, f2)
                
                # used in testing stage for calculating testing time
#                out = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
        return out

    def conv(self, intensor, in_channels, out_channels, filter_size, step, is_bias, is_act, lock, alpha, scope):
        with tf.variable_scope(scope):
            if lock:
                ini_filters= tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
                filters = tf.Variable(ini_filters, name='weights', trainable=False)
                if is_bias:
                    ini_bias = tf.constant(0.0, shape=[out_channels])
                    bias = tf.Variable(ini_bias, name = 'biases', trainable=False)
            else:    
                filters = tf.get_variable(shape=[filter_size, filter_size, in_channels, out_channels],
                                          initializer = tf.contrib.layers.xavier_initializer(), 
                                          regularizer=self.regularizer, name='weights')
                if is_bias:
                    ini_bias = tf.constant(0.0, shape=[out_channels])
                    bias = tf.get_variable(initializer = ini_bias, regularizer=self.regularizer, name = 'biases')
                          
            net =  tf.nn.conv2d(intensor, filters, strides=[1, step, step, 1], padding='SAME')
            if is_bias:
                net = tf.nn.bias_add(net, bias)
            if is_act:
                net = self.leaky_relu(net, alpha)        
        return net

    def conv_bn(self, intensor, in_channels, out_channels, filter_size, step, is_act, alpha, lock, istraining, scope):
        with tf.variable_scope(scope):
            if lock:
                ini_filters= tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
                filters = tf.Variable(ini_filters, name='weights', trainable=False)
            else: 
                filters = tf.get_variable(shape=[filter_size, filter_size, in_channels, out_channels],
                                          initializer = tf.contrib.layers.xavier_initializer(), 
                                          regularizer=self.regularizer, name='weights')
                
            net =  tf.nn.conv2d(intensor, filters, strides=[1, step, step, 1], padding='SAME')            
            net = self.batch_norm(net, lock, istraining)
            if is_act:
                net = self.leaky_relu(net, alpha)        
        return net

    def res_conv_bn(self, intensor, shortcut, in_channels, out_channels, filter_size, step, is_act, alpha, lock, istraining, scope):
        net_conv = self.conv_bn(intensor,in_channels, out_channels, filter_size, step, is_act, alpha, lock, istraining,  scope)
        resnet = net_conv + shortcut
        return resnet    

    def build_network(self, images, depth_outputs, alpha, is_training, scope='yolo'):
        with tf.variable_scope(scope):
            # Training Stage 1: heads only: "lock=True" for layers 1-52, "lock=False" for other layers
            # Training Stage 2: all layers: "lock=False" for all layers
            
            # backbone
            net = self.conv_bn(intensor = images, in_channels = 3, out_channels = 32, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional1')
            
            skip1 = net
            
            net = self.conv_bn(intensor = net, in_channels = 32, out_channels = 64, filter_size = 3, step = 2, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional2')
            
            shortcut = net
            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 32, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional3')
            
            net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 32, out_channels = 64, 
                                   filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True,
                                   istraining= is_training, scope = 'convolutional4')

            skip2 = net
            
            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 128, filter_size = 3, step = 2, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional5')
            
            shortcut = net
            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 64, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional6')
            
            net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 64, out_channels = 128, 
                                   filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True,
                                   istraining= is_training, scope = 'convolutional7')
            
            shortcut = net
            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 64, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional8')
            
            net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 64, out_channels = 128, 
                                   filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True,
                                   istraining= is_training, scope = 'convolutional9')

            skip3 = net

            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 256, filter_size = 3, step = 2, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional10')
            
            for i in range(8):
                shortcut = net
                str1 = 'convolutional' + str(2*i + 11)
                net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 128, filter_size = 1, step = 1, 
                                   is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                                   scope = str1)
                
                str2 = 'convolutional' + str(2*i + 12)
                net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 128, out_channels = 256, 
                                       filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True, 
                                       istraining= is_training, scope = str2)
            
            skip4 = net
            
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 512, filter_size = 3, step = 2, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional27')
            
            for ii in range(8):
                shortcut = net
                str1 = 'convolutional' + str(2*ii + 28)
                net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 256, filter_size = 1, step = 1, 
                                   is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                                   scope = str1)
                
                str2 = 'convolutional' + str(2*ii + 29)
                net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 256, out_channels = 512, 
                                       filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True, 
                                       istraining= is_training, scope = str2)
            
            skip5 = net
            
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 1024, filter_size = 3, step = 2, 
                               is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                               scope = 'convolutional44')
            
            for iii in range(4):
                shortcut = net
                str1 = 'convolutional' + str(2*iii + 45)
                net = self.conv_bn(intensor = net, in_channels = 1024, out_channels = 512, filter_size = 1, step = 1, 
                                   is_act = True, alpha = alpha, lock = True, istraining= is_training, 
                                   scope = str1)
                
                str2 = 'convolutional' + str(2*iii + 46)
                net = self.res_conv_bn(intensor = net, shortcut = shortcut, in_channels = 512, out_channels = 1024, 
                                       filter_size = 3, step = 1, is_act = True, alpha = alpha, lock = True, 
                                       istraining= is_training, scope = str2)
        
            
            # yolov3_1: first scale for large object box detection
            net = self.conv_bn(intensor = net, in_channels = 1024, out_channels = 512, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional53')
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 1024, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional54')             
            net = self.conv_bn(intensor = net, in_channels = 1024, out_channels = 512, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional55')
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 1024, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional56')         
            net = self.conv_bn(intensor = net, in_channels = 1024, out_channels = 512, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional57')
            
            yolov3_1 = self.conv_bn(intensor = net, in_channels = 512, out_channels = 1024, filter_size = 3, step = 1, 
                                    is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                                    scope = 'convolutional58')
            yolov3_1 = self.conv(intensor =  yolov3_1, in_channels = 1024, out_channels = depth_outputs, filter_size =1, 
                                 step =1, is_bias = True, is_act = False, lock = False, alpha = alpha, 
                                 scope = 'convolutional59')
            shape1 = tf.shape(yolov3_1)
            yolov3_1 = tf.reshape(yolov3_1, (shape1[0], shape1[1], shape1[2], self.num_anchor, 5 + self.num_class))


            # yolov3-2: second scale for medium object box detection
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 256, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional60')
            
            size = tf.shape(net)[1]
            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_1')
            net = tf.concat([skip5, net], axis=-1, name='concat_1')
            
            net = self.conv_bn(intensor = net, in_channels = 768, out_channels = 256, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional61')
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 512, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional62')
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 256, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional63')
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 512, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional64')
            net = self.conv_bn(intensor = net, in_channels = 512, out_channels = 256, filter_size = 1, step = 1, 
                                is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                                scope = 'convolutional65')
            
            yolov3_2 = self.conv_bn(intensor = net, in_channels = 256, out_channels = 512, filter_size = 3, step = 1, 
                                    is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                                    scope = 'convolutional66')
            yolov3_2 = self.conv(intensor =  yolov3_2, in_channels = 512, out_channels = depth_outputs, filter_size =1, 
                                 step =1, is_bias = True, is_act = False, lock = False, alpha = alpha, 
                                 scope = 'convolutional67')
            shape2 = tf.shape(yolov3_2)
            yolov3_2 = tf.reshape(yolov3_2, (shape2[0], shape2[1], shape2[2], self.num_anchor, 5 + self.num_class))


            # yolov3-3 third scale for small object box detection
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 128, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional68')
            
            size = tf.shape(net)[1]   
            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_2')
            net = tf.concat([skip4, net], axis=-1, name='concat_2')

            net = self.conv_bn(intensor = net, in_channels = 384, out_channels = 128, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional69')
            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 256, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional70')
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 128, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional71')
            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 256, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional72')
            net = self.conv_bn(intensor = net, in_channels = 256, out_channels = 128, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional73')
            
            yolov3_3 = self.conv_bn(intensor = net, in_channels = 128, out_channels = 256, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional74') 
            yolov3_3 = self.conv(intensor = yolov3_3, in_channels = 256, out_channels = depth_outputs, filter_size =1, 
                            step =1, is_bias = True, is_act = False, lock = False, alpha = alpha, 
                            scope = 'convolutional75')
            shape3 = tf.shape(yolov3_3)
            yolov3_3 = tf.reshape(yolov3_3, (shape3[0], shape3[1], shape3[2], self.num_anchor, 5 + self.num_class))
            
            yolos = [yolov3_3, yolov3_2, yolov3_1]
            # predictions provide paramters for computing detections and the yolov3 loss
            # detections output boxes and provide proposals for mask subnet             
            predictions = self.interpret_output(yolos)
            detections  = self.filter_detections(predictions[2], predictions[3], predictions[5], 
                                                 batch_window=self.clip_window, obj_thresh=self.det_thresh, nms=True, 
                                                 nms_thresh=cfg.IOU_THRESHOLD, max_detection=cfg.MAX_DETECTION)
            
            '''mask subnet with stride=4 (m=1/4)'''
#            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 64, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional76')
#            
#            size = tf.shape(net)[1]
#            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_3')
#            net = tf.concat([skip3, net], axis=-1, name='concat_3')
#            
#            net = self.conv_bn(intensor = net, in_channels = 192, out_channels = 64, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional77')           
#            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 128, filter_size = 3, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional78')
#            mask_pos = self.conv(intensor = net, in_channels = 128, out_channels = self.k_mapout, filter_size = 1, step = 1, 
#                               is_bias = True, is_act = False, lock = False, alpha = alpha,
#                               scope = 'convolutional79')        

            '''mask subnet with stride=2 (m=1/2)'''
            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 64, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional76')
            
            size = tf.shape(net)[1]     
            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_3')
            net = tf.concat([skip3, net], axis=-1, name='concat_3')

            net = self.conv_bn(intensor = net, in_channels = 192, out_channels = 64, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional77')           
            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 128, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional78')

            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 32, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional79') 
            
            size = tf.shape(net)[1] 
            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_4')
            net = tf.concat([skip2, net], axis=-1, name='concat_4')   
            
            net = self.conv_bn(intensor = net, in_channels = 96, out_channels = 32, filter_size = 1, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional80')   
            net = self.conv_bn(intensor = net, in_channels = 32, out_channels = 64, filter_size = 3, step = 1, 
                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
                               scope = 'convolutional81')
            mask_pos = self.conv(intensor = net, in_channels = 64, out_channels = self.k_mapout, filter_size = 1, step = 1, 
                               is_bias = True, is_act = False, lock = False, alpha = alpha,
                               scope = 'convolutional82')
            
            '''mask subnet with stride=1 (m=1)'''
#            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 64, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional76')
#            
#            size = tf.shape(net)[1]     
#            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_3')
#            net = tf.concat([skip3, net], axis=-1, name='concat_3')
#
#            net = self.conv_bn(intensor = net, in_channels = 192, out_channels = 64, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional77')           
#            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 128, filter_size = 3, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional78')
#            
#            net = self.conv_bn(intensor = net, in_channels = 128, out_channels = 32, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional79') 
#            
#            size = tf.shape(net)[1] 
#            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_4')
#            net = tf.concat([skip2, net], axis=-1, name='concat_4')   
#            
#            net = self.conv_bn(intensor = net, in_channels = 96, out_channels = 32, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional80')   
#            net = self.conv_bn(intensor = net, in_channels = 32, out_channels = 64, filter_size = 3, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional81')           
#
#            net = self.conv_bn(intensor = net, in_channels = 64, out_channels = 16, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional82') 
#            
#            size = tf.shape(net)[1] 
#            net = tf.image.resize_nearest_neighbor(net, [2*size, 2*size], name='upsample_5')
#            net = tf.concat([skip1, net], axis=-1, name='concat_5')   
#
#            net = self.conv_bn(intensor = net, in_channels = 48, out_channels = 16, filter_size = 1, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional83')   
#            net = self.conv_bn(intensor = net, in_channels = 16, out_channels = 32, filter_size = 3, step = 1, 
#                               is_act = True, alpha = alpha, lock = False, istraining= is_training, 
#                               scope = 'convolutional84')           
#            mask_pos = self.conv(intensor = net, in_channels = 32, out_channels = self.k_mapout, filter_size = 1, step = 1, 
#                               is_bias = True, is_act = False, lock = False, alpha = alpha,
#                               scope = 'convolutional85')
            
            return [predictions, detections, mask_pos]

    def interpret_output(self, predicts):

        anchors_pwhs = []
        conf_logits = []
        class_logits = []
        pred_coords = []
        pred_norm_coords = []
         
        batch_size = tf.shape(predicts[2])[0]
        net_h = tf.shape(predicts[2])[1] * 32
        net_w = tf.shape(predicts[2])[2] * 32  
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), (1,1,1,1,2))
        
        for i in [0 ,1, 2]:
            preds = predicts[i]
            grid_h      = tf.shape(preds)[1]
            grid_w      = tf.shape(preds)[2]
            grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), (1,1,1,1,2))
                    
            # format the logits
            pred_conf  = tf.expand_dims(preds[..., 4], 4)
            pred_class = preds[..., 5:]
            pred_cxy   = tf.sigmoid(preds[..., :2])
            pred_twh   = preds[..., 2:4]
            pred_coord = tf.concat([pred_cxy, pred_twh], -1)
            
            # compute normalized coordinates
            cell_grid  = tf.tile(self.offset, [batch_size, 1, 1, 1, 1])
            box_xy     = cell_grid[:,:grid_h,:grid_w,:,:] + pred_cxy
            
            anchors_pw  = tf.constant(self.anchors, tf.float32)[3*i : 3*i+3, 0]
            anchors_ph  = tf.constant(self.anchors, tf.float32)[3*i : 3*i+3, 1]
            anchors_pw  = tf.tile(anchors_pw, [batch_size * grid_h * grid_w])
            anchors_ph  = tf.tile(anchors_ph, [batch_size * grid_h * grid_w])
            anchors_pw  = tf.reshape(anchors_pw, (batch_size, grid_h, grid_w, self.num_anchor, 1))
            anchors_ph  = tf.reshape(anchors_ph, (batch_size, grid_h, grid_w, self.num_anchor, 1))
            anchors_pwh = tf.concat([anchors_pw, anchors_ph], -1) # [batch, grid_h, grid_w, 3, 2]
            box_wh      = tf.exp(preds[..., 2:4]) * anchors_pwh
            
            box_xy = box_xy / grid_factor
            box_wh = box_wh / net_factor
            pred_norm_coord = tf.concat([box_xy, box_wh], -1)

            anchors_pwhs.append(anchors_pwh)
            conf_logits.append(pred_conf)
            class_logits.append(pred_class)
            pred_coords.append(pred_coord)
            pred_norm_coords.append(pred_norm_coord)
            
        return [[net_h, net_w], anchors_pwhs, conf_logits, class_logits, pred_coords, pred_norm_coords]                


    def filter_detections(self, conf_logit, class_logit, pred_norm_coord, batch_window,
                          obj_thresh=0.25, nms=True, nms_thresh=0.3, max_detection=20):
        # return det_out = array[batch, max_detection, 6(y1, x1, y2, x2, classid, class-conf)]

        detections = []
        for i in range(self.batchsize):            
            # merge predictions and put them in the form of array[num_anchor,...]
            pred_confs = []
            pred_classes = []
            pred_norm_boxes = []
            for j in [0 ,1, 2]:
                pred_conf = tf.squeeze(tf.sigmoid(conf_logit[j][i,...]))
                pred_conf = tf.reshape(pred_conf, [-1])
                pred_confs.append(pred_conf)
                
                pred_class = tf.nn.softmax(class_logit[j][i,...])
                pred_class = tf.reshape(pred_class, [-1, self.num_class])
                pred_classes.append(pred_class)
                
                pred_norm_box = pred_norm_coord[j][i,...]
                pred_norm_box = tf.reshape(pred_norm_box, [-1, 4])
                pred_norm_boxes.append(pred_norm_box)
            
            pred_conf = tf.concat(pred_confs, 0)
            pred_class = tf.concat(pred_classes, 0)
            pred_norm_box = tf.concat(pred_norm_boxes, 0)
            
            # compute class-specific confidence
            pred_classid    = tf.cast(tf.argmax(pred_class, -1), tf.int32)
            indices         = tf.stack([tf.range(tf.shape(pred_class)[0]), pred_classid], axis=1)
            pred_classmax   = tf.gather_nd(pred_class, indices)
            pred_conf_class = pred_conf * pred_classmax
            
            # In traing, clip pred_norm_box to image boundaries, i.e. 0-1 range.
            # In testing, clip boxes to clipwindow that excludes the padding.
            xc, yc, w, h  = tf.split(pred_norm_box, 4, axis=1) 
            pred_norm_box = tf.concat([yc - h/2.0, xc - w/2.0, yc + h/2.0, xc + w/2.0], -1)
            window        = batch_window[i,...]
            pred_norm_box = self.clip_boxes_graph(pred_norm_box, window)
            
            # filter detections by class-specific confidence
            keep = tf.where(pred_conf_class > obj_thresh)[:, 0]
            pre_nms_classid = tf.gather(pred_classid, keep)
            pre_nms_score   = tf.gather(pred_conf_class, keep)
            pre_nms_box     = tf.gather(pred_norm_box, keep)

            # we provide two NMS methods, you can try different forms for better results
            # Method 1 : Apply per-class NMS
            unique_pre_nms_classid = tf.unique(pre_nms_classid)[0]
            def nms_keep_map(class_id):
                ixs = tf.where(tf.equal(pre_nms_classid, class_id))[:, 0]
                class_keep = tf.image.non_max_suppression(
                        tf.gather(pre_nms_box, ixs),
                        tf.gather(pre_nms_score, ixs),
                        max_output_size=max_detection,
                        iou_threshold=nms_thresh)
                class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
                
                # Pad with -1 so returned tensors have the same shape
                gap = max_detection - tf.shape(class_keep)[0]
#                class_keep = tf.pad(class_keep, [(0, gap)],
#                                    mode='CONSTANT', constant_values=-1)
                padarray = tf.cast(-1 * tf.ones([gap]), tf.int64)
                class_keep = tf.concat([class_keep, padarray], 0)                

                class_keep.set_shape([max_detection])
                return class_keep
            
            if nms:
                # Map over class IDs
                nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_classid, dtype=tf.int64)
                nms_keep = tf.reshape(nms_keep, [-1])
                nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
                keep     = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                                    tf.expand_dims(nms_keep, 0))
                keep     = tf.sparse_tensor_to_dense(keep)[0]

#            # Method 2 : Apply no-class-specific NMS
#            if nms:
#                ixs = tf.where(pre_nms_classid > -1)[:, 0]
#                nms_keep = tf.image.non_max_suppression(
#                                pre_nms_box,
#                                pre_nms_score,
#                                max_output_size=max_detection,
#                                iou_threshold=nms_thresh)
#                nms_keep = tf.gather(keep, tf.gather(ixs, nms_keep))
#                keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
#                                                tf.expand_dims(nms_keep, 0))
#                keep = tf.sparse_tensor_to_dense(keep)[0]
    
            # Keep top detections
            roi_count  = max_detection
            score_keep = tf.gather(pred_conf_class, keep)
            num_keep   = tf.minimum(tf.shape(score_keep)[0], roi_count)
            top_ids    = tf.nn.top_k(score_keep, k=num_keep, sorted=True)[1]
            keep       = tf.gather(keep, top_ids)
        
            # format output as array[N, (y1, x1, y2, x2, classid, class-conf)]
            detection = tf.concat([
                    tf.gather(pred_norm_box, keep),
                    tf.to_float(tf.gather(pred_classid, keep))[..., tf.newaxis],
                    tf.gather(pred_conf_class, keep)[..., tf.newaxis]
                    ], axis=1)
        
            # Pad with zeros if detections < max_detection
            gap = max_detection - tf.shape(detection)[0]
            detection = tf.pad(tensor = detection, paddings = [(0, gap), (0, 0)], mode = "CONSTANT")
            
            detections.append(detection)
        
        det_out = tf.concat([tf.expand_dims(det, 0) for det in detections], axis = 0)
        return det_out


    def loss_yolo(self, predicts, true_boxes, labels_value, scope='loss_yolo'):
        # compute the box detection loss for training yolov3

        with tf.variable_scope(scope):
            # for loss monitoring
            objloss   = 0.
            noobjloss = 0.
            xyloss    = 0.
            whloss    = 0.
            # three losses for training
            confloss  = 0.
            classloss = 0.
            coordloss = 0.
            
            batch_size       = self.batchsize
            net_h, net_w     = predicts[0]
            anchors_pwhs     = predicts[1]
            conf_logits      = predicts[2]
            class_logits     = predicts[3]
            pred_coords      = predicts[4]
            pred_norm_coords = predicts[5]
            
            for i, loss_i in enumerate(['Loss_yolo3', 'Loss_yolo2', 'Loss_yolo1']):
                with tf.variable_scope(loss_i):
                    
                    # ignor confidence loss from the boxes that have iou with ground-true box > 0.5
                    preds_box_coord  = pred_norm_coords[i]
                    pred_xy          = tf.expand_dims(preds_box_coord[..., :2], 4)
                    pred_wh          = tf.expand_dims(preds_box_coord[..., 2:4], 4)
                    pred_wh_half     = pred_wh / 2.
                    pred_mins        = pred_xy - pred_wh_half
                    pred_maxes       = pred_xy + pred_wh_half
                    # true boxes
                    true_xy          = true_boxes[..., 0:2]
                    true_wh          = true_boxes[..., 2:4]
                    true_wh_half     = true_wh / 2.
                    true_mins        = true_xy - true_wh_half
                    true_maxes       = true_xy + true_wh_half                    
                    # compute iou between predicted and true boxes
                    intersect_mins   = tf.maximum(pred_mins,  true_mins)
                    intersect_maxes  = tf.minimum(pred_maxes, true_maxes)
                    intersect_wh     = tf.maximum(intersect_maxes - intersect_mins, 0.)
                    intersect_areas  = intersect_wh[..., 0] * intersect_wh[..., 1]
                    true_areas       = true_wh[..., 0] * true_wh[..., 1]
                    pred_areas       = pred_wh[..., 0] * pred_wh[..., 1]
                    union_areas      = tf.maximum( pred_areas + true_areas - intersect_areas, 1e-10)
                    iou_scores       = tf.clip_by_value(intersect_areas / union_areas, 0.0, 1.0)
                    # generate mask for confidence loss
                    best_ious        = tf.reduce_max(iou_scores, axis=4)
                    conf_ignore_mask = tf.expand_dims(tf.to_float(best_ious < cfg.IGNORE_THRESH), 4)
                    
                    # total loss = confidence loss + classification loss + coordinate loss
                    object_value = labels_value[i]
                    
                    # 1. confidence loss
                    pred_conf      = conf_logits[i]
                    object_mask    = tf.expand_dims(object_value[:, :, :, :, 4], 4)
                    noobject_mask  = tf.ones_like(object_mask, dtype=tf.float32) - object_mask                    
                    noobj_ignore   = conf_ignore_mask * noobject_mask
                    # sigmoid_cross_entropy loss
                    object_delta   = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf) * self.object_scale
                    object_loss    = tf.reduce_mean(tf.reduce_sum(object_delta, axis=[1, 2, 3, 4]), name='object_loss') 
                    noobject_delta = noobj_ignore * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf) * self.noobject_scale
                    noobject_loss  = tf.reduce_mean(tf.reduce_sum(noobject_delta, axis=[1, 2, 3, 4]), name='noobject_loss')                   
                    conf_loss      = object_loss + noobject_loss
                    
                    # 2. classification loss
                    pred_classes   = class_logits[i] 
                    true_classes   = tf.argmax(object_value[:, :, :, :, 5:], -1)
                    # softmax_cross_entropy loss
                    class_delta    = object_mask *tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=true_classes, logits=pred_classes), 4) * self.class_scale
                    class_loss     = tf.reduce_mean(tf.reduce_sum(class_delta, axis=[1, 2, 3, 4]), name='class_loss')

                    # 3. coordinate loss
                    pred_cxy       = pred_coords[i][..., :2]
                    pred_twh       = pred_coords[i][..., 2:4]
                    
                    grid_h         = tf.shape(pred_conf)[1]
                    grid_w         = tf.shape(pred_conf)[2]
                    grid_factor    = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), (1,1,1,1,2)) 
                    cell_grid      = tf.tile(self.offset, [batch_size, 1, 1, 1, 1])
                    true_box_coord = object_value[:, :, :, :, 0:4]
                    true_cxy       = true_box_coord[:, :, :, :, 0:2] * grid_factor - cell_grid[:,:grid_h,:grid_w,:,:]
                    anchors_pwh    = anchors_pwhs[i]
                    net_factor     = tf.reshape(tf.cast([net_w, net_h], tf.float32), (1,1,1,1,2))
                    true_twh       = true_box_coord[:, :, :, :, 2:4] * net_factor
                    true_twh       = tf.clip_by_value(tf.log(true_twh / anchors_pwh), -1e2, 1e2) # avoid log(0)=-inf 
                    
                    # dynamic loss scale: smaller the box, bigger the scale
                    wh_scale       = true_box_coord[:, :, :, :, 2:4]
                    wh_scale       = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
                    cxy_delta      = object_mask * (pred_cxy - true_cxy)
                    twh_delta      = object_mask * (pred_twh - true_twh)
                    xy_loss    = tf.reduce_mean(tf.reduce_sum(tf.square(cxy_delta)* tf.square(wh_scale) * self.coord_scale, axis=[1, 2, 3, 4]), name='xy_loss')
                    wh_loss    = tf.reduce_mean(tf.reduce_sum(tf.square(twh_delta)* tf.square(wh_scale) * self.coord_scale, axis=[1, 2, 3, 4]), name='wh_loss')
                    coord_loss = xy_loss + wh_loss
                    
                    # sum the loss from different yolov3 box scales
                    objloss   += object_loss
                    noobjloss += noobject_loss
                    xyloss    += xy_loss
                    whloss    += wh_loss                    
                    
                    confloss  += conf_loss
                    classloss += class_loss
                    coordloss += coord_loss

            tf.losses.add_loss(confloss)
            tf.losses.add_loss(classloss)
            tf.losses.add_loss(coordloss)
            
            tf.summary.scalar('object_loss', objloss)
            tf.summary.scalar('noobject_loss', noobjloss)
            tf.summary.scalar('class_loss', classloss)
            tf.summary.scalar('xy_loss', xyloss)
            tf.summary.scalar('wh_loss', whloss)
   

    def loss_mask(self, box_out, mask_out, true_boxes, true_masks, iou_threshold=0.5, scope="loss_mask"):
        # compute the mask loss for training mask subnet
        
        with tf.variable_scope(scope): 
            mask_loss = 0.
            for i in range(self.batchsize):
                # 1. format detections
                proposals     = box_out[i, :, :4]
                pred_masks    = mask_out[i, ...]
                prop_nonzeros = tf.cast(tf.reduce_sum(tf.abs(proposals), axis=1), tf.bool)
                proposals     = tf.boolean_mask(proposals, prop_nonzeros, name="trim_proposals")
                
                # 2. format ground truth annotations
                gt_boxes     = true_boxes[i, 0, 0, 0, :, :4] 
                gt_class_ids = tf.cast(true_boxes[i, 0, 0, 0, :, 4], tf.int32)
                gt_masks     = true_masks[i,...]
                gt_nonzeros  = tf.cast(tf.reduce_sum(tf.abs(gt_boxes), axis=1), tf.bool)
                gt_boxes     = tf.boolean_mask(gt_boxes, gt_nonzeros, name="trim_gt_boxes")
                gt_class_ids = tf.boolean_mask(gt_class_ids, gt_nonzeros, name="trim_gt_class_ids")
                gt_masks     = tf.boolean_mask(gt_masks, gt_nonzeros, name="trim_gt_masks")
                # resize gt_masks to the size of pred_mask
                gt_masks     = tf.expand_dims(gt_masks, -1) 
                size         = tf.shape(pred_masks)[1]
                gt_masks     = tf.image.resize_images(tf.cast(gt_masks, tf.float32), [size, size])
                gt_masks     = tf.squeeze(gt_masks, axis=3)
                gt_masks     = tf.round(gt_masks)
                
                # 3. Compute overlaps between [proposals, gt_boxes]
                xc, yc, w, h  = tf.split(gt_boxes, 4, axis=1)
                gt_boxes      = tf.concat([yc - h/2.0, xc - w/2.0, yc + h/2.0, xc + w/2.0], -1)
                # add gt_box for training mask at eariler epochs, adjust number if memory is not enough
                add_gt_boxes  = tf.random_shuffle(gt_boxes)
                add_proposals = tf.random_shuffle(proposals)
                proposals     = tf.concat([add_proposals[:7,:], add_gt_boxes[:3,:]], 0)
                overlaps      = self.overlaps_graph(proposals, gt_boxes)
        
                # 4. determine positive proposals and assign ground-truth masks
                roi_iou_max       = tf.reduce_max(overlaps, axis=1)
                positive_roi_bool = (roi_iou_max >= iou_threshold)
                positive_indices  = tf.where(positive_roi_bool)[:, 0]
                positive_rois     = tf.gather(proposals, positive_indices)
                positive_overlaps = tf.gather(overlaps, positive_indices)
                roi_gt_box_assignment = tf.cond(
                        tf.greater(tf.shape(positive_overlaps)[1], 0),
                        lambda: tf.argmax(positive_overlaps, axis=1),
                        lambda: tf.cast(tf.constant([]),tf.int64))
                roi_masks = tf.gather(gt_masks, roi_gt_box_assignment)
                
                # assemble masks from position-sensitive score maps
                def assemble_kmask_from_box(box):
                    y1    = box[0]
                    x1    = box[1]
                    y2    = box[2]
                    x2    = box[3]
                    w     = x2 - x1
                    h     = y2 - y1
                    sub_w = w / self.k
                    sub_h = h / self.k

                    # manually switch following code based on the value of k=(3, 5 ,7)
                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+sub_w), tf.int32), 
                              tf.cast(tf.round(x1+ 2*sub_w), tf.int32), tf.cast(x2, tf.int32)]
                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+sub_h), tf.int32), 
                              tf.cast(tf.round(y1+ 2*sub_h), tf.int32), tf.cast(y2, tf.int32)]
#                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+ sub_w), tf.int32), tf.cast(tf.round(x1+ 2*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 3*sub_w), tf.int32), tf.cast(tf.round(x1+ 4*sub_w), tf.int32), tf.cast(x2, tf.int32)]
#                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+ sub_h), tf.int32), tf.cast(tf.round(y1+ 2*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 3*sub_h), tf.int32), tf.cast(tf.round(y1+ 4*sub_h), tf.int32), tf.cast(y2, tf.int32)]
#                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+ sub_w), tf.int32), tf.cast(tf.round(x1+ 2*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 3*sub_w), tf.int32), tf.cast(tf.round(x1+ 4*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 5*sub_w), tf.int32), tf.cast(tf.round(x1+ 6*sub_w), tf.int32), tf.cast(x2, tf.int32)]
#                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+ sub_h), tf.int32), tf.cast(tf.round(y1+ 2*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 3*sub_h), tf.int32), tf.cast(tf.round(y1+ 4*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 5*sub_h), tf.int32), tf.cast(tf.round(y1+ 6*sub_h), tf.int32), tf.cast(y2, tf.int32)]
                    
                    sub_mask = []
                    for bin_y in range(self.k):
                        for bin_x in range(self.k):
                            sub_y1 = grid_y[bin_y]
                            sub_x1 = grid_x[bin_x]
                            sub_y2 = grid_y[bin_y + 1]
                            sub_x2 = grid_x[bin_x + 1]
                            sub_hh = sub_y2 - sub_y1
                            z0 = tf.zeros([sub_y1, size])
                            z1 = tf.concat([tf.zeros([sub_hh, sub_x1]),tf.ones([sub_hh, sub_x2 - sub_x1]),
                                            tf.zeros([sub_hh, size - sub_x2])],axis=1)
                            z2 = tf.zeros([size - sub_y2, size])
                            sub_mask.append(tf.expand_dims(tf.concat([z0, z1, z2], axis=0), axis = -1))
                    channel_mask = tf.concat(sub_mask, axis = -1)
                    return channel_mask
                
                # 5. assemble masks for positive_rois
                positiverois  = tf.round(positive_rois * tf.to_float(size))
                channel_masks = tf.map_fn(assemble_kmask_from_box, positiverois, dtype=tf.float32)
                pred_masks    = tf.tile(tf.expand_dims(pred_masks, 0), [tf.shape(positive_rois)[0], 1, 1, 1])
                pred_masks    = tf.reduce_sum(pred_masks * channel_masks, axis = -1)
                
                # 6. Only consider the mask loss inside positive_rois
                mask_object  = tf.reduce_sum(channel_masks, axis = -1)
                def f1():
                    maskloss = tf.reduce_sum(mask_object * tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=roi_masks, logits=pred_masks), axis = [1,2]) 
                    maskloss = self.mask_scale *tf.reduce_mean(maskloss / tf.reduce_sum(mask_object, axis = [1,2]))
                    return maskloss
                
                out_loss   = tf.cond(tf.size(positive_rois) > 0, lambda: f1(), lambda: tf.constant(0.0))
                mask_loss += out_loss
            
            eachmask_loss  = mask_loss / self.batchsize
            tf.losses.add_loss(eachmask_loss)
            tf.summary.scalar('mask_loss', eachmask_loss)

    def val_test(self, box_out, mask_out):
        '''
            output box and mask using GPU during validation and testing stage
            output: det_box  = [[num_box, 6(y1, x1, y2, x2, classid, class-conf],...]
                    det_mask = [[num_box, output height, output width],...]
        ''' 
        det_box = []
        det_mask = []
        for i in range(self.batchsize):
            proposals  = box_out[i, ...]
            pred_masks = mask_out[i, ...]
            size       = tf.shape(pred_masks)[1]

            # filter out paddings and the detections with -/0 width or height
            pred_boxes = tf.round(proposals[:, :4] * tf.to_float(size))
            keep_ix    = tf.where(tf.logical_and(((pred_boxes[:, 2]-pred_boxes[:, 0]) > 0), 
                                                 ((pred_boxes[:, 3]-pred_boxes[:, 1]) > 0)))[:, 0]
            proposals  = tf.gather(proposals, keep_ix)
            pred_boxes = tf.gather(pred_boxes, keep_ix)

            # assemble masks for predicted boxes
            def f1(predbox, predmask, proposal):
                def assemble_kmask_from_box(box):
                    y1    = box[0]
                    x1    = box[1]
                    y2    = box[2]
                    x2    = box[3]
                    w     = x2 - x1
                    h     = y2 - y1
                    sub_w = w / self.k
                    sub_h = h / self.k
                    
                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+sub_w), tf.int32), 
                              tf.cast(tf.round(x1+ 2*sub_w), tf.int32), tf.cast(x2, tf.int32)]
                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+sub_h), tf.int32), 
                              tf.cast(tf.round(y1+ 2*sub_h), tf.int32), tf.cast(y2, tf.int32)]     
#                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+ sub_w), tf.int32), tf.cast(tf.round(x1+ 2*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 3*sub_w), tf.int32), tf.cast(tf.round(x1+ 4*sub_w), tf.int32), tf.cast(x2, tf.int32)]
#                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+ sub_h), tf.int32), tf.cast(tf.round(y1+ 2*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 3*sub_h), tf.int32), tf.cast(tf.round(y1+ 4*sub_h), tf.int32), tf.cast(y2, tf.int32)]
#                    grid_x = [tf.cast(x1, tf.int32), tf.cast(tf.round(x1+ sub_w), tf.int32), tf.cast(tf.round(x1+ 2*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 3*sub_w), tf.int32), tf.cast(tf.round(x1+ 4*sub_w), tf.int32), 
#                              tf.cast(tf.round(x1+ 5*sub_w), tf.int32), tf.cast(tf.round(x1+ 6*sub_w), tf.int32), tf.cast(x2, tf.int32)]
#                    grid_y = [tf.cast(y1, tf.int32), tf.cast(tf.round(y1+ sub_h), tf.int32), tf.cast(tf.round(y1+ 2*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 3*sub_h), tf.int32), tf.cast(tf.round(y1+ 4*sub_h), tf.int32), 
#                              tf.cast(tf.round(y1+ 5*sub_h), tf.int32), tf.cast(tf.round(y1+ 6*sub_h), tf.int32), tf.cast(y2, tf.int32)]
                    
                    sub_mask = []
                    for bin_y in range(self.k):
                        for bin_x in range(self.k):
                            sub_y1 = grid_y[bin_y]
                            sub_x1 = grid_x[bin_x]
                            sub_y2 = grid_y[bin_y + 1]
                            sub_x2 = grid_x[bin_x + 1]
                            sub_hh = sub_y2 - sub_y1
                            z0 = tf.zeros([sub_y1, size])
                            z1 = tf.concat([tf.zeros([sub_hh, sub_x1]),tf.ones([sub_hh, sub_x2 - sub_x1]),
                                            tf.zeros([sub_hh, size - sub_x2])],axis=1)
                            z2 = tf.zeros([size - sub_y2, size])
                            sub_mask.append(tf.expand_dims(tf.concat([z0, z1, z2], axis=0), axis = -1))
                    channel_mask = tf.concat(sub_mask, axis = -1)
                    return channel_mask
                
                channel_masks = tf.map_fn(assemble_kmask_from_box, predbox, dtype=tf.float32)                
                predmask      = tf.tile(tf.expand_dims(predmask, 0), [tf.shape(predbox)[0], 1, 1, 1]) 
                assemble_pred_mask = tf.reduce_sum(predmask * channel_masks, axis = -1)
                assemble_pred_mask = tf.sigmoid(assemble_pred_mask)
                
                return assemble_pred_mask
            
            # if no proposal generated for current image, it is an empty array with shape of (0, 6)
            detmask = tf.cond(tf.size(proposals) > 0, lambda: f1(pred_boxes, pred_masks, proposals), lambda: tf.constant(0.0))
            
            det_box.append(proposals)
            det_mask.append(detmask)
            
        return [det_box, det_mask]

    def clip_boxes_graph(self, boxes, window):
        # boxes: [N, (y1, x1, y2, x2)]
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped
    
    def overlaps_graph(self, boxes1, boxes2):
        # Computes IoU overlaps between two sets of boxes[N, (y1, x1, y2, x2)]

        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections, if x2<x1 or y2<y1, then intersection = 0
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps
