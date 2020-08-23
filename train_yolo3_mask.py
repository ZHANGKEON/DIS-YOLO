# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:29:10 2019

@author: Chaobo
"""

import tensorflow as tf
import datetime
import os
import numpy as np
import yolo.config as cfg
from yolo.yolo3_net_pos import YOLONet
from utils.timer import Timer
from utils.train_data import defect_train
from utils.val_data import defect_val
from utils.validation_map import MAP
slim = tf.contrib.slim

class Solver(object):   

    def __init__(self, net, data, evalu): 
        self.net           = net
        self.data          = data
        self.eval          = evalu
        self.start_iter    = 1
        self.max_iter      = cfg.MAX_ITER
        self.summary_iter  = cfg.SUMMARY_ITER
        self.save_iter     = cfg.SAVE_ITER
        self.ckpt_dir      = os.path.join(cfg.OUTPUT_DIR, 'checkpoint')  
        self.loss_dir      = os.path.join(cfg.OUTPUT_DIR, 'lossnp')
        self.ckpt_file     = os.path.join(self.ckpt_dir,  'model.ckpt')
        self.save_cfg()
        
        self.summary_op    = tf.summary.merge_all()
        self.writer        = tf.summary.FileWriter(self.ckpt_dir, flush_secs = 60)
        self.global_step   = tf.Variable(0, trainable=False)
        self.learning_rate = 1e-4
        
        # deal with variables
        all_var_list       = tf.global_variables()                      
        all_train_list     = tf.trainable_variables()
        print("*** Train variables ***")    
        for idxs, vs in enumerate(all_train_list):
             print("  param {:3}: {:15}   {}".format(idxs, str(vs.get_shape()), vs.name))

        save_list          = []
        conv_save          = range(1, 83) # 82 conv layers from yolo3_net_pos.py
        for i in conv_save:
            conv_name = 'convolutional' + str(i)
            conv_var_save = [val  for val in all_var_list if conv_name in val.name]
            save_list += conv_var_save  
        save_list = sorted(set(save_list), key=save_list.index)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)        

        self.saver = tf.train.Saver(var_list=save_list, max_to_keep=None)
        
        # For Training Stage 2, restore pretrained weights for all layers
        self.restore_all = tf.train.Saver(var_list=save_list , max_to_keep=None)
        
        # gpu_options
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        restore_weight = True
        if restore_weight:
            print('Restoring weights from: ' + cfg.WEIGHTS_FILE)       

            # Training Stage 1: fine-tuning network heads (yolov3 head + mask subnet)
            # restore pretrained weight "yolov3_3class_coco.ckpt"
            include = []
            conv_bn = []
            for i in range(1, 53):
                conv_bn.append(i)
            for i in range(53, 59):
                conv_bn.append(i)
            for i in range(60, 67):
                conv_bn.append(i)
            for i in range(68, 75):
                conv_bn.append(i)    
            for i in conv_bn:
                conv_name = 'convolutional' + str(i)
                k_w  = 'yolo' + '/' + conv_name + '/' + 'weights'
                k_b  = 'yolo' + '/' + conv_name + '/' + 'BatchNorm' + '/beta'
                k_s  = 'yolo' + '/' + conv_name + '/' + 'BatchNorm' + '/gamma'
                k_pm = 'yolo' + '/' + conv_name + '/' + 'BatchNorm' + '/moving_mean'
                k_pv = 'yolo' + '/' + conv_name + '/' + 'BatchNorm' + '/moving_variance'
                include.append(k_w)
                include.append(k_b)            
                include.append(k_s)
                include.append(k_pm)
                include.append(k_pv)
            conv_layer = [59, 67, 75]
            for i in conv_layer:
                scope = 'convolutional' + str(i)
                k_w   = 'yolo' + '/' + scope + '/' + 'weights'
                k_b   = 'yolo' + '/' + scope + '/' + 'biases'
                include.append(k_w)
                include.append(k_b)
            variables_to_restore = slim.get_variables_to_restore(include=include)
            init_fn = slim.assign_from_checkpoint_fn(cfg.WEIGHTS_FILE, variables_to_restore, 
                                                     ignore_missing_vars=True)
            init_fn(self.sess)
             
            # Training Stage 2: fine-tuning all layers
            # restore fully pretrained weight, please change WEIGHTS_FILE and MAX_ITER in config
#            self.restore_all.restore(self.sess, cfg.WEIGHTS_FILE)
            
            assign_op = self.global_step.assign(0)
            self.sess.run(assign_op)
        self.writer.add_graph(self.sess.graph)
        
    def train(self):
        load_timer = Timer()
        train_timer = Timer()
        val_map = np.zeros((800, 9))
        imagesval, imageidsval, window_vals = defect_val('val').get()
        num_val = len(imageidsval)
        if not(num_val % cfg.BATCH_SIZE == 0):
            print('Please manually change the number of validation data.')
        
        epoch_loss = 0.0
        for step in range(self.start_iter, self.max_iter + 1):

            # Training Stage 1: fine-tuning network heads for 20 epochs, MAX_ITER=10000    
            if step <= 10000:                    # 20epoch
                self.learning_rate = 1e-3
                
            # Training Stage 2: fine-tuning all layers for 60 epochs,MAX_ITER=30000
            if step <= 10000:                    # 20epoch
                self.learning_rate = 1e-3
            elif step > 10000 and step <= 20000: # 20epoch
                self.learning_rate = 1e-4
            elif step > 20000 and step <= 25000: # 10epoch
                self.learning_rate = 1e-5 
            elif step > 25000 and step <= 30000: # 10epoch
                self.learning_rate = 1e-6 
                
            load_timer.tic()

            images, true_masks, true_boxes, yolo_3, yolo_2, yolo_1, window_train = self.data.get()
            feed_dict = {self.net.is_training: True, self.net.det_thresh: [np.float32(cfg.OBJ_THRESHOLD)], 
                         self.net.clip_window: window_train, self.net.images: images, 
                         self.net.true_boxes: true_boxes, self.net.true_masks: true_masks,
                         self.net.yolo1: yolo_1, self.net.yolo2: yolo_2, self.net.yolo3: yolo_3}
            
            load_timer.toc()
            
            if step % self.summary_iter == 0:
                
                # do validation and print accuracy at the end of each epoch
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.optimizer], 
                                                         feed_dict=feed_dict)
                    train_timer.toc()
                    epoch_loss += loss
                    
                    detect = []
                    for v in range (num_val//cfg.BATCH_SIZE):
                        start_step = int(cfg.BATCH_SIZE * v)
                        end_step   = int(cfg.BATCH_SIZE * v + cfg.BATCH_SIZE)
                        inputs     = imagesval[start_step : end_step,...]
                        window_val = window_vals[start_step : end_step, :]
                        imgnames   = [imageidsval[i] for i in range(start_step, end_step)]
                        
                        feed_val   = {self.net.is_training: False, self.net.det_thresh: [np.float32(cfg.OBJ_THRESHOLD)], 
                                                              self.net.clip_window: window_val, self.net.images: inputs}

                        det_boxes, det_masks = self.sess.run(self.net.evaluation, feed_dict=feed_val)
                        detect.extend([{'boxes': det_boxes[i], 'masks': det_masks[i], 
                                        'imname': imgnames[i]} for i in range(cfg.BATCH_SIZE)])
                    
                    thresh_out = self.eval.do_python_eval(detect)[0]
                    
                    record_loss = epoch_loss/self.save_iter
                    val_map[int(step/(self.summary_iter * 10))-1, :] = [step, self.data.epoch, record_loss, 
                            thresh_out['AP'][0], thresh_out['AP'][1], thresh_out['AP'][2], 
                            thresh_out['mAP'][0], thresh_out['mAP'][1], thresh_out['mAP'][2]]
                             
                    log_str = ('{} Epoch: {}, Step: {}, Image: {}, Batch: {}, Learning rate: {},'
                        ' Loss: {:5.3f}, crack: {:5.3f}, spall: {:5.3f}, rebar: {:5.3f}, mAP50: {:5.3f},'
                        '\nSpeed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        self.data.image_size,
                        self.data.batch_size,
                        round(self.learning_rate, 6),
                        record_loss,
                        thresh_out['AP'][0],
                        thresh_out['AP'][1],
                        thresh_out['AP'][2],
                        thresh_out['mAP'][2],
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)
                    
                    epoch_loss = 0.0
                    
                else:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.optimizer], 
                                                         feed_dict=feed_dict)
                    train_timer.toc()
                    epoch_loss += loss
                self.writer.add_summary(summary_str, step)     

            else:
                train_timer.tic()
                loss, _ = self.sess.run([self.net.total_loss, self.optimizer], feed_dict=feed_dict)    
                train_timer.toc()
                epoch_loss += loss
                
            # save the checkpoint and loss files every SAVE_ITER
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.ckpt_dir))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=step)
                np.save(os.path.join(self.loss_dir,str(step) + 'map.npy'), val_map)

    def save_cfg(self):
        with open(os.path.join(self.ckpt_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo   = YOLONet(training=True)
    defect = defect_train('train')
    evalu  = MAP()

    solver_train = Solver(yolo, defect, evalu)

    print('Start training ...')
    solver_train.train()
    print('Done training.')

if __name__ == '__main__':

    main()
