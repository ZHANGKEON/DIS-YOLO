# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 9:28:11 2019

@author: Chaobo
"""

import os
import numpy as np
import cv2
import pickle as cPickle
import yolo.config as cfg

class defect_val(object):
    def __init__(self, phase):
        self.phase        = phase
        self.image_size   = cfg.TEST_SIZE
        self.val_path     = os.path.join(cfg.DATASET, self.phase)
        self.imagesetfile =  os.path.join(self.val_path, 'cache', 'val.txt')
        self.image_paths  = self.load_labels()
        self.num_labels   = len(self.image_paths)
        
    def get(self):
        clipwindow_val =  np.zeros((self.num_labels, 4), dtype=np.float32) 
        imagesval      = np.zeros((self.num_labels, self.image_size, self.image_size, 3), dtype=np.float32)
        img_names      = []
        for i in range(self.num_labels):
            imname = self.image_paths[i]
            image_array, window_array = self.image_read(imname)
            imagesval[i, :, :, :] = image_array
            clipwindow_val[i,:] = window_array
            img_names.append(os.path.splitext(os.path.basename(imname))[0])

        return imagesval, img_names, clipwindow_val

    def image_read(self, imname):
        
        window = np.array([0., 0., 1., 1.], dtype=np.float32)
        image = cv2.imread(imname)
        img_h, img_w, _ = image.shape
        if (float(self.image_size)/img_w) < (float(self.image_size)/img_h): 
            img_h = (img_h * self.image_size)//img_w
            img_w = self.image_size
        else:
            img_w = (img_w * self.image_size)//img_h
            img_h = self.image_size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (img_w, img_h),interpolation = cv2.INTER_LINEAR)

        # prepare the clip window for boxes in validation mode
        top       = (self.image_size - img_h)//2
        left      = (self.image_size - img_w)//2
        window[0] = top / self.image_size
        window[1] = left / self.image_size
        window[2] = (img_h + top) / self.image_size
        window[3] = (img_w + left) / self.image_size

        # embed the image intostandard letter box
        new_image = np.ones((self.image_size, self.image_size, 3)) * 127.
        new_image[(self.image_size - img_h)//2:(self.image_size + img_h)//2, 
                  (self.image_size - img_w)//2:(self.image_size + img_w)//2, :]= image  
        new_image = new_image / 255.0
        return new_image, window

    def load_labels(self):
        images_path = os.path.join(self.val_path, 'images')
        
        cache_path = os.path.join(self.val_path, 'cache')
        ground_truth_cache = os.path.join(cache_path, 'ground_truth_cache.pkl')
        with open(ground_truth_cache, 'rb') as f:
            annotations = cPickle.load(f)
        
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        with open(self.imagesetfile, 'r') as f:
            val_index = [x.strip() for x in f.readlines()]            
        assert len(val_index)==len(annotations)
                        
        imagefile_path  = []
        for i, index in enumerate(val_index):
            a = annotations[i]
            filename = os.path.splitext(a['filename'])[0]
            assert filename == index
            
            image_path = os.path.join(images_path, index + '.jpg')
            imagefile_path.append(image_path)
            
        return imagefile_path
