# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:10:24 2019

@author: Chaobo
"""

import numpy as np
import os
import yolo.config as cfg
import pickle as cPickle
import skimage.draw
import cv2
from utils.voc_eval_mask import voc_eval

class MAP(object):
    def __init__(self):
        self.num_class    = len(cfg.CLASSES)
        self.classid      = [i for i in range(self.num_class)]
        self.class_to_ind = dict(zip(cfg.CLASSES, range(self.num_class)))
        self.val_path     = os.path.join(cfg.DATASET, 'val')      
        self.imagesetfile =  os.path.join(self.val_path, 'cache','val.txt')
        self.groundtruth  = self.get_groundtruth()

    def get_groundtruth(self):
        cache_path = cache_path = os.path.join(self.val_path, 'cache')
        
        val_labels_cache = os.path.join(cache_path, 'gt_labels_' + 'val' + '.pkl')
        if os.path.isfile(val_labels_cache):
            print('Loading validation labels from: ' + val_labels_cache)
            with open(val_labels_cache, 'rb') as f:
                recs = cPickle.load(f)
                print('Number of validation data: ' +str(len(recs[0])))
            return recs      

        ground_truth_cache = os.path.join(cache_path, 'ground_truth_cache.pkl')
        print('Processing validation labels from: ' + ground_truth_cache)
        with open(ground_truth_cache, 'rb') as f:
            annotations = cPickle.load(f)
        
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        with open(self.imagesetfile, 'r') as f:
            val_index = [x.strip() for x in f.readlines()]        
        assert len(val_index)==len(annotations)
        
        recs_mask = {}
        recs_size = {}
        for i, index in enumerate(val_index):
            a = annotations[i]
            filename = os.path.splitext(a['filename'])[0]
            assert filename == index
            
            polygons    = [r['shape_attributes'] for r in a['regions'].values()]
            class_names = [r['region_attributes'] for r in a['regions'].values()]

            # format GT annotation for evaluation
            image_h, image_w = a['size']
            mask_label = self.load_masklabel(filename, image_h, image_w, polygons, class_names)            
            recs_mask[index] = mask_label
            recs_size[index] = [image_h, image_w]
            
        recs = [recs_mask, recs_size, val_index]

        print('Saving validation labels to: ' + val_labels_cache)
        with open(val_labels_cache, 'wb') as f:
            cPickle.dump(recs, f)
            print('Number of validation data: ' +str(len(recs_mask)))
        return recs

    def load_masklabel(self, imname, image_h, image_w, polygons, class_names):
        
        # each polygon=list[dict{'name':str, 'type':'in/out', 'all_points_x':[...], 'all_points_y':[...]}]
        mask = np.zeros([len(polygons), image_h, image_w], dtype=np.bool)
        for i, each_instance in enumerate(polygons):
            each_mask = np.zeros([image_h, image_w], dtype=np.bool)
            for each_poly in each_instance:
                subtype  = each_poly['type']
                x_points = each_poly['all_points_x']
                y_points = each_poly['all_points_y']
                rr, cc   = skimage.draw.polygon(y_points, x_points)
                if subtype == 'out':                
                    each_mask[rr, cc] = True
                    each_mask[np.array(y_points), np.array(x_points)] = True
                else:
                    each_mask[rr, cc] = False
                    each_mask[np.array(y_points), np.array(x_points)] = True                
                
            mask[i, :,:] = each_mask
            
        # generate mask lable for computing mask-level mAP
        mask_index = np.where(np.any(mask, axis= (1,2)))[0]
        assert len(mask_index)==len(class_names)
        masklabel = []
        for index in mask_index:
            eachclass = class_names[index]
            classid   = self.class_to_ind[eachclass]
            eachmask  = mask[index,...]
            masklabel.append({'imageid': imname, 'classid': classid, 'difficult': int(0), 'mask': eachmask})
            
        return masklabel

    def do_python_eval(self, detdata):
        '''
            detdata = list[{'boxes':  , 'masks':  , 'imname':  }]
                -boxes: box num[array(y1  x1  y2  x2  classid  class-conf)] 
                -masks: box num[array(mask_height, mask_width)] after sigmoid activation
                -imname: image id                
            return the AP of each class and mAP at the mask-level iou of 0.5
        '''
        val_mask = self.groundtruth[0]
        val_size = self.groundtruth[1]
        val_index= self.groundtruth[2]
        
        assert len(detdata) == len(val_size) == len(val_index)
        
        detfile = {} 
        cracklist = []
        spalllist = []
        rebarlist = []
        for i in range(len(detdata)):
            imageid = detdata[i]['imname']
            assert imageid==val_index[i]
            
            image_h, image_w = val_size[imageid]
            
            if np.sum(detdata[i]['masks']) == 0.0:
                continue            
 
            proposals   = detdata[i]['boxes'][:, :4]
            classids    = (detdata[i]['boxes'][:, 4]).astype(int)
            class_confs = detdata[i]['boxes'][:, 5]
            mask_out    = detdata[i]['masks']
            
            # correct the boxes and masks into original image size
            for k in range(len(classids)):
                classid = classids[k]
                score = class_confs[k]
                pred_mask = mask_out[k]
                
                # correct boxes
                y1_norm, x1_norm, y2_norm, x2_norm = proposals[k,:]
                x1, y1, x2, y2 = self.correct_yolo_boxes(x1_norm, y1_norm, x2_norm, y2_norm, image_h, image_w, cfg.TEST_SIZE, cfg.TEST_SIZE)

                if (y2-y1)*(x2-x1) <= 0:
                    continue

                # correct masks
                size      = pred_mask.shape[0]
                y1_norm   = np.around(y1_norm * size).astype(np.int32)
                x1_norm   = np.around(x1_norm * size).astype(np.int32)
                y2_norm   = np.around(y2_norm * size).astype(np.int32)
                x2_norm   = np.around(x2_norm * size).astype(np.int32)
                crop_mask = pred_mask[y1_norm:y2_norm, x1_norm:x2_norm]
                mask      = cv2.resize(crop_mask, (x2 - x1, y2 - y1), interpolation = cv2.INTER_LINEAR)
                mask      = np.where(mask > 0.5, 1, 0).astype(np.bool)
                full_mask = np.zeros([image_h, image_w], dtype=np.bool)
                full_mask[y1:y2, x1:x2] = mask
                
                if classid==0:
                    cracklist.append({'imageid': imageid, 'score': score, 'mask': full_mask})
                elif classid==1:
                    spalllist.append({'imageid': imageid, 'score': score, 'mask': full_mask})
                elif classid==2:
                    rebarlist.append({'imageid': imageid, 'score': score, 'mask': full_mask})
                    
        detfile['0']=cracklist
        detfile['1']=spalllist
        detfile['2']=rebarlist

        # compute mask-level AP and mAP
        thresh     = 0.5
        thresh_out = []
        res        = []
        pres       = []
        aps        = []
        for i, clsid in enumerate(self.classid):
            if not detfile[str(clsid)]:
                recall    = 0.
                precision = 0.
                ap        = 0.
                res      += [recall]
                pres     += [precision]                    
                aps      += [ap]
                continue
            recall, precision, ap = voc_eval(detfile[str(clsid)], val_mask, self.imagesetfile, 
                                                                    clsid, ovthresh= thresh, use_07_metric = False)
            res  += [recall]
            pres += [precision]
            aps  += [ap]
            
        mean_rec  = np.mean(res)
        mean_prec = np.mean(pres)
        mean_ap   = np.mean(aps)
        thresh_out.append({'thresh': thresh, 'AP': aps, 'mAP': [mean_rec, mean_prec, mean_ap]})
        
        return thresh_out

    def correct_yolo_boxes(self, x1, y1, x2, y2, image_h, image_w, net_h, net_w):

        if (float(net_w)/image_w) < (float(net_h)/image_h):
            new_w = net_w
            new_h = (image_h*net_w)//image_w
        else:
            new_h = net_h
            new_w = (image_w*net_h)//image_h
            
        x_offset, x_scale = float((net_w - new_w)//2)/net_w, float(new_w)/net_w
        y_offset, y_scale = float((net_h - new_h)//2)/net_h, float(new_h)/net_h
        
        x1 = max(min(np.around((x1 - x_offset) / x_scale * image_w).astype(np.int32), image_w), 0)
        x2 = max(min(np.around((x2 - x_offset) / x_scale * image_w).astype(np.int32), image_w), 0)
        y1 = max(min(np.around((y1 - y_offset) / y_scale * image_h).astype(np.int32), image_h), 0)
        y2 = max(min(np.around((y2 - y_offset) / y_scale * image_h).astype(np.int32), image_h), 0)
        
        return x1, y1, x2, y2

#    def sigmoid(self, x):
#        return 1. / (1. + np.exp(-x))

    def sigmoid(self, x):
        a = -1. * x
        a = np.clip(a, -50., 50.)
        a = 1. / (1. + np.exp(a))
        return a
