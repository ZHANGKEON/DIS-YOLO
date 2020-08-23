# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:41:42 2019

@author: Chaobo
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import yolo.config as cfg
import time
import pickle as cPickle
import skimage.draw
from yolo.yolo3_net_pos import YOLONet
from utils.voc_eval_mask import voc_eval

class MAP(object):
    
    def __init__(self, test_path, evaluation=True):
        self.num_class    = len(cfg.CLASSES)
        self.classid      = [i for i in range(self.num_class)]
        self.class_to_ind = dict(zip(cfg.CLASSES, range(self.num_class)))
        self.test_path    = test_path
        self.imagesetfile = os.path.join(self.test_path, 'cache', 'test.txt')
        if evaluation:            
            self.groundtruth = self.get_groundtruth()

    def get_groundtruth(self):
        cache_path = cache_path = os.path.join(self.test_path, 'cache')
        
        test_labels_cache = os.path.join(cache_path, 'gt_labels_' + 'test' + '.pkl')
        if os.path.isfile(test_labels_cache):
            print('Loading testing labels from: ' + test_labels_cache)
            with open(test_labels_cache, 'rb') as f:
                recs = cPickle.load(f)
                print('Number of testing data: ' +str(len(recs[0])))
            return recs      

        ground_truth_cache = os.path.join(cache_path, 'ground_truth_cache.pkl')
        print('Processing testing labels from: ' + ground_truth_cache)
        with open(ground_truth_cache, 'rb') as f:
            annotations = cPickle.load(f)
        
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        with open(self.imagesetfile, 'r') as f:
            test_index = [x.strip() for x in f.readlines()]        
        assert len(test_index)==len(annotations)
        
        recs_mask      = {}
        recs_mergemask = {}
        recs_size      = {}
        for i, index in enumerate(test_index):
            a        = annotations[i]
            filename = os.path.splitext(a['filename'])[0]
            assert filename == index
            
            polygons    = [r['shape_attributes'] for r in a['regions'].values()]
            class_names = [r['region_attributes'] for r in a['regions'].values()]

            # return a list[{'imageid':filename, 'classid':classid, 'difficult':int(0), 'mask':bool[image_h, image_w]]}]
            image_h, image_w         = a['size']
            mask_label,  merged_mask = self.load_masklabel(filename, image_h, image_w, polygons, class_names)
            
            recs_mask[index]      = mask_label
            recs_mergemask[index] = merged_mask
            recs_size[index]      = [image_h, image_w]
            
        recs = [recs_mask, recs_mergemask, recs_size, test_index]

        print('Saving testing labels to: ' + test_labels_cache)
        with open(test_labels_cache, 'wb') as f:
            cPickle.dump(recs, f)
            print('Number of testing data: ' +str(len(recs_mask)))
        return recs

    def load_masklabel(self, imname, image_h, image_w, polygons, class_names):

        mask = np.zeros([len(polygons), image_h, image_w], dtype=np.bool)
        merged_annotatemask = np.zeros((image_h, image_w), dtype=np.uint8)
        
        for i, each_instance in enumerate(polygons):
            each_mask = np.zeros([image_h, image_w], dtype=np.bool)
            for each_poly in each_instance:
                subtype = each_poly['type']
                x_points = each_poly['all_points_x']
                y_points = each_poly['all_points_y']
                rr, cc = skimage.draw.polygon(y_points, x_points)
                if subtype == 'out':                
                    each_mask[rr, cc] = True
                    each_mask[np.array(y_points), np.array(x_points)] = True
                else:
                    each_mask[rr, cc] = False
                    each_mask[np.array(y_points), np.array(x_points)] = True
                          
            mask[i, :,:] = each_mask

            # generate merged mask for computing mIoU
            if class_names[i] == 'crack':
                merged_annotatemask[mask[i,...]==True] = 1
            elif class_names[i] == 'spall':
                merged_annotatemask[mask[i,...]==True] = 2
            elif class_names[i] == 'rebar':
                merged_annotatemask[mask[i,...]==True] = 3  

        # generate masklabel for computing mask-level mAP
        mask_index = np.where(np.any(mask, axis= (1,2)))[0]
        assert len(mask_index)==len(class_names)
        masklabel  = []
        for index in mask_index:
            eachclass = class_names[index]
            classid   = self.class_to_ind[eachclass]
            eachmask  = mask[index,...]
            masklabel.append({'imageid': imname, 'classid': classid, 'difficult': int(0), 'mask': eachmask})
            
        return masklabel, merged_annotatemask

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

def image_read(image_rgb, image_size):
    
    window = np.array([0., 0., 1., 1.], dtype=np.float32)
    imgh, imgw, _ = image_rgb.shape
    if (float(image_size)/imgw) < (float(image_size)/imgh): 
        imgh = (imgh * image_size)//imgw
        imgw = image_size
    else:
        imgw = (imgw * image_size)//imgh
        imgh = image_size

    image = image_rgb.astype(np.float32)
    image = cv2.resize(image, (imgw, imgh),interpolation = cv2.INTER_LINEAR)

    # prepare the window for clip_boxes in testing mode
    top = (image_size - imgh)//2
    left = (image_size - imgw)//2
    window[0] = top / image_size
    window[1] = left / image_size
    window[2] = (imgh + top) / image_size
    window[3] = (imgw + left) / image_size

    # embed the image into standard letter box
    new_image = np.ones((image_size, image_size, 3)) * 127.
    new_image[(image_size - imgh)//2:(image_size + imgh)//2, 
              (image_size - imgw)//2:(image_size + imgw)//2, :]= image  
    new_image = new_image / 255.0
    return new_image, window


''' Computing mask-level mAP and mIoU '''
def evaluate(weights_file, test_path, net, eval_map):

    sess  = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, weights_file)
    
    txtname = os.path.join(test_path, 'cache', 'test.txt')
    with open(txtname, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]

    val_mask      = eval_map.groundtruth[0]
    val_mergemask = eval_map.groundtruth[1]
    val_index     = eval_map.groundtruth[3]
        
    t_prediction    = 0
    t_crop_assemble = 0 
    
    det_masks = {}
    detfile   = {}
    cracklist = []
    spalllist = []
    rebarlist = []   
    for i, index in enumerate(image_index):
        print(index)
        assert index==val_index[i]
        
        imname    = os.path.join(test_path, 'images', index + '.jpg')
        image_rgb = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB) 
        image_h, image_w, _ = image_rgb.shape        
        input_image, input_window = image_read(image_rgb, cfg.TEST_SIZE)
        image_array  = np.expand_dims(input_image, 0)
        window_array =  np.expand_dims(input_window, 0)
        
        feed_val = {net.is_training: False, net.det_thresh: [np.float32(cfg.OBJ_THRESHOLD)], 
                    net.clip_window: window_array, net.images: image_array}

        t = time.time()
        det_box, det_mask = sess.run(net.evaluation, feed_dict=feed_val)
        t_prediction += (time.time() - t)
        
        if np.sum(det_mask[0]) == 0.0:
            merged_detectmask = np.zeros((image_h, image_w), dtype=np.uint8)
            det_masks[index] = merged_detectmask
            continue
        
        proposals   = det_box[0][:, :4]
        classids    = (det_box[0][:, 4]).astype(int)
        class_confs = det_box[0][:, 5]
        mask_out    = det_mask[0]
        
        merged_detectmask = np.zeros((image_h, image_w), dtype=np.uint8)
        # correct the boxes and masks into original image size
        for k in range(len(classids)):
            classid   = classids[k]
            score     = class_confs[k]
            pred_mask = mask_out[k]
            
            # correct boxes
            y1_norm, x1_norm, y2_norm, x2_norm = proposals[k,:]
            x1, y1, x2, y2 = eval_map.correct_yolo_boxes(x1_norm, y1_norm, x2_norm, y2_norm, image_h, image_w, cfg.TEST_SIZE, cfg.TEST_SIZE)

            if (y2-y1)*(x2-x1) <= 0:
                continue

            # correct masks
            t = time.time()
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
            t_crop_assemble += (time.time() - t)
            
            if classid==0:
                cracklist.append({'imageid': index, 'score': score, 'mask': full_mask})
                merged_detectmask[full_mask==True] = 1
            elif classid==1:
                spalllist.append({'imageid': index, 'score': score, 'mask': full_mask})
                merged_detectmask[full_mask==True] = 2
            elif classid==2:
                rebarlist.append({'imageid': index, 'score': score, 'mask': full_mask})
                merged_detectmask[full_mask==True] = 3

        det_masks[index] = merged_detectmask 
        
    detfile['0']=cracklist
    detfile['1']=spalllist
    detfile['2']=rebarlist

    # compute mask-level AP and mAP
    thresh     = 0.5
    thresh_out = []
    res        = []
    pres       = []
    aps        = []
    for i, clsid in enumerate(eval_map.classid):
        if not detfile[str(clsid)]:
            recall    = 0.
            precision = 0.
            ap        = 0.
            res      += [recall]
            pres     += [precision]                    
            aps      += [ap]
            continue
        recall, precision, ap = voc_eval(detfile[str(clsid)], val_mask, txtname, 
                                                                clsid, ovthresh= thresh, use_07_metric = False)
        res  += [recall]
        pres += [precision]
        aps  += [ap]
        
    mean_rec  = np.mean(res)
    mean_prec = np.mean(pres)
    mean_ap   = np.mean(aps)    
    thresh_out.append({'thresh': thresh, 'AP': aps, 'mAP': [mean_rec, mean_prec, mean_ap]})
 
    t_prediction = t_prediction + t_crop_assemble
    print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction / len(image_index)))

    # compute semantic segmentation accuracy mIoU
    p_bg    = [0, 0, 0, 0] 
    p_crack = [0, 0, 0, 0]
    p_spall = [0, 0, 0, 0]
    p_rebar = [0, 0, 0, 0]      
    
    num_all_true_pixels = 0
    for index in val_index:
        true_mask = val_mergemask[index]
        pred_mask = det_masks[index]
        assert true_mask.shape == pred_mask.shape
        
        num_all_true_pixels = num_all_true_pixels + int(true_mask.shape[0] * true_mask.shape[1])
        
        # prediction = background(bg)
        p_bg[0]    = p_bg[0] + np.sum((true_mask==0) * (pred_mask==0))
        p_crack[0] = p_crack[0] + np.sum((true_mask==1) * (pred_mask==0))
        p_spall[0] = p_spall[0] + np.sum((true_mask==2) * (pred_mask==0))
        p_rebar[0] = p_rebar[0] + np.sum((true_mask==3) * (pred_mask==0))
        # prediction = crack
        p_bg[1]    = p_bg[1] + np.sum((true_mask==0) * (pred_mask==1))
        p_crack[1] = p_crack[1] + np.sum((true_mask==1) * (pred_mask==1))
        p_spall[1] = p_spall[1] + np.sum((true_mask==2) * (pred_mask==1))
        p_rebar[1] = p_rebar[1] + np.sum((true_mask==3) * (pred_mask==1))
        # prediction = spall
        p_bg[2]    = p_bg[2] + np.sum((true_mask==0) * (pred_mask==2))
        p_crack[2] = p_crack[2] + np.sum((true_mask==1) * (pred_mask==2))
        p_spall[2] = p_spall[2] + np.sum((true_mask==2) * (pred_mask==2))
        p_rebar[2] = p_rebar[2] + np.sum((true_mask==3) * (pred_mask==2))
        # prediction = rebar
        p_bg[3]    = p_bg[3] + np.sum((true_mask==0) * (pred_mask==3))
        p_crack[3] = p_crack[3] + np.sum((true_mask==1) * (pred_mask==3))
        p_spall[3] = p_spall[3] + np.sum((true_mask==2) * (pred_mask==3))
        p_rebar[3] = p_rebar[3] + np.sum((true_mask==3) * (pred_mask==3))
    
    bg_iou    = p_bg[0] / (np.sum(p_bg) + p_bg[0] + p_crack[0] + p_spall[0] + p_rebar[0] - p_bg[0])
    crack_iou = p_crack[1] / (np.sum(p_crack) + p_bg[1] + p_crack[1] + p_spall[1] + p_rebar[1] - p_crack[1])
    spall_iou = p_spall[2] / (np.sum(p_spall) + p_bg[2] + p_crack[2] + p_spall[2] + p_rebar[2] - p_spall[2])
    rebar_iou = p_rebar[3] / (np.sum(p_rebar) + p_bg[3] + p_crack[3] + p_spall[3] + p_rebar[3] - p_rebar[3])
    miou      = np.mean([bg_iou, crack_iou, spall_iou, rebar_iou])
    
    mask_acc  = [bg_iou, crack_iou, spall_iou, rebar_iou, miou]
    
    return thresh_out, mask_acc

                    
if __name__ == '__main__':   
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    
    cfg.BATCH_SIZE = 1
    
    yolo = YOLONet(False)
    
    test_path   = os.path.join(cfg.DATASET, "test")
    test_weight = os.path.join(cfg.OUTPUT_DIR, "TRAINED MODEL")
    
    eval_map = MAP(test_path, evaluation=True)    

    thresh_out, mask_acc = evaluate(test_weight, test_path, yolo, eval_map)
    
    print('AP of each class:  ' + '  crack ' + str(format(thresh_out[0]['AP'][0], '.3f'))
                                + '  spall ' + str(format(thresh_out[0]['AP'][1], '.3f'))
                                + '  rebar ' + str(format(thresh_out[0]['AP'][2], '.3f')))
    print('mAP:  ' + '  recall '    + str(format(thresh_out[0]['mAP'][0], '.3f'))
                   + '  precision ' + str(format(thresh_out[0]['mAP'][1], '.3f'))
                   + '  mAP '       + str(format(thresh_out[0]['mAP'][2], '.3f')))
