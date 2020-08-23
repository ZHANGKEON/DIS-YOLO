# -*- coding: utf-8 -*-
"""
Created on Thu Jun 6 9:10:18 2019

@author: Chaobo
"""

import os
import numpy as np
import cv2
import pickle as cPickle
import copy
import yolo.config as cfg
import random
import math
from pyblur import *
import skimage.draw

class defect_train(object):
    def __init__(self, phase):
        self.phase        = phase
        self.datapath     = os.path.join(cfg.DATASET, self.phase)
        self.batch_size   = cfg.BATCH_SIZE
        self.image_size   = cfg.IMAGE_SIZE
        self.base_grid    = cfg.BASE_GRID
        self.mask_size    = (self.image_size, self.image_size)
        
        self.max_box_per_image = cfg.MAX_BOX_PER_IMAGE
        self.anchors      = cfg.ANCHORS
        self.num_anchor   = 3
        self.num_class    = len(cfg.CLASSES)
        self.class_to_ind = dict(zip(cfg.CLASSES, range(self.num_class))) # {'crack':0, 'spall': 1. 'rebar': 2}
               
        self.flipped      = cfg.FLIPPED
        self.blur_noise_light = cfg.BLUR_NOISE_LIGHT        
        self.cursor       = 0
        self.epoch        = 1
        
        # load and format the ground-truth annotations
        self.gt_labels    = self.load_labels()
        np.random.shuffle(self.gt_labels)
        self.random_labels = copy.deepcopy(self.gt_labels)

    def get(self):
        clipwindow_train = np.zeros((self.batch_size, 4), dtype=np.float32)
        clipwindow_train[:, 0:4] = [0., 0., 1., 1.] # y1, x1, y2, x2
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        true_boxes = np.zeros((self.batch_size, 1, 1, 1, self.max_box_per_image, 5), dtype=np.float32) # 4 coordinates + classid
        true_masks = np.zeros((self.batch_size, self.max_box_per_image, self.mask_size[0], self.mask_size[1]), dtype=np.bool)
        yolo_1 = np.zeros((self.batch_size, self.base_grid, self.base_grid, self.num_anchor, 5 + self.num_class), dtype=np.float32)
        yolo_2 = np.zeros((self.batch_size, 2*self.base_grid, 2*self.base_grid, self.num_anchor, 5 + self.num_class), dtype=np.float32)
        yolo_3 = np.zeros((self.batch_size, 4*self.base_grid, 4*self.base_grid, self.num_anchor, 5 + self.num_class), dtype=np.float32)
        
        count = 0
        while count < self.batch_size: 
            net_h     = self.image_size
            net_w     = self.image_size
            grid_h    = self.base_grid
            grid_w    = self.base_grid            
            yolo1_val = np.zeros((grid_h, grid_w, self.num_anchor, 5 + self.num_class), dtype=np.float32)
            yolo2_val = np.zeros((2*grid_h, 2*grid_w, self.num_anchor, 5 + self.num_class), dtype=np.float32)
            yolo3_val = np.zeros((4*grid_h, 4*grid_w, self.num_anchor, 5 + self.num_class), dtype=np.float32)
            yolos     = [yolo3_val, yolo2_val, yolo1_val]
               
            # select one GT data
            onelabel  = self.random_labels[self.cursor]
            imname    = onelabel['imname']
            image     = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB)
            image_h, image_w, _ = image.shape
            
            # load GT masks in original image size [max_box_per_image, image_h, image_w]
            polygons    = onelabel['polygons']
            class_names = onelabel['class_names']
            if len(polygons) > self.max_box_per_image:
                print('More than ' + str(self.max_box_per_image) + ' instances in ' + imname)
                print('Maybe increase the value of max_box_per_image.')
                polygons = polygons[:self.max_box_per_image]
                class_names = class_names[:self.max_box_per_image]
            mask = self.load_mask(self.max_box_per_image, image_h, image_w, polygons)

            # load GT boxes (x1, y1, x2, y2, classid) in original image size
            bbox, mask_index = self.load_box(mask, class_names)
            bbox_index       = np.where(np.any(bbox[0,0,0], axis=1))[0]
            assert (mask_index==bbox_index).all()
            
            # Data augmentation 1-Step 1: random scale and crop (1: NO, 2: YES)
            scale_crop = np.random.randint(low=1, high=3)
            if scale_crop == 2:
                jitter = 0.2
                new_ar = image_w/image_h * np.random.uniform(1-jitter,1+jitter)/np.random.uniform(1-jitter,1+jitter)
                scale  = np.random.uniform(0.75, 1.5)
                if new_ar < 1:
                    new_h = int(scale * net_h);
                    new_w = int(new_h * new_ar);
                else:
                    new_w = int(scale * net_w)
                    new_h = int(new_w / new_ar)
                # place image
                dx = int(np.random.uniform(0, net_w - new_w))
                dy = int(np.random.uniform(0, net_h - new_h)) 
                sx, sy = float(new_w)/image_w, float(new_h)/image_h
                
                # keep all defects inside image for avoiding error cropping
                # you can remove the following codes for different use
                check_x1 = []
                check_y1 = []
                check_x2 = []
                check_y2 = []
                for j in bbox_index:
                    check_x1.append(bbox[0,0,0,j, 0] *sx + dx)
                    check_y1.append(bbox[0,0,0,j, 1] *sy + dy)
                    check_x2.append(bbox[0,0,0,j, 2] *sx + dx)
                    check_y2.append(bbox[0,0,0,j, 3] *sy + dy)
                x1_min = np.amin(check_x1)
                y1_min = np.amin(check_y1)
                x2_max = np.amax(check_x2)
                y2_max = np.amax(check_y2)
                if x1_min < 0 or y1_min < 0 or x2_max >= net_w or y2_max >= net_h:
                    scale_crop = 1
                    
            if scale_crop == 1:
                new_ar = image_w/image_h
                scale  = 1.0
                if new_ar < 1:
                    new_h = int(scale * net_h);
                    new_w = int(new_h * new_ar);
                else:
                    new_w = int(scale*net_w)
                    new_h = int(new_w/new_ar)
                dx = (net_w-new_w)//2
                dy = (net_h-new_h)//2
                sx, sy = float(new_w)/image_w, float(new_h)/image_h
 
            # prepare training input for yolos
            for index in bbox_index:
                cls_ind = int(bbox[0,0,0,index, 4])               
                # update the bbox to be [xc, yc, w, h] in self.image_size
                x1 = float(bbox[0,0,0,index, 0])
                y1 = float(bbox[0,0,0,index, 1])
                x2 = float(bbox[0,0,0,index, 2])
                y2 = float(bbox[0,0,0,index, 3])
                x1 = max(min(x1 * sx + dx, net_w - 1), 0)
                y1 = max(min(y1 * sy + dy, net_h - 1), 0)
                x2 = max(min(x2 * sx + dx, net_w - 1), 0)
                y2 = max(min(y2 * sy + dy, net_h - 1), 0) 
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                bbox[0, 0, 0, index, :4] = [boxes[0], boxes[1], boxes[2], boxes[3]]
                
                # determine the anchor (best iou) for this bounding box
                anchors_min   = -np.asarray(self.anchors / 2., dtype='float32')
                anchors_max   = -anchors_min
                anchors_areas = anchors_max[:,1]*anchors_max[:,0]*4
                box_wh    = boxes[2:4]
                half      = box_wh / np.array([2,2])
                box_half  = np.repeat(np.asarray(half, dtype='float32').reshape((1,2)), 3* self.num_anchor, axis=0)        
                box_min   = -box_half
                box_max   = box_half
                box_areas = box_half[:,0]*box_half[:,1]*4
                # computer the overlap
                intersect_min   = np.maximum(box_min, anchors_min)
                intersect_max   = np.minimum(box_max, anchors_max)
                intersect_box   = np.maximum(intersect_max-intersect_min, 0.)
                intersect_areas = intersect_box[:, 0]*intersect_box[:, 1]
                iou             = intersect_areas/(box_areas + anchors_areas - intersect_areas)
                maximum_iou     = np.max(iou)
                
                if maximum_iou > 0:
                    iou_index   = np.argmax(iou)
                    yolo        = yolos[iou_index//3]
                    grid_height = yolo.shape[0]
                    grid_width  = yolo.shape[1]
                    x_ind       = int(boxes[0] * grid_width / net_w)
                    y_ind       = int(boxes[1] * grid_height / net_h)
                    if yolos[iou_index//3][y_ind, x_ind, iou_index%3, 4] == 1:
                        continue
                    yolos[iou_index//3][y_ind, x_ind, iou_index%3, 0:4] = [boxes[0], boxes[1], boxes[2], boxes[3]]
                    yolos[iou_index//3][y_ind, x_ind, iou_index%3, 4]   = 1
                    yolos[iou_index//3][y_ind, x_ind, iou_index%3, 5 + cls_ind] = 1.
                else:
                    print('No anchor has iou > 0 for this ground-truth box')
            
            # save all parameters in onelabel
            onelabel['yolo1_obj'] = yolos[2]
            onelabel['yolo2_obj'] = yolos[1]
            onelabel['yolo3_obj'] = yolos[0]
            onelabel['true_box']  = bbox
            onelabel['sc']        = [scale_crop, new_w, new_h, dx, dy]
                        
            # Data augmentation-Step 2: randomly horizontal or vertical flip
            onelabel['flip'] = 1
            if self.flipped:
                flip = np.random.randint(low=1, high=4)
                if flip == 1: # no flip
                    onelabel['flip'] = 1
                elif flip == 2: # horizontal flip
                    onelabel['flip'] = 2                 
                    onelabel['true_box'][0,0,0,bbox_index, 0] = net_w - 1 - onelabel['true_box'][0,0,0,bbox_index, 0]
                    onelabel['yolo1_obj'] = onelabel['yolo1_obj'][:, ::-1, :, :]
                    onelabel['yolo2_obj'] = onelabel['yolo2_obj'][:, ::-1, :, :]
                    onelabel['yolo3_obj'] = onelabel['yolo3_obj'][:, ::-1, :, :]
                    for i in range(self.base_grid):
                        for j in range(self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo1_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo1_obj'][i, j, k, 0] = net_w - 1 - onelabel['yolo1_obj'][i, j, k, 0]
                    for i in range(2*self.base_grid):
                        for j in range(2*self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo2_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo2_obj'][i, j, k, 0] = net_w - 1 - onelabel['yolo2_obj'][i, j, k, 0]
                    for i in range(4*self.base_grid):
                        for j in range(4*self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo3_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo3_obj'][i, j, k, 0] = net_w - 1 - onelabel['yolo3_obj'][i, j, k, 0]
                elif flip == 3: # vertical flip
                    onelabel['flip'] = 3
                    onelabel['true_box'][0,0,0,bbox_index, 1] = net_h - 1 - onelabel['true_box'][0,0,0,bbox_index, 1]
                    onelabel['yolo1_obj'] = onelabel['yolo1_obj'][::-1, :, :, :]
                    onelabel['yolo2_obj'] = onelabel['yolo2_obj'][::-1, :, :, :]
                    onelabel['yolo3_obj'] = onelabel['yolo3_obj'][::-1, :, :, :]
                    for i in range(self.base_grid):
                        for j in range(self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo1_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo1_obj'][i, j, k, 1] = net_h - 1 - onelabel['yolo1_obj'][i, j, k, 1]
                    for i in range(2*self.base_grid):
                        for j in range(2*self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo2_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo2_obj'][i, j, k, 1] = net_h - 1 - onelabel['yolo2_obj'][i, j, k, 1]
                    for i in range(4*self.base_grid):
                        for j in range(4*self.base_grid):
                            for k in range (self.num_anchor):
                                if onelabel['yolo3_obj'][i, j, k, 4] == 1:
                                    onelabel['yolo3_obj'][i, j, k, 1] = net_h - 1 - onelabel['yolo3_obj'][i, j, k, 1]

            # Data augmentation-Step 3: random motion blur, add noise or change light
            onelabel['bnl'] = 1
            if self.blur_noise_light:            
                bnl = np.random.randint(low=1, high=5)
                if bnl == 1: # no effet
                    onelabel['bnl'] = 1
                elif bnl == 2: # motion blur
                    onelabel['bnl'] = 2
                elif bnl == 3: # add noise
                    onelabel['bnl'] = 3
                elif bnl == 4: # change light
                    onelabel['bnl'] = 4        
            
            sc_para = onelabel['sc']
            flip    = onelabel['flip']
            bnl     = onelabel['bnl']
            
            # Prepare the training inputs
            images[count, :, :, :] = self.image_read(image, sc_para, flip, bnl, mode='image')
            true_masks[count, :, :, :] = self.resize_mask(mask, sc_para, flip, bbox_index, mode='mask')
            onelabel['true_box'][..., 0:4] = onelabel['true_box'][..., 0:4] / self.image_size
            onelabel['yolo1_obj'][..., 0:4] = onelabel['yolo1_obj'][..., 0:4] / self.image_size
            onelabel['yolo2_obj'][..., 0:4] = onelabel['yolo2_obj'][..., 0:4] / self.image_size
            onelabel['yolo3_obj'][..., 0:4] = onelabel['yolo3_obj'][..., 0:4] / self.image_size
            true_boxes[count, :, :, :, :, :] = onelabel['true_box']
            yolo_3[count, :, :, :, :] = onelabel['yolo3_obj']
            yolo_2[count, :, :, :, :] = onelabel['yolo2_obj']
            yolo_1[count, :, :, :, :] = onelabel['yolo1_obj']
                      
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                self.random_labels = None
                np.random.shuffle(self.gt_labels)
                self.random_labels = copy.deepcopy(self.gt_labels) 
                self.cursor = 0
                self.epoch += 1
            
        return images, true_masks, true_boxes, yolo_3, yolo_2, yolo_1, clipwindow_train

    def load_labels(self):
        images_path = os.path.join(self.datapath, 'images')
        
        cache_path  = os.path.join(self.datapath, 'cache')
        gt_labels_cache = os.path.join(cache_path, 'gt_labels_' + self.phase + '.pkl')
        if os.path.isfile(gt_labels_cache):
            print('Loading training labels from: ' + gt_labels_cache)
            with open(gt_labels_cache, 'rb') as f:
                gt_labels = cPickle.load(f)
                print('Number of training data: ' +str(len(gt_labels)))
            return gt_labels
        
        ground_truth_cache = os.path.join(cache_path, 'ground_truth_cache.pkl')
        print('Processing training labels from: ' + ground_truth_cache)
        with open(ground_truth_cache, 'rb') as f:
            annotations = cPickle.load(f)
        
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        txtname = os.path.join(cache_path, 'train.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
            
        assert len(self.image_index)==len(annotations)
                        
        gt_labels = []
        for i, index in enumerate(self.image_index):
            a        = annotations[i]
            filename = os.path.splitext(a['filename'])[0]
            assert filename == index
            # Extract the class and polygons for each instance
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            class_names = [r['region_attributes'] for r in a['regions'].values()]
            image_path = os.path.join(images_path, a['filename'])
            gt_labels.append({'imname': image_path, 'class_names': class_names, 'polygons': polygons})
            
        print('Saving training labels to: ' + gt_labels_cache)
        with open(gt_labels_cache, 'wb') as f:
            cPickle.dump(gt_labels, f)
            print('Number of training data: ' +str(len(gt_labels)))
        return gt_labels

    def load_mask(self, max_box_per_image, image_h, image_w, polygons):
        
        mask = np.zeros([max_box_per_image, image_h, image_w], dtype=np.float32)
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
                    each_mask[rr, cc] = False # remove the inside background region
                    each_mask[np.array(y_points), np.array(x_points)] = True
            
            mask[i, :,:] = each_mask.astype(np.float32)
        return mask

    def load_box(self, mask_perimg, class_perimg):
        # return true_box as 4 coordinates + classid
        box_perimg = np.zeros((1, 1, 1, self.max_box_per_image, 5), dtype=np.float32)        
        mask_perimg = mask_perimg.astype(np.uint8)
        
        # remove the zero padded masks        
        mask_index = np.where(np.any(mask_perimg, axis= (1,2)))[0]
        
        assert len(mask_index)==len(class_perimg)        
        for index in mask_index:
            eachclass = class_perimg[index]
            classid   = self.class_to_ind[eachclass]
            eachmask  = mask_perimg[index,...]
            x1, y1, x2, y2 = self.extract_bboxes(eachmask)
            box_perimg[0, 0, 0, index, :5] = [x1, y1, x2, y2, classid]
        return box_perimg, mask_index

    def extract_bboxes(self, eachmask):
        """ Compute bounding boxes from masks.
            each mask: [height, width]. Mask pixels are either 1 or 0.
            Returns: bbox coordinate y1, x1, y2, x2.
        """
        m = eachmask
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]

        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        # as in python that the first index is included and the last is not
        x2 += 1
        y2 += 1
        return x1, y1, x2, y2

    def image_read(self, image, sc_para, flip, bnl, mode):

        new_w = sc_para[1]
        new_h = sc_para[2]
        dx = sc_para[3]
        dy = sc_para[4]
        
        # get input image in (net_w, net_h, 3)
        image = self.apply_random_scale_and_crop(image, new_w, new_h, dx, dy, mode)
        
        if flip == 1:
            image = image
        elif flip == 2:
            image = image[:, ::-1, :]
        elif flip == 3:
            image = image[::-1, :, :]

        if bnl == 1:
            image = image
        elif bnl == 2:
            image = self.add_salt_pepper_noise(image)   
        elif bnl == 3: 
            image = self.change_light(image) 
        elif bnl == 4:
#            image = self.gaussain_blur(image, 3, 0)
            image = self.linearmotion_blur3C(image)
        
        image = image.astype(np.float32)
        image = image / 255.0
        return image

    def resize_mask(self, mask, sc_para, flip, bbox_index, mode):

        new_w = sc_para[1]
        new_h = sc_para[2]
        dx = sc_para[3]
        dy = sc_para[4]
        
        nozero_mask= []
        for i in bbox_index:
            # return net_mask in (net_w, net_h, 1)
            net_mask = self.apply_random_scale_and_crop(mask[i], new_w, new_h, dx, dy, mode)

            if flip == 1:
                net_mask = net_mask
            elif flip == 2:
                net_mask = net_mask[:, ::-1, :]
            elif flip == 3:
                net_mask = net_mask[::-1, :, :]            
            
            net_mask = np.around(np.squeeze(net_mask)).astype(np.bool)
            nozero_mask.append(np.expand_dims(net_mask, axis=0))
            
        all_net_mask = np.concatenate(nozero_mask, axis=0)
        assert len(bbox_index) == all_net_mask.shape[0]
        
        padded_net_mask = np.zeros((self.max_box_per_image, all_net_mask.shape[1], all_net_mask.shape[2]), dtype=np.bool)
        padded_net_mask[:all_net_mask.shape[0], :, :] = all_net_mask
        
        return padded_net_mask

    def apply_random_scale_and_crop(self, image, new_w, new_h, dx, dy, mode):
        # resize the image to (new_w, new_h), and then place and pad the image to (net_w, net_h)
        
        net_w, net_h = self.image_size, self.image_size
        im_sized = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)

        if mode=='image':
            pad_value = 127
        elif mode=='mask':
            pad_value = 0.
            im_sized = np.expand_dims(im_sized, axis=-1)
        
        if dx > 0: 
            im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=pad_value)
        else:
            im_sized = im_sized[:,-dx:,:]
        if (new_w + dx) < net_w:
            im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=pad_value)
                   
        if dy > 0: 
            im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=pad_value)
        else:
            im_sized = im_sized[-dy:,:,:]
            
        if (new_h + dy) < net_h:
            im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=pad_value)
            
        return im_sized[:net_h, :net_w, :]
    
    def linearmotion_blur3C(self, img):
        """ 
            Performs motion blur on an image with 3 channels.
            Code modifed from https://github.com/debidatta/syndata-generation
        """
        def randomAngle(kerneldim):
            # Returns a random angle used to produce motion blurring
            kernelCenter     = int(math.floor(kerneldim/2))
            numDistinctLines = kernelCenter * 4
            validLineAngles  = np.linspace(0,180, numDistinctLines, endpoint = False)
            angleIdx         = np.random.randint(0, len(validLineAngles))
            return int(validLineAngles[angleIdx])
        
#        lineLengths = [3,5,7,9]
        lineLengths   = [3] # avoid disappear of thin crack
        lineTypes     = ["right", "left", "full"]
        lineLengthIdx = np.random.randint(0, len(lineLengths))
        lineTypeIdx   = np.random.randint(0, len(lineTypes)) 
        lineLength    = lineLengths[lineLengthIdx]
        lineType      = lineTypes[lineTypeIdx]
        lineAngle     = randomAngle(lineLength)

        blurred_img = img
        for i in range(3):
            imgdata = LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType)
            blurred_img[:,:,i] = np.array(imgdata, np.uint8).reshape(imgdata.size[1], imgdata.size[0])
            
        return blurred_img

    def gaussain_blur(self, img, max_filiter_size = 3, sigma = 0):
        
    	if max_filiter_size >= 3 :
    		filter_size = random.randint(3, max_filiter_size)
    		if filter_size % 2 == 0 :
    			filter_size += 1

    		out = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
    	return out 

    def add_salt_pepper_noise(self, im):

        row, col, _    = im.shape
        salt_vs_pepper = 0.2
        amount         = 0.004
        num_salt       = np.ceil(amount * im.size * salt_vs_pepper)
        num_pepper     = np.ceil(amount * im.size * (1.0 - salt_vs_pepper))
        
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
        im[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in im.shape]
        im[coords[0], coords[1], :] = 0
        return im
    
    def change_light(self, image):
        # change lighting in HLS space
        image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        image_HLS = np.array(image_HLS, dtype = np.float64)
        coeff = np.random.uniform() + 0.5
        image_HLS[:,:,1] = image_HLS[:,:,1] * coeff # scale channel 1-Lightness
        image_HLS[:,:,1][image_HLS[:,:,1] > 255]  = 255
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
        return image_RGB
