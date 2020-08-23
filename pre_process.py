# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:20:13 2019

@author: Chaobo
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle as cPickle
import cv2
import skimage.draw
from PIL import Image,ImageDraw,ImageFont

def load_verify_contour(data_path, phase = 'train'):
    data_path = os.path.join(data_path, phase)
    annotation_path = os.path.join(data_path, 'annotations')
    rgb_path = os.path.join(data_path, 'images')
    updated_mask = os.path.join(data_path, 'masks')
    cache_path = os.path.join(data_path, 'cache')
    verify_path = os.path.join(data_path, 'verify')
    
    # verify the extracted contour and bounding box, image saved in "verify"
    do_verification = False
    
    ground_truth_cache = os.path.join(cache_path, 'ground_truth_cache.pkl')
    if os.path.isfile(ground_truth_cache):
        print('Loading gt_labels from: ' + ground_truth_cache)
        with open(ground_truth_cache, 'rb') as f:
            gt_data = cPickle.load(f)
        return gt_data 
    
    f_wrect = open(os.path.join(cache_path, phase + '.txt'), 'w') # creat image ID text
    annotations = []
    imgfile = os.listdir(rgb_path)
    error_mask = 0
    for i in range(len(imgfile)):
        file = imgfile[i]
        filename = os.path.splitext(file)[0]
        print(filename)
        f_wrect.write(filename+'\n')
 
        #Load image, load bounding box info from XML file in PASCAL VOC format
        annoname = os.path.join(annotation_path, filename + '.xml')
        if os.path.exists(annoname):
            objects = []
            tree = ET.parse(annoname)
            objs = tree.findall('object')
            for obj in objs:
                obj_struct = {}       
                cls_name = obj.find('name').text.lower().strip()
                obj_struct['class'] = cls_name
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                obj_struct['bbox'] = [x1, y1, x2, y2]
                objects.append(obj_struct)
            # extract 'merge' box in list[[x1, y1, x2, y2],[x1, y1, x2, y2]...]  
            object_merge = [obj['bbox'] for obj in objects if obj['class']=='merge']
                  
        rgb_file = os.path.join(rgb_path, filename + '.jpg')
        spallmask_file = os.path.join(updated_mask, filename + 'spall' + '.jpg')
        rebarmask_file = os.path.join(updated_mask, filename + 'rebar' + '.jpg')
        crackmask_file = os.path.join(updated_mask, filename + 'crack' + '.jpg')    
        
        # load contours from mask file
        spall_contours = []
        rebar_contours = []
        crack_contours = []
        if os.path.exists(rebarmask_file):
            img_binary = cv2.imread(rebarmask_file, cv2.IMREAD_GRAYSCALE)
            ret, rebarthresh = cv2.threshold(img_binary, 127, 255, 0)        
            im2, rebar_contours, rebar_hierarchy = cv2.findContours(rebarthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if os.path.exists(spallmask_file):
            img_binary = cv2.imread(spallmask_file, cv2.IMREAD_GRAYSCALE)
            ret, spallthresh = cv2.threshold(img_binary, 127, 255, 0)
            im2, spall_contours, spall_hierarchy = cv2.findContours(spallthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)          
        if os.path.exists(crackmask_file):
            img_binary = cv2.imread(crackmask_file, cv2.IMREAD_GRAYSCALE)
            ret, crackthresh = cv2.threshold(img_binary, 127, 255, 0)
            im2, crack_contours, crack_hierarchy = cv2.findContours(crackthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # generate contour cache
        regions = {}
        count = 0
        pair = {}                
        if not crack_contours==[]:
            classname = 'crack'
            for j in range(len(crack_contours)):
                shape_groups = []
                one_contour = crack_contours[j][:, 0, :]
                all_x = np.array(one_contour[:, 0]).tolist()
                all_y = np.array(one_contour[:, 1]).tolist()
                if crack_hierarchy[0,j,3]==-1:  
                    shape_groups.append({'type': 'out' ,'all_points_x':all_x, 'all_points_y':all_y})
                    regions[str(count)]={'region_attributes': classname, 'shape_attributes':shape_groups}
                    pair[str(j)] = count
                    count = count + 1
                else:
                    indexvalue = crack_hierarchy[0,j,3]
                    if not crack_hierarchy[0,indexvalue,3] == -1:
                        print('There may be errors in mask ' + filename + 'crack' + '.jpg')
                        error_mask = error_mask + 1
                        continue
                    index = pair[str(indexvalue)]
                    shape_groups = regions[str(index)]['shape_attributes']
                    shape_groups.append({'type': 'in' ,'all_points_x':all_x, 'all_points_y':all_y})
                    regions[str(index)]={'region_attributes': classname, 'shape_attributes':shape_groups}          
        pair = {}
        if not spall_contours==[]:
            classname = 'spall'
            for j in range(len(spall_contours)):
                shape_groups = []
                one_contour = spall_contours[j][:, 0, :]
                all_x = np.array(one_contour[:, 0]).tolist()
                all_y = np.array(one_contour[:, 1]).tolist()
                # check if the contour is inside another and thus [:,:,3]parent is not ==-1
                if spall_hierarchy[0,j,3]==-1:  
                    shape_groups.append({'type': 'out' ,'all_points_x':all_x, 'all_points_y':all_y})
                    regions[str(count)]={'region_attributes': classname, 'shape_attributes':shape_groups}
                    pair[str(j)] = count
                    count = count + 1
                else:
                    indexvalue = spall_hierarchy[0,j,3]
                    if not spall_hierarchy[0,indexvalue,3] == -1: # second inside defect masks, usually should not happen
                        print('There may be errors in mask ' + filename + 'spall' + '.jpg')
                        error_mask = error_mask + 1
                        continue
                    index = pair[str(indexvalue)] # find the count of the parent contour
                    shape_groups = regions[str(index)]['shape_attributes']
                    shape_groups.append({'type': 'in' ,'all_points_x':all_x, 'all_points_y':all_y})
                    regions[str(index)]={'region_attributes': classname, 'shape_attributes':shape_groups}                   
        pair = {}
        if not rebar_contours==[]:
            classname = 'rebar'      
            for j in range(len(rebar_contours)):
                shape_groups = []
                one_contour = rebar_contours[j][:, 0, :]
                all_x = np.array(one_contour[:, 0]).tolist()
                all_y = np.array(one_contour[:, 1]).tolist()
                if rebar_hierarchy[0,j,3]==-1:  
                    shape_groups.append({'type': 'out' ,'all_points_x':all_x, 'all_points_y':all_y})
                    regions[str(count)]={'region_attributes': classname, 'shape_attributes':shape_groups}
                    pair[str(j)] = count
                    count = count + 1
                else:
                    indexvalue = rebar_hierarchy[0,j,3]
                    if not rebar_hierarchy[0,indexvalue,3] == -1:
                        print('There may be errors in mask ' + filename + 'rebar' + '.jpg')
                        error_mask = error_mask + 1
                        continue
                    else:
                        index = pair[str(indexvalue)]
                        shape_groups = regions[str(index)]['shape_attributes']
                        shape_groups.append({'type': 'in' ,'all_points_x':all_x, 'all_points_y':all_y})
                        regions[str(index)]={'region_attributes': classname, 'shape_attributes':shape_groups}

        # merge instances acording to "object_merge"
        if os.path.exists(annoname):
            merge_groups = {}
            name_list = {}             
            for jj in range(len(object_merge)):
                merge_groups[str(jj)]=[]
                name_list[str(jj)]=[]
                
            # assign each instance to merge_groups  
            instance_num = len(regions)
            for k in range(instance_num):
                one_region = regions[str(k)]
                polygons = one_region['shape_attributes']
                classname = one_region['region_attributes']
                
                check = 0
                old_center_dis = 4000
                
                polygon = polygons[0] # only need consider the outmost contour
                all_x1 = polygon['all_points_x']
                all_y1 = polygon['all_points_y']
                rr, cc = skimage.draw.polygon(all_y1, all_x1)
                all_p1 = np.column_stack([np.array(all_x1), np.array(all_y1)])
                contour = np.expand_dims(all_p1, axis = 1)               
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                for ii in range(len(object_merge)):
                    [x1,y1,x2,y2] = object_merge[ii]
                    if cX <= x1 or cX >= x2 or cY <= y1 or cY >= y2:
                        continue 
                    center_disx = (x1+x2)/2-cX
                    center_dixy = (y1+y2)/2-cY
                    new_center_dis = (center_disx**2 + center_dixy**2)**0.5
                    if new_center_dis < old_center_dis:
                        dis_index = ii
                        old_center_dis = new_center_dis
                if (ii+1) == len(object_merge):  
                    [x1,y1,x2,y2] = object_merge[dis_index]
                    if cX >= x1 and cX <= x2 and cY >= y1 and cY <= y2:
                        merge_groups[str(dis_index)].extend(polygons)
                        name_list[str(dis_index)].extend([classname])
                        check = 1
                if not check == 1:
                    print('No merged box belongs to the defect in ' + file)
            
            # update "regions"
            new_regions = {}
            count = 0
            for jj in range(len(object_merge)):
                if merge_groups[str(jj)]==[]:
                    print('No defect belongs to this merged box ' + file)                  
                else:   
                    # determine the class name for this merge box: [crack, spall, rebar] or [crack, spall]
                    namelist = name_list[str(jj)]
                    if 'crack' in namelist:
                        classname = 'crack'
                    elif 'spall' in namelist and 'rebar' not in namelist:
                        classname = 'spall'
                    elif 'rebar' in namelist:    
                        classname = 'rebar'
                    new_regions[str(count)]={'region_attributes': classname, 'shape_attributes':merge_groups[str(jj)]}
                    count = count +1

        damage_bgr = cv2.imread(rgb_file) # read and save in BGR mode 
        height, width, _ = damage_bgr.shape
        if os.path.exists(annoname):
            copy_regions = new_regions
        else:
            copy_regions = regions
        # save in annotations list         
        annotations.append({'filename': file, 'regions': copy_regions, 'size': [height, width]})

        # verify the annotation for each image
        if do_verification:
            damage_rgb = cv2.cvtColor(damage_bgr, cv2.COLOR_BGR2RGB)
            instance_num = len(copy_regions)
            boxes = np.zeros([instance_num, 4], dtype=np.int32)
            boxes_name = []
            instance_mask = []
            
            for k in range(instance_num):
                one_region = copy_regions[str(k)]
                class_name = one_region['region_attributes']
                polygons = one_region['shape_attributes']
                each_mask  = np.zeros([height, width], dtype=np.bool)    
                for each_poly in polygons:
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
    
                instance_mask.append(each_mask)
                # extract the box from mask
                y1, x1, y2, x2 = extract_bboxes(each_mask)
                boxes[k] = np.array([y1, x1, y2, x2]).astype(np.int32)
                boxes_name.append(class_name)
    
            # creat merged new mask for each class of each image
            crack_mask = np.zeros([height, width], dtype=np.uint8)
            spall_mask = np.zeros([height, width], dtype=np.uint8)
            rebar_mask = np.zeros([height, width], dtype=np.uint8)
            for k in range(instance_num):
                defectname = boxes_name[k]
                defectmask = instance_mask[k]
                defectmask = (defectmask * 255).astype(np.uint8)
                
                if defectname == 'crack':
                    crack_mask = np.where(defectmask == 255, 255, crack_mask)
                elif defectname == 'spall': 
                    spall_mask = np.where(defectmask == 255, 255, spall_mask)
                elif defectname == 'rebar':
                    rebar_mask = np.where(defectmask == 255, 255, rebar_mask)
                    
            # plot masks on original image
            if np.max(crack_mask) == 255:
                color = [255,255,0] # yellow
                for c in range(3):
                    damage_rgb[:, :, c] = np.where(crack_mask == 255, damage_rgb[:, :, c] *0.8 + 0.2 * color[c], damage_rgb[:, :, c])
            if np.max(spall_mask) == 255:
                color = [0,255, 255] # Cyan
                for c in range(3):
                    damage_rgb[:, :, c] = np.where(spall_mask == 255, damage_rgb[:, :, c] *0.85 + 0.15 * color[c], damage_rgb[:, :, c])   
            if np.max(rebar_mask) == 255: 
                color = [255,0, 255] # Magenta
                for c in range(3):
                    damage_rgb[:, :, c] = np.where(rebar_mask == 255, damage_rgb[:, :, c] *0.8 + 0.2 * color[c], damage_rgb[:, :, c])
            
            # draw bounding boxes
            img_draw = Image.fromarray(damage_rgb)
            draw = ImageDraw.Draw(img_draw)
            font = ImageFont.truetype(font='fonttype/FiraMono-Medium.otf', size=int(0.02*height))
            for j in range(instance_num):
                y1, x1, y2, x2 = boxes[j,:]
                draw.line([x1,y1,x1,y2], fill=(255,0,0), width=2)
                draw.line([x2,y1,x2,y2], fill=(255,0,0), width=2)
                draw.line([x1,y1,x2,y1], fill=(255,0,0), width=2)
                draw.line([x1,y2,x2,y2], fill=(255,0,0), width=2)
                text_str = str(j) + ' ' + boxes_name[j]
                draw.text(np.array([x1,  y1]), text_str, font=font, fill=(0,0,255))
            del draw
    
            imagedir=os.path.join(verify_path, filename + '.jpg')
            img_draw.save(imagedir)        
        
    print('Number of error mask is ' + str(error_mask))
     
    print('Saving gt_labels to: ' + ground_truth_cache)
    with open(ground_truth_cache, 'wb') as f:
        cPickle.dump(annotations, f)        
        
    return annotations

def extract_bboxes(each_mask):
    # Compute bounding boxes from masks[height, width] either 1 or 0
    m = each_mask
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    # x2 and y2 should not be part of the box. Increment by 1.
    x2 += 1
    y2 += 1
    return y1, x1, y2, x2

if __name__ == '__main__':

    '''
    This code generate the annotation cache for traing and testing DIS-YOLO. 
    You can also prepare your own code for generating ground-truth cache. 
    The ground-truth format for this study is as follows:
    
    annotations = 
    list[
        {
          'filename': 'xxx.jpg',
          'regions': {
                      '0': {
                            'region_attributes': class name, 
                            'shape_attributes' : polygon list[
                                                             {
                                                              'type': 'in/out',
                                                              'all_points_x': list [int...],
                                                              'all_points_y': list [int...]
                                                              }, ... more polygons based on 'type' and .xml file
                                                             ]
                            },
                      ..."1", "2",... more regions
                      },
           'size': [height, width]
         },
         ...more annotations
         ]
    
    Note: each region represent one individual defect instacne;
          "class name" is the assigned defect type for each region;
          polygon with "type'="in' indicates the inside empty mask, hence real mask = out mask - in mask
    '''
    
    model_path = os.getcwd()
    dataset = os.path.join(model_path, 'data')
    phase = 'train' # train, val or test 
    annotations = load_verify_contour(dataset, phase)

    
    