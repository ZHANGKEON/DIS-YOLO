# --------------------------------------------------------
# Modified from the GitHub code "py-faster-rcnn"
# reference [1] https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# reference [2] https://github.com/matterport/Mask_RCNN
# --------------------------------------------------------

import numpy as np

def voc_ap(rec, prec, use_07_metric):

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_overlaps_masks(masks1, masks2):
    # Computes IoU overlaps between two sets of masks[Height, Width, instances]
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1  = np.sum(masks1, axis=0)
    area2  = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union    = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def voc_eval(detfile,
             recs,
             imagesetfile,
             classid,
             ovthresh=0.5,
             use_07_metric = False):
    
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # extract gt objects for classid
    class_recs = {}
    npos       = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['classid'] == classid]
        if len(R) > 0:
            bbox = np.concatenate([np.expand_dims(x['mask'], -1) for x in R], -1)
        else:
            bbox = np.array([])        
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det  = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'mask': bbox,
                                 'difficult': difficult,
                                 'det': det}
        
    # read detfile
    image_ids = [x['imageid'] for x in detfile]
    confidence = np.array([float(x['score']) for x in detfile])
    # masks are not in the same shape, hence save in list
    BB = [np.expand_dims(x['mask'], -1) for x in detfile]

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    if len(sorted_ind) == 0:
        rec, prec, ap = 0., 0., 0.
        return rec, prec, ap
#    sorted_scores = np.sort(-confidence)
    BB = [BB[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down detections and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R     = class_recs[image_ids[d]]
        bb    = BB[d].astype(float)
        ovmax = -np.inf
        BBGT  = R['mask'].astype(float) 

        if BBGT.size > 0:
            # compute overlaps
            overlaps = compute_overlaps_masks(bb, BBGT)
            ovmax    = np.max(overlaps[0])
            jmax     = np.argmax(overlaps[0])

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1 
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp   = np.cumsum(fp)
    tp   = np.cumsum(tp)
    rec  = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap   = voc_ap(rec, prec, use_07_metric)
    
    recall    = tp[-1] / float(npos)
    precision = tp[-1] / np.maximum(tp[-1] + fp[-1], np.finfo(np.float64).eps)
    return recall, precision, ap
