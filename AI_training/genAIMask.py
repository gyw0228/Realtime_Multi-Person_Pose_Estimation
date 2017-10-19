import json
import sys
import os
from PIL import Image
import numpy as np
mode = 1 #1 for training, 0 for validation
mask_kind = 1 #1 for bbox mask

if mode == 0:
    annoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
               'ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    absolute_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger' \
                   '/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
    mask_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
               'ai_challenger_keypoint_validation_20170911/keypoint_validation_mask_20170911'
elif mode == 1:
    annoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
               '/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json'
    absolute_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger' \
                   '/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902'
    mask_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
               'ai_challenger_keypoint_train_20170902/keypoint_train_images_mask_20170902'

aichallenger_kpt = json.load(open(annoFile))
print len(aichallenger_kpt)

for i in range(0, len(aichallenger_kpt)):
#for i in range(0, 2):
    img_path = absolute_dir + '/' + aichallenger_kpt[i]['image_id'] + '.jpg'
    #print img_path
    if mode == 1 and mask_kind != 1:
        img_name1 = mask_dir + '/' + aichallenger_kpt[i]['image_id'] + '_train_mask_miss.jpg'
    elif mode == 1 and mask_kind == 1:
        img_name1 = mask_dir + '/' + aichallenger_kpt[i]['image_id'] + '_train_mask_miss_with_bbox.jpg' #only bbox area has 1
    else:
        img_name1 = mask_dir + '/' + aichallenger_kpt[i]['image_id'] + '_validation_mask_miss.jpg'
    print img_name1
    if os.path.exists(img_name1):
        continue
    if os.path.exists(img_path):
        img = Image.open(img_path)
        [width, height] = img.size
        if mask_kind == 0:
            mask_miss = np.ones((height, width), dtype=np.uint8)

        elif mask_kind == 1:
            mask_miss = np.zeros((height, width), dtype=np.uint8)
            bbox_dict = aichallenger_kpt[i]["human_annotations"]
            for key in bbox_dict:
                bbox = bbox_dict[key]
                print "bbox:", bbox
                #slice copy
                x1 = max(bbox[0], 0)
                y1 = max(bbox[1], 0)
                x2 = min(bbox[2], width-1)
                y2 = min(bbox[3], height-1)
#                bbox_temp = np.ones((y2-y1, x2-x1), dtype=np.uint8)
#                temp = mask_miss[x1:x2+1, y1:y2+1] or bbox_temp
#                print temp
#                mask_miss[x1:x2+1, y1:y2+1] = temp.astype(np.uint8)
                for index2 in range(x1, x2+1):
                    for index1 in range(y1, y1+1):
                        mask_miss[index1][index2] = (mask_miss[index1][index2] or 1)
                        mask_miss[index1][index2].astype(np.uint8)

            print np.count_nonzero(mask_miss)
 #           print mask_miss

        img_mask_miss = Image.fromarray(mask_miss)
        print 'image cnt ', i, ':', img_mask_miss

        img_mask_miss.save(img_name1)
