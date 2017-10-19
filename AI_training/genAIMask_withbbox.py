import json
import sys
import os
from PIL import Image
import numpy as np

annoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
           'ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
mode = 0 #1 for training, 0 for validation

absolute_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger' \
               '/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'

mask_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
           'ai_challenger_keypoint_validation_20170911/keypoint_validation_mask_20170911'

aichallenger_kpt = json.load(open(annoFile))
print len(aichallenger_kpt)

for i in range(0, len(aichallenger_kpt)):
#for i in range(0, 2):
    img_path = absolute_dir + '/' + aichallenger_kpt[i]['image_id'] + '.jpg'
    #print img_path

    if mode == 1:
        img_name1 = mask_dir + '/' + aichallenger_kpt[i]['image_id'] + '_train_mask_miss.jpg'
    else:
        img_name1 = mask_dir + '/' + aichallenger_kpt[i]['image_id'] + '_validation_mask_miss.jpg'
    #print img_name1
    if os.path.exists(img_name1):
        continue
    if os.path.exists(img_path):
        img = Image.open(img_path)
        [width, height] = img.size
        mask_miss = np.ones((height, width), dtype=np.uint8)
        img_mask_miss = Image.fromarray(mask_miss)
        print 'image cnt ', i, ':', img_mask_miss
        img_mask_miss.save(img_name1)
