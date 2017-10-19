import json
import sys
import os
from PIL import Image
import numpy as np
from AI_util import *

# get construct [one_annotation, two_annotation, ....]

#load anno json
#annoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
#           'ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json'
annoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
           'ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'

mode = 0 #1 for training, 0 for validation

#newAnnoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
#           'ai_challenger_keypoint_train_20170902/new_keypoint_train_annotations_20170909.json'
newAnnoFile = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
           'ai_challenger_keypoint_validation_20170911/new_keypoint_validation_annotations_20170911.json'

absolute_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger' \
               '/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'

count = 1
makefigure = 0
validationCount = 0
isValidation = 0

if mode == 1:
    anno_list = json.load(open(annoFile))
else:
    anno_val = json.load(open(annoFile))


if mode == 1:
    RELEASE = anno_list
else:
    RELEASE = anno_val
    pass

#store each annotation of each person
joint_all = []

# for each image
for i in range(0, len(RELEASE)):
#for i in range(0, 10): #i for index of image, p for num_index of people
    numPeople = len(RELEASE[i]['human_annotations'])
    print "idx ", i, "has ", numPeople, "people---------------------------------"
    prev_center = []

    img_path = absolute_dir + '/' + RELEASE[i]['image_id'] + '.jpg'
    print "img_id: ", RELEASE[i]['image_id']
    if not os.path.exists(img_path):
        continue
    else:
        image = Image.open(img_path)
        [w, h] = image.size
        for p in range(1, numPeople+1):
            print 'store peopel index ', p, '======================================'
            annoHumanName = "human" + str(p)
            kpt_list = RELEASE[i]["keypoint_annotations"][annoHumanName]
            print "human index:", annoHumanName, kpt_list
            useful_kpts = get_useful_kpt_num(kpt_list, 2)
            visible_kpts = get_useful_kpt_num(kpt_list, 1)
            bbox = RELEASE[i]["human_annotations"][annoHumanName]
            print annoHumanName, bbox
            bbox_area = get_area(bbox)
            print annoHumanName, 'bbox_area, useful_kpts, visile_kpts', bbox_area, useful_kpts, visible_kpts
            ## skip the person if parts number is too low or if bbox area is too small
            if useful_kpts < 5 or bbox_area < 32 * 32:
                continue

            ## skip the person if the distance to exixting person is too small
            person_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
            flag = 0
            for pi in range(0, len(prev_center)):
                dx = prev_center[pi][0] - person_center[0]
                dy = prev_center[pi][1] - person_center[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < prev_center[pi][2] * 0.3: # prev_center[cx, cy,  max(width, height)]
                    flag = 1
                    continue
            if flag == 1:
                continue

            one_person_anno = {}
            one_person_anno['image_path'] = ''
            one_person_anno['image_id'] = RELEASE[i]['image_id']
            one_person_anno['width'] = w
            one_person_anno['height'] = h
            one_person_anno['obj_pos'] = person_center
            one_person_anno['bbox'] = bbox
            one_person_anno['bbox_area'] = bbox_area
            one_person_anno['useful_kpts_num'] = useful_kpts
            one_person_anno['visible_kpts_num'] = visible_kpts

            ### set part label: joint_all is (np-3-n Train) all keypoints set (center person)
            joint_self = np.zeros((14, 3), dtype=int)
            for part in range(0, 14):
                joint_self[part][0] = kpt_list[part*3 + 0]
                joint_self[part][1] = kpt_list[part*3 + 1]

                if(kpt_list[part * 3 + 2] == 3): #unvisible but labeled
                    joint_self[part][2] =  3
                elif (kpt_list[part * 3 + 2] == 1): #visible
                    joint_self[part][2] = 1
                else: #unlabeled
                    joint_self[part][2] = 2 #unpredicted or not in this place

            one_person_anno['joint_itself'] = joint_self.tolist()

            ### set scale
            one_person_anno['scale_provide'] = bbox[3] * 1.0 / 368

            ### for other person on the same page do the same thing
            count_other = 1
            other_person_anno = {}
            other_person_anno['scale_provided_other'] = {}
            other_person_anno['objpos_other'] = {}
            other_person_anno['bbox_other'] = {}
            other_person_anno['area_other'] = {}
            other_person_anno['usefull_kpts_num_other'] = {}
            other_person_anno['visible_kpts_num_other'] = {}
            other_person_anno['joint_others'] = {}

            #one_person_anno['other_person'] = other_person_anno
            for op in range(1, numPeople+1):
                if op == p:
                    continue
                ot_annoHumanName = "human" + str(op)
                ot_kpt_list = RELEASE[i]["keypoint_annotations"][ot_annoHumanName]
                print i,'other people', ot_annoHumanName, ot_kpt_list
                ot_useful_kpts = get_useful_kpt_num(ot_kpt_list, 2)
                ot_visible_kpts = get_useful_kpt_num(ot_kpt_list, 1)
                if ot_useful_kpts == 0:
                    continue;

                ot_bbox = RELEASE[i]["human_annotations"][ot_annoHumanName]
                print ot_annoHumanName, ot_bbox
                ot_bbox_area = get_area(ot_bbox)
                ot_person_center = [(ot_bbox[0] + ot_bbox[2] )/ 2, (ot_bbox[1] + ot_bbox[3]) / 2]

                other_person_anno['scale_provided_other'][count_other] = ot_bbox[3] * 1.0 / 368
                other_person_anno['objpos_other'][count_other] = ot_person_center
                other_person_anno['bbox_other'][count_other] = ot_bbox
                other_person_anno['area_other'][count_other] = ot_bbox_area
                other_person_anno['usefull_kpts_num_other'][count_other] = ot_useful_kpts
                other_person_anno['visible_kpts_num_other'][count_other] = ot_visible_kpts

                joint_other_temp = np.zeros((14,3), dtype=int)
                for part in range(0, 14):
                    joint_other_temp[part][0] = ot_kpt_list[part * 3 + 0]
                    joint_other_temp[part][1] = ot_kpt_list[part * 3 + 1]

                    if (ot_kpt_list[part * 3 + 2] == 3):  # unvisible but labeled
                        joint_other_temp[part][2] = 3
                    elif (ot_kpt_list[part * 3 + 2] == 1):  # visible
                        joint_other_temp[part][2] = 1
                    else:  # unlabeled
                        joint_other_temp[part][2] = 2
                other_person_anno['joint_others'][count_other] = joint_other_temp.tolist()


                count_other += 1

            prev_center.append([one_person_anno['obj_pos'][0], one_person_anno['obj_pos'][1]
                                   , max(bbox[3] - bbox[1], bbox[2] - bbox[0])])
            #print prev_center
            if mode == 0:
                one_person_anno['isValidation'] = 1
            one_person_anno['other_person'] = other_person_anno
            one_person_anno['numOtherPeople'] = len(one_person_anno['other_person']['joint_others'])
            one_person_anno['annolist_index'] = i
            one_person_anno['people_index'] = p
            #print 'people index', p
            #print 'idx ', i, one_person_anno

            #add to all joint list
            joint_all.append(one_person_anno)
            print 'img_index :', i, 'joint_all_length:', len(joint_all)
            print "all person count", count, "|||||||||||||||||||||||||||||||||||||||"
            count += 1

#print joint_all
with open(newAnnoFile,"w") as f:
    json.dump(joint_all, f)



