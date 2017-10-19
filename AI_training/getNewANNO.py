import  json
import os
import sys

annofile_path = ''
dataset_dir = '/mnt/sda1/yihongwei/dataset/AIChallenger/'
#os.system('mkdir ' + dataset_dir + '/mat')

mode = 1 #1 for validation
if mode == 1:
    annFile = dataset_dir + 'ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909json'
else:
    annFile = ''

aichallenger = json.load(open(annFile, 'r'))
print len(aichallenger)
