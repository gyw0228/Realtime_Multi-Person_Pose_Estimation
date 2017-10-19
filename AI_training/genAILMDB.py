import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import sys, os

# change your caffe path here
#sys.path.insert(0, os.path.join('/home/zhecao/caffe/', 'python/'))
#sys.path.insert(0, '/home/haoquan/HardDisk/yihongwei/caffe_train/distribute/python')
sys.path.insert(0, '/home/haoquan/HardDisk/yihongwei/caffe_train/python')
import caffe
import os.path
import struct


def writeLMDB(datasets, lmdb_path, validation): #validation 0 for training, 1 for has validation data
    env = lmdb.open(lmdb_path, map_size=int(1e12))  # env for open lmdb handle
    txn = env.begin(write=True)  # txn for text flow of env
    data = []
    numSample = 0

    for d in range(len(datasets)):
        if (datasets[d] == "MPI"):
            print datasets[d]
            with open('MPI.json') as data_file:
                data_this = json.load(data_file)
                data_this = data_this['root']
                data = data + data_this
            numSample = len(data)
            # print data
            print numSample
        elif (datasets[d] == "COCO"):
            print datasets[d]
            with open('dataset/COCO/json/COCO.json') as data_file:
                data_this = json.load(data_file)
                data_this = data_this['root']
                data = data + data_this
            numSample = len(data)
            # print data
            print numSample
        elif (datasets[d] == "AIChallenger"):
            print datasets[d]
            with open('/mnt/sda1/yihongwei/dataset/AIChallenger/' \
                      'ai_challenger_keypoint_train_20170902/new_keypoint_train_annotations_20170909.json') as data_file:
#            with open('/mnt/sda1/yihongwei/dataset/AIChallenger/' \
#                                        'ai_challenger_keypoint_validation_20170911/' \
#                      'new_keypoint_validation_annotations_20170911.json') as data_file:
                data_this = json.load(data_file)
                print len(data_this)
                data = data + data_this
            numSample = len(data)
           # numSample = 20 #for test

    # data - load keypoint json file
    random_order = np.random.permutation(numSample).tolist()
    isValidationArray = []
    #isValidationArray = [data[i]['isValidation'] for i in range(numSample)];
    if (validation == 1):
        totalWriteCount = isValidationArray.count(0.0);
    else:
        totalWriteCount = len(data)
    print totalWriteCount;
    writeCount = 0

    for count in range(numSample):
        idx = random_order[count]
        data[idx]['isValidation'] = 0
        if (data[idx]['isValidation'] != 0 and validation == 1):  # validation used for test not in training
            print '%d/%d skipped' % (count, idx)
            continue

        data[idx]['dataset'] = "AIChallenger"
        if "MPI" in data[idx]['dataset']:
            path_header = 'dataset/MPI/images/'
        elif "COCO" in data[idx]['dataset']:
            path_header = '/media/posenas4b/User/zhe/Convolutional-Pose-Machines/training/dataset/COCO/images/'
        elif "AIChallenger" in datasets:
            path_header = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
                          'ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
            mask_path_header = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
                          'ai_challenger_keypoint_train_20170902/keypoint_train_images_mask_20170902/'
#            path_header = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
#                          'ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'
#            mask_path_header = '/mnt/sda1/yihongwei/dataset/AIChallenger/' \
#                               'ai_challenger_keypoint_validation_20170911/keypoint_validation_mask_20170911/'

        #print os.path.join(path_header, data[idx]['img_paths'])
        img_path = path_header + data[idx]["image_id"] + ".jpg"
        print 'idx :', idx, " path:", img_path
        #img = cv2.imread(os.path.join(path_header, data[idx]['img_paths']))
        img = cv2.imread(img_path)
        # print data[idx]['img_paths']
        #img_idx = data[idx]['img_paths'][-16:-3];
        img_idx = data[idx]["image_id"]
        # print img_idx
        if "AIChallenger" in datasets:
            mask_miss = cv2.imread(mask_path_header + data[idx]["image_id"] + "_train_mask_miss_with_bbox.jpg", 0)
            #mask_miss = cv2.imread(mask_path_header + data[idx]["image_id"] + "_validation_mask_miss.jpg", 0)
        elif "COCO_val" in data[idx]['dataset']:
            mask_all = cv2.imread(path_header + 'mask2014/val2014_mask_all_' + img_idx + 'png', 0)
            mask_miss = cv2.imread(path_header + 'mask2014/val2014_mask_miss_' + img_idx + 'png', 0)
        # print path_header+'mask2014/val2014_mask_miss_'+img_idx+'png'
        elif "COCO" in data[idx]['dataset']:
            mask_all = cv2.imread(path_header + 'mask2014/train2014_mask_all_' + img_idx + 'png', 0)
            mask_miss = cv2.imread(path_header + 'mask2014/train2014_mask_miss_' + img_idx + 'png', 0)
        # print path_header+'mask2014/train2014_mask_miss_'+img_idx+'png'
        elif "MPI" in data[idx]['dataset']:
            img_idx = data[idx]['img_paths'][-13:-3];
            # print img_idx
            mask_miss = cv2.imread('dataset/MPI/masks/mask_' + img_idx + 'jpg', 0)
        # mask_all = mask_miss

        height = img.shape[0]
        width = img.shape[1]
        if (width < 64):
            img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            cv2.imwrite('padded_img.jpg', img)
            width = 64
        # no modify on width, because we want to keep information
        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        # print type(img), img.shape
        # print type(meta_data), meta_data.shape
        clidx = 0  # current line index
        # dataset name (string)
        data[idx]['dataset'] = "AIChallenger"
        for i in range(len(data[idx]['dataset'])):
            meta_data[clidx][i] = ord(data[idx]['dataset'][i])
        clidx = clidx + 1
        # image height, image width
        #height_binary = float2bytes(data[idx]['img_height'])
        height_binary = float2bytes(float(data[idx]['height']))
        for i in range(len(height_binary)):
            meta_data[clidx][i] = ord(height_binary[i])
        #width_binary = float2bytes(data[idx]['img_width'])
        width_binary = float2bytes(float(data[idx]['width']))
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = ord(width_binary[i])
        clidx = clidx + 1
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
        data[idx]['isValidation'] = 0
        meta_data[clidx][0] = data[idx]['isValidation']
        meta_data[clidx][1] = data[idx]['numOtherPeople']
        meta_data[clidx][2] = data[idx]['people_index']
        annolist_index_binary = float2bytes(float(data[idx]['annolist_index'])) #based on 0
        for i in range(len(annolist_index_binary)):  # 3,4,5,6
            meta_data[clidx][3 + i] = ord(annolist_index_binary[i]) #based on 0
        count_binary = float2bytes(float(writeCount))  # note it's writecount instead of count!
        for i in range(len(count_binary)):
            meta_data[clidx][7 + i] = ord(count_binary[i])
        totalWriteCount_binary = float2bytes(float(totalWriteCount))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11 + i] = ord(totalWriteCount_binary[i])
        nop = int(data[idx]['numOtherPeople'])
        clidx = clidx + 1
        # (b) objpos_x (float), objpos_y (float)
        #data[idx]['obj_pos'] = [float(temp) for temp in data[idx]['obj_pose']]
        objpos_binary = float2bytes(data[idx]['obj_pos'])
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = ord(objpos_binary[i])
        clidx = clidx + 1
        # (c) scale_provided (float)
        #scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        scale_provided_binary = float2bytes(data[idx]['scale_provide'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = ord(scale_provided_binary[i])
        clidx = clidx + 1
        # (d) joint_self (3*16) (float) (3 line)
        joints = np.asarray(data[idx]['joint_itself']).T.tolist()  # transpose to 3*16
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = ord(row_binary[j])
            clidx = clidx + 1
        # (e) check nop, prepare arrays
        if (nop != 0):
 #           if (nop == 1):  # only one others
 #               joint_other = [data[idx]['joint_others']]
 #               objpos_other = [data[idx]['objpos_other']]
 #               scale_provided_other = [data[idx]['scale_provided_other']]
 #          else:
            joint_other = data[idx]['other_person']['joint_others'] #for dict
            objpos_other = data[idx]['other_person']['objpos_other']
            scale_provided_other = [data[idx]['other_person']['scale_provided_other'][key] for key in data[idx]['other_person']['scale_provided_other']]
            # (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
            #for i in range(nop):
            for i in joint_other: #for key travesal
                objpos_binary = float2bytes(objpos_other[i])
                for j in range(len(objpos_binary)):
                    meta_data[clidx][j] = ord(objpos_binary[j])
                clidx = clidx + 1
            # (g) scale_provided_other (nop floats in 1 line)

            scale_provided_other_binary = float2bytes(scale_provided_other)
            for j in range(len(scale_provided_other_binary)):
            #for j in rangelen(joint_other):
                meta_data[clidx][j] = ord(scale_provided_other_binary[j])
            clidx = clidx + 1
            # (h) joint_others (3*16) (float) (nop*3 lines)
            #for n in range(nop):
            for n in joint_other:
                joints = np.asarray(joint_other[n]).T.tolist()  # transpose to 3*16
                for i in range(len(joints)):
                    row_binary = float2bytes(joints[i])
                    for j in range(len(row_binary)):
                        meta_data[clidx][j] = ord(row_binary[j])
                    clidx = clidx + 1

        # print meta_data[0:12,0:48]
        # total 7+4*nop lines
        data[idx]['dataset'] = ['AIChallenger']
        if "AIChallenger" in data[idx]['dataset']:
            pass
        #    print meta_data
        #   print mask_miss[..., None]

            img4ch = np.concatenate((img, meta_data, mask_miss[..., None]), axis=2)
        elif "COCO" in data[idx]['dataset']:  # 3-dimension concatenate
            img4ch = np.concatenate((img, meta_data, mask_miss[..., None], mask_all[..., None]), axis=2)
        # img4ch = np.concatenate((img, meta_data, mask_miss[...,None]), axis=2)
        elif "MPI" in data[idx]['dataset']:
            img4ch = np.concatenate((img, meta_data, mask_miss[..., None]), axis=2)

        img4ch = np.transpose(img4ch, (2, 0, 1))  #
        #print img4ch
        #print img4ch.shape

        datum = caffe.io.array_to_datum(img4ch, label=0)  # img4ch and label 0 into lmdb
        key = '%07d' % writeCount  # key is followed writeCount;
        txn.put(key, datum.SerializeToString())
        if (writeCount % 1000 == 0):
            txn.commit()
            txn = env.begin(write=True)
        print '%d/%d/%d/%d' % (count, writeCount, idx, numSample)
        writeCount = writeCount + 1

    txn.commit()
    env.close()


def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    return struct.pack('%sf' % len(floats), *floats)


if __name__ == "__main__":
    # writeLMDB(['MPI'], '/home/zhecao/MPI_pose/lmdb', 1)
    #writeLMDB(['COCO'], '/home/zhecao/COCO_kpt/lmdb', 1)
    #writeLMDB(['AIChallenger'], '/mnt/sda1/yihongwei/dataset/AIChallenger'\
    #                            '/ai_challenger_keypoint_train_20170902/AIChallenger_lmdb_20170902', 0)
    #writeLMDB(['AIChallenger'], '/mnt/sda1/yihongwei/dataset/AIChallenger' \
    #                           '/ai_challenger_keypoint_validation_20170911/AIChallenger_val_lmdb_20170911', 0)
    #writeLMDB(['AIChallenger'], '/mnt/sda1/yihongwei/dataset/AIChallenger' \
    #                                '/ai_challenger_keypoint_train_20170902/test_lmdb', 0)
    writeLMDB(['AIChallenger'], '/mnt/sda1/yihongwei/dataset/AIChallenger' \
                                        '/ai_challenger_keypoint_train_20170902/AIChallenger_lmdb_withbox_20170902', 0)
