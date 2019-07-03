# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:52:35 2017

@author: chlian

Apply pruned wH-FCN after training...
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio

from keras.models import Model
from pruned_hfcn import HierachicalNet, acc
from pruned_sequential_loader import tst_data_flow
from loss import binary_cross_entropy, mse

NUM_CHNS = 1
FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
NUM_MIDIATE_FEATURES = 64 
NUM_GLOBAL_FEATURES = 64 

WITH_BN = True
WITH_DROPOUT = True
DROP_PROB = 0.5

SHARING_WEIGHTS = False

PATCH_SIZE = 25

BATCH_SIZE = 1

IMAGE_PATH = "/shenlab/lab_stor4/cflian/mnorm_AD2_ADNC/"

CENTER_MAT_NAME = 'top_lmks_d{0}_ADNI2'.format(PATCH_SIZE)
CENTER_MAT = sio.loadmat('files/'+CENTER_MAT_NAME+'.mat')
CENTER_MAT = CENTER_MAT['landmarks']
CENTER_MAT = np.round(CENTER_MAT).astype(int) - 1

PRUNED_NEIGHBORS = sio.loadmat('files/pruned_nns_idx_ADNI1.mat')
PRUNED_NEIGHBORS = PRUNED_NEIGHBORS['pruned_nns_idx']

PRUNED_PATCHES = sio.loadmat('files/pruned_pths_ADNI1.mat')
PRUNED_PATCHES = PRUNED_PATCHES['pruned_patches']
PRUNED_PATCHES = PRUNED_PATCHES.flatten().tolist()

NUM_PATCHES = len(PRUNED_PATCHES)
NUM_REGION_OUTPUTS = np.shape(PRUNED_NEIGHBORS)[0]
NUM_SUBJECT_OUTPUTS = 1

TST_SUBJ_LIST = range(1, 361)
LABEL_NAME = 'labels_ADNI2'
TST_SUBJ_LBLS = sio.loadmat('files/'+LABEL_NAME+'.mat')
TST_SUBJ_LBLS = TST_SUBJ_LBLS[LABEL_NAME]
TST_SUBJ_LBLS[np.where(TST_SUBJ_LBLS==-1)[0]] = 0

MODEL_PATH = "saved_model/"
MODEL_NAME = 'pruned_whfcn_step3.best.hd5'


if __name__ == '__main__':
    
    params = {'num_chns': NUM_CHNS, 
              'num_midiate_outputs': NUM_REGION_OUTPUTS, 
              'num_global_outputs': NUM_SUBJECT_OUTPUTS,
              'feature_depth': FEATURE_DEPTH, 
              'midiate_features': NUM_MIDIATE_FEATURES,
              'global_features': NUM_GLOBAL_FEATURES,
              'patch_size': PATCH_SIZE, 
              'num_patches': NUM_PATCHES,
              'batch_size': BATCH_SIZE, 
              'with_bn': WITH_BN,
              'with_dropout': WITH_DROPOUT, 
              'drop_prob': DROP_PROB, 
              'sharing_weights': SHARING_WEIGHTS,
              'img_path': IMAGE_PATH, 
              'model_path': MODEL_PATH, 
              'model_name': MODEL_NAME,
              'center_mat': CENTER_MAT,
              'pruned_pths': PRUNED_PATCHES,
              'pruned_nns': PRUNED_NEIGHBORS,
              'tst_list': TST_SUBJ_LIST,
              'tst_lbls': TST_SUBJ_LBLS}
              
    
    Net = HierachicalNet(patch_size=params['patch_size'],
                         num_patches=params['num_patches'],
                         pruned_nns=params['pruned_nns'],
                         num_chns=params['num_chns'],
                         num_outputs=params['num_global_outputs'],
                         feature_depth=params['feature_depth'],
                         num_region_features=params['midiate_features'],
                         num_subject_features=params['global_features'],
                         with_bn=params['with_bn'],
                         with_dropout=params['with_dropout'],
                         drop_prob=params['drop_prob'],
                         region_sparse=0.0,
                         subject_sparse=0.0).get_global_net()
    Net.compile(loss=binary_cross_entropy, loss_weights=[1.0, 1.0, 1.0],
                optimizer='Adam', metrics=[acc])                     
    Net.load_weights(params['model_path']+params['model_name'])
    
    patch_outputs = np.zeros((len(params['tst_list']), params['num_patches']))
    patch_scores = np.zeros((len(params['tst_list']), params['num_patches']))
    region_outputs = np.zeros((len(params['tst_list']), params['num_midiate_outputs']))
    region_scores = np.zeros((len(params['tst_list']), params['num_midiate_outputs']))
    subject_outputs = np.zeros((len(params['tst_list']), params['num_global_outputs']))
    subject_scores = np.zeros((len(params['tst_list']), params['num_global_outputs']))                     
    for i in range(len(params['tst_list'])):
        inputs, outputs = tst_data_flow(img_path=params['img_path'], 
                                        sample_idx=params['tst_list'][i], 
                                        sample_lbl=params['tst_lbls'][i], 
                                        center_cors=params['center_mat'],
                                        patch_idxs=params['pruned_pths'],
                                        patch_size=params['patch_size'], 
                                        num_chns=params['num_chns'], 
                                        num_patches=params['num_patches'],
                                        num_region_outputs=params['num_midiate_outputs'], 
                                        num_subject_outputs=params['num_global_outputs'])                            
        predicts = Net.predict(inputs)

        patch_outputs[i, :] = predicts[0]
        region_outputs[i, :] = predicts[1]
        subject_outputs[i, :] = predicts[2]
        patch_scores[i, :] = 1.0 - np.abs(outputs[0] - predicts[0])
        region_scores[i, :] = 1.0 - np.abs(outputs[1] - predicts[1])
        subject_scores[i, :] = 1.0 - np.abs(outputs[2] - predicts[2])
        print i

    sio.savemat(params['model_path']+'ADNI2_wHFCN_predictionscc.mat',
                {'patch_outputs': patch_outputs, 'patch_scores': patch_scores, 
                'region_outputs': region_outputs, 'region_scores': region_scores, 
                'subject_outputs': subject_outputs, 'subject_scores': subject_scores})


    
