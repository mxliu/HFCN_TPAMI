# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:31:54 2017

@author: chlian

Train nH-FCN (i.e., without anatomical-landmark-based prior) from scratch...
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio

from keras.models import Model
from keras.optimizers import Adam

from nhfcn_network import HierachicalNet, acc
from noprior_loader import data_flow
from loss import binary_cross_entropy, mse

from sklearn.cross_validation import train_test_split


NUM_CHNS = 1
FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
NUM_REGION_FEATURES = 64
NUM_SUBJECT_FEATURES = 64

WITH_BN = True
WITH_DROPOUT = True
DROP_PROB = 0.5

SHARING_WEIGHTS = False

PATCH_SIZE = 25

TRN_BATCH_SIZE = 5
TST_BATCH_SIZE = 5

NUM_EPOCHS = 100

IMAGE_PATH = "/shenlab/lab_stor4/cflian/mnorm_AD1_ADNC/"

# proposal locations in template space
TEMPLATE_MAT_NAME = 'patch{0}_centers'.format(PATCH_SIZE)
TEMPLATE_MAT = sio.loadmat('files/'+TEMPLATE_MAT_NAME+'.mat')
TEMPLATE_MAT = TEMPLATE_MAT['patch_centers']

# proposal locations in subject space
CENTER_MAT_NAME = 'ADNI1_patch{0}_centers'.format(PATCH_SIZE)
CENTER_MAT = sio.loadmat('files/'+CENTER_MAT_NAME+'.mat')
CENTER_MAT = CENTER_MAT['patch_centers']
CENTER_MAT = np.round(CENTER_MAT).astype(int) - 1

NUM_PATCHES_PER_AXIS = []
for i_cor in range(3):
    NUM_PATCHES_PER_AXIS.append(len(np.unique(TEMPLATE_MAT[i_cor,:])))

NUM_PATCHES = np.shape(CENTER_MAT)[1]
NUM_REGION_OUTPUTS = np.prod(np.array(NUM_PATCHES_PER_AXIS)-1)
NUM_SUBJECT_OUTPUTS = 1


SUBJECT_IDXS = range(1, 429)
LABEL_NAME = 'labels_ADNI1'
LABELS = sio.loadmat('files/'+LABEL_NAME+'.mat')
LABELS = LABELS[LABEL_NAME]
LABELS[np.where(LABELS==-1)[0]] = 0

NEG_SUBJ_IDXS = [SUBJECT_IDXS[i] for i in np.where(LABELS==0)[0]]
POS_SUBJ_IDXS = [SUBJECT_IDXS[i] for i in np.where(LABELS==1)[0]]
NEG_LABELS = [LABELS[i][0] for i in np.where(LABELS==0)[0]]
POS_LABELS = [LABELS[i][0] for i in np.where(LABELS==1)[0]]

TRN_NEG_IDXS, VAL_NEG_IDXS, TRN_NEG_LBLS, VAL_NEG_LBLS = \
    train_test_split(NEG_SUBJ_IDXS, NEG_LABELS, test_size=0.1)
TRN_POS_IDXS, VAL_POS_IDXS, TRN_POS_LBLS, VAL_POS_LBLS = \
    train_test_split(POS_SUBJ_IDXS, POS_LABELS, test_size=0.1)

TRN_SUBJ_LIST = TRN_NEG_IDXS + TRN_POS_IDXS
TRN_SUBJ_LBLS = TRN_NEG_LBLS + TRN_POS_LBLS
TST_SUBJ_LIST = VAL_NEG_IDXS + VAL_POS_IDXS
TST_SUBJ_LBLS = VAL_NEG_LBLS + VAL_POS_LBLS

sio.savemat('files/'+'data_partition.mat', 
            {'trn_list': TRN_SUBJ_LIST, 
            'trn_lbls': TRN_SUBJ_LBLS,
            'val_list': TST_SUBJ_LIST,
            'val_lbls': TST_SUBJ_LBLS})
'''
DATA_PARTITION = sio.loadmat('files/'+'data_partition.mat')
TRN_SUBJ_LIST = DATA_PARTITION['trn_list'][0].tolist()
TRN_SUBJ_LBLS = DATA_PARTITION['trn_lbls'][0].tolist()
TST_SUBJ_LIST = DATA_PARTITION['val_list'][0].tolist()
TST_SUBJ_LBLS = DATA_PARTITION['val_lbls'][0].tolist()
'''

TRN_STEPS = int(np.round(len(TRN_SUBJ_LIST) / TRN_BATCH_SIZE))
TST_STEPS = int(np.round(len(TST_SUBJ_LIST) / TST_BATCH_SIZE))

MODEL_PATH = "saved_model/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
MODEL_NAME = 'nhfcn_patch{0}.best.hd5'.format(PATCH_SIZE)

HISTORY_NAME = 'nhfcn_patch{0}_history'.format(PATCH_SIZE)


if __name__ == '__main__':
    
    params = {'num_chns': NUM_CHNS, 
              'num_region_outputs': NUM_REGION_OUTPUTS, 
              'num_subject_outputs': NUM_SUBJECT_OUTPUTS,
              'feature_depth': FEATURE_DEPTH, 
              'region_features': NUM_REGION_FEATURES,
              'subject_features': NUM_SUBJECT_FEATURES,
              'patch_size': PATCH_SIZE, 
              'num_patches': NUM_PATCHES,
              'num_patches_per_axis':NUM_PATCHES_PER_AXIS,
              'trn_batch_size': TRN_BATCH_SIZE,
              'tst_batch_size': TST_BATCH_SIZE,
              'num_epochs': NUM_EPOCHS, 
              'with_bn': WITH_BN,
              'with_dropout': WITH_DROPOUT, 
              'drop_prob': DROP_PROB, 
              'sharing_weights': SHARING_WEIGHTS,
              'trn_img_path': IMAGE_PATH,
              'tst_img_path': IMAGE_PATH,
              'model_path': MODEL_PATH, 
              'model_name': MODEL_NAME,
              'history_name': HISTORY_NAME,
              'trn_center_mat': CENTER_MAT,
              'tst_center_mat': CENTER_MAT,
              'template_mat':TEMPLATE_MAT,
              'trn_list': TRN_SUBJ_LIST,
              'trn_lbls': TRN_SUBJ_LBLS, 
              'tst_list': TST_SUBJ_LIST,
              'tst_lbls': TST_SUBJ_LBLS, 
              'trn_steps': TRN_STEPS,
              'tst_steps': TST_STEPS}

    Net = HierachicalNet(patch_size=params['patch_size'],
                         num_patches=params['num_patches'],
                         num_patches_per_axis=params['num_patches_per_axis'],
                         num_chns=params['num_chns'], 
                         num_outputs=params['num_subject_outputs'],
                         feature_depth=params['feature_depth'],
                         num_region_features=params['region_features'],
                         num_subject_features=params['subject_features'],
                         with_bn=params['with_bn'],
                         with_dropout=params['with_dropout'],
                         keep_prob=params['drop_prob'],
                         region_sparse=0.005,
                         subject_sparse=0.005,
                         predict_sparse=0.0).get_global_net()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Net.compile(loss=binary_cross_entropy, loss_weights=[1.0, 1.0, 1.0], 
                optimizer=adam, metrics=[acc])
    Net.summary()

    train_flow = data_flow(img_path=params['trn_img_path'], 
                           sample_list=params['trn_list'], 
                           sample_labels=params['trn_lbls'], 
                           center_cors=params['trn_center_mat'],
                           template_cors=params['template_mat'],
                           batch_size=params['trn_batch_size'], 
                           patch_size=params['patch_size'], 
                           num_chns=params['num_chns'], 
                           num_patches=params['num_patches'],
                           num_region_outputs=params['num_region_outputs'], 
                           num_subject_outputs=params['num_subject_outputs'],
                           shuffle_flag=True, shift_flag=True, 
                           scale_flag=True, flip_flag=True,
                           scale_range=[0.95, 1.05], flip_axis=0,
                           shift_range=[-2, -1, 0, 1, 2])
    test_flow = data_flow(img_path=params['tst_img_path'], 
                           sample_list=params['tst_list'], 
                           sample_labels=params['tst_lbls'], 
                           center_cors=params['tst_center_mat'],
                           template_cors=params['template_mat'],
                           batch_size=params['tst_batch_size'], 
                           patch_size=params['patch_size'], 
                           num_chns=params['num_chns'], 
                           num_patches=params['num_patches'],
                           num_region_outputs=params['num_region_outputs'], 
                           num_subject_outputs=params['num_subject_outputs'],
                           shuffle_flag=False, shift_flag=False, 
                           scale_flag=False, flip_flag=False)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
                      filepath=params['model_path']+params['model_name'],
                      monitor='val_subject_outputs_acc', mode='max',
                      save_best_only=True)
    history = keras.callbacks.History()
    
    
    # TRAIN max_queue_size=10
    Net.fit_generator(generator=train_flow,
                      steps_per_epoch=params['trn_steps'],
                      epochs=params['num_epochs'],
                      validation_data=test_flow,
                      validation_steps=params['tst_steps'],
                      callbacks=[checkpoint, history])

    sio.savemat(params['model_path']+params['history_name']+'.mat', 
                {'train_history': history.history})
