# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:06:38 2017

@author: chlian

Train pruned wH-FCN, Step 3 (optimize only subject-level subnet)
"""

import keras
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio

from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adadelta,Adam

from pruned_hfcn import HierachicalNet, acc
from pruned_sequential_loader import data_flow
from loss import binary_cross_entropy, mse


NUM_CHNS = 1
FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
NUM_REGION_FEATURES = 64
NUM_SUBJECT_FEATURES = 64

WITH_BN = True
WITH_DROPOUT = False
DROP_PROB = 0.5

SHARING_WEIGHTS = False

PATCH_SIZE = 25

TRN_BATCH_SIZE = 5
TST_BATCH_SIZE = 5

NUM_EPOCHS = 1000

IMAGE_PATH = "/shenlab/lab_stor4/cflian/mnorm_AD1_ADNC/"

CENTER_MAT_NAME = 'top_lmks_d{0}_ADNI1'.format(PATCH_SIZE)
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

DATA_PARTITION = sio.loadmat('files/'+'data_partition.mat')
TRN_SUBJ_LIST = DATA_PARTITION['trn_list'][0].tolist()
TRN_SUBJ_LBLS = DATA_PARTITION['trn_lbls'][0].tolist()
TST_SUBJ_LIST = DATA_PARTITION['val_list'][0].tolist()
TST_SUBJ_LBLS = DATA_PARTITION['val_lbls'][0].tolist()

TRN_STEPS = int(np.round(len(TRN_SUBJ_LIST) / TRN_BATCH_SIZE))
TST_STEPS = int(np.round(len(TST_SUBJ_LIST) / TST_BATCH_SIZE))


MODEL_PATH = "saved_model/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH) 
MODEL_NAME = 'pruned_whfcn_step3nn.best.hd5'

HISTORY_NAME = 'pruned_whfcn_step3nn_history'


if __name__ == '__main__':
    
    params = {'num_chns': NUM_CHNS, 
              'num_region_outputs': NUM_REGION_OUTPUTS, 
              'num_subject_outputs': NUM_SUBJECT_OUTPUTS,
              'feature_depth': FEATURE_DEPTH, 
              'region_features': NUM_REGION_FEATURES,
              'subject_features': NUM_SUBJECT_FEATURES,
              'patch_size': PATCH_SIZE, 
              'num_patches': NUM_PATCHES,
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
              'pruned_pths': PRUNED_PATCHES,
              'pruned_nns': PRUNED_NEIGHBORS,
              'trn_list': TRN_SUBJ_LIST,
              'trn_lbls': TRN_SUBJ_LBLS, 
              'tst_list': TST_SUBJ_LIST,
              'tst_lbls': TST_SUBJ_LBLS, 
              'trn_steps': TRN_STEPS,
              'tst_steps': TST_STEPS}
              
    Net = HierachicalNet(patch_size=params['patch_size'],
                         num_patches=params['num_patches'],
                         pruned_nns=params['pruned_nns'],
                         num_chns=params['num_chns'], 
                         num_outputs=params['num_subject_outputs'],
                         feature_depth=params['feature_depth'],
                         num_region_features=params['region_features'],
                         num_subject_features=params['subject_features'],
                         with_bn=params['with_bn'],
                         with_dropout=params['with_dropout'],
                         drop_prob=params['drop_prob'],
                         region_sparse=0.0,
                         subject_sparse=0.0).get_global_net()
                         
    sgd = SGD(lr=1e-2, decay=1e-5, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    for i_layer in range(len(Net.layers)):
        Net.layers[i_layer].trainable = False
    Net.get_layer('subject_conv').trainable = True
    Net.get_layer('subject_outputs').trainable = True
    
    Net.compile(loss=binary_cross_entropy, loss_weights=[0.0, 0.0, 1.0],
                optimizer=adam, metrics=[acc])
    Net.load_weights(params['model_path']+'pruned_whfcn_step2.best.hd5')
    Net.summary()
         
    train_flow = data_flow(img_path=params['trn_img_path'], 
                           sample_list=params['trn_list'], 
                           sample_labels=params['trn_lbls'], 
                           center_cors=params['trn_center_mat'], 
                           patch_idxs=params['pruned_pths'],
                           batch_size=params['trn_batch_size'], 
                           patch_size=params['patch_size'], 
                           num_chns=params['num_chns'], 
                           num_patches=params['num_patches'],
                           num_region_outputs=params['num_region_outputs'], 
                           num_subject_outputs=params['num_subject_outputs'],
                           shuffle_flag=True, shift_flag=True,
                           scale_flag=False, flip_flag=True,
                           scale_range=[0.99, 1.01], flip_axis=0,
                           shift_range=[-2, -1, 0, 1, 2])
    test_flow = data_flow(img_path=params['tst_img_path'],
                           sample_list=params['tst_list'],
                           sample_labels=params['tst_lbls'],
                           center_cors=params['tst_center_mat'],
                           patch_idxs=params['pruned_pths'],
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
