# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:48:30 2017

@author: chlian
"""

import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
import itertools
import random

def data_flow(img_path, sample_list, sample_labels, center_cors, 
              template_cors, batch_size, patch_size, num_chns, 
              num_patches, num_region_outputs, num_subject_outputs,
              shuffle_flag=False, shift_flag=False, scale_flag=False, 
              flip_flag=False, scale_range=[0.95, 1.05], flip_axis=0, 
              shift_range=[-2, -1, 0, 1, 2]):

    if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')
            
    margin = int(np.floor((patch_size-1)/2.0))
    
    input_shape = (batch_size, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (batch_size, num_patches)
    mid_ot_shape = (batch_size, num_region_outputs)
    high_ot_shape = (batch_size, num_subject_outputs)
    
    center_z = np.unique(template_cors[0,:]).tolist()
    center_y = np.unique(template_cors[1,:]).tolist()
    center_x = np.unique(template_cors[2,:]).tolist()
    
    while True:
        if shuffle_flag:
            sample_list = np.array(sample_list)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_list))
            np.take(sample_list, permut, out=sample_list)
            np.take(sample_labels, permut, out=sample_labels)
            sample_list = sample_list.tolist()
            sample_labels = sample_labels.tolist()
            
        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = [np.ones(low_ot_shape, dtype='int8'), 
                   np.ones(mid_ot_shape, dtype='int8'),
                   np.ones(high_ot_shape, dtype='int8')]
                   
        i_batch = 0
        for i_iter in range(len(sample_list)):
            # random flip
            if flip_flag:
                flip_action = np.random.randint(0, 2)
            else:
                flip_action = 0
                
            i_subject = sample_list[i_iter]
                
            img_dir = img_path + 'Img_{0}.nii.gz'
            I = sitk.ReadImage(img_dir.format(i_subject))
            img = np.array(sitk.GetArrayFromImage(I))
            
            # random rescale
            if scale_flag:
                scale = np.random.uniform(scale_range[0], scale_range[1], 3)
                img = nd.interpolation.zoom(img, scale, order=1)
                
            i_patch = 0
            for i_x in center_x:
                for i_z in center_z:
                    for i_y in center_y:
                            
                        idxs_z = np.where(template_cors[0, :]==i_z)[0]
                        idxs_y = np.where(template_cors[1, :]==i_y)[0]
                        idxs_x = np.where(template_cors[2, :]==i_x)[0]
                            
                        idx_cor = [v for v in idxs_z if v in idxs_y if v in idxs_x][0]
                        z_cor = center_cors[0, idx_cor, i_subject-1]
                        y_cor = center_cors[1, idx_cor, i_subject-1]
                        x_cor = center_cors[2, idx_cor, i_subject-1]
                            
                        if shift_flag:
                            x_scor = x_cor + np.random.choice(shift_range)
                            y_scor = y_cor + np.random.choice(shift_range)
                            z_scor = z_cor + np.random.choice(shift_range)
                        else:
                            x_scor, y_scor, z_scor = x_cor, y_cor, z_cor
              
                        img_patch = img[z_scor-margin: z_scor+margin+1, 
                                        y_scor-margin: y_scor+margin+1, 
                                        x_scor-margin: x_scor+margin+1]
                                    
                        if flip_action == 1:
                            if flip_axis == 0:
                                img_patch = img_patch[:, :, ::-1]
                            elif flip_axis == 1:
                                img_patch = img_patch[:, ::-1, :]
                            elif flip_axis == 2:
                                img_patch = img_patch[::-1, :, :]

                        inputs[i_patch][i_batch, 0, :, :, :] = img_patch
                        i_patch += 1
                    
            outputs[0][i_batch,:] = sample_labels[i_iter]*outputs[0][i_batch,:]
            outputs[1][i_batch,:] = sample_labels[i_iter]*outputs[1][i_batch,:]
            outputs[2][i_batch,:] = sample_labels[i_iter]*outputs[2][i_batch,:]
                    
            i_batch += 1
                
            if i_batch == batch_size:
                yield(inputs, outputs)
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = [np.ones(low_ot_shape, dtype='int8'), 
                           np.ones(mid_ot_shape, dtype='int8'),
                           np.ones(high_ot_shape, dtype='int8')]
                i_batch = 0
                    

def data_flow_template(img_path, sample_list, sample_labels, center_cors, 
              template_cors, batch_size, patch_size, num_chns, 
              num_patches, num_region_outputs, num_subject_outputs,
              shuffle_flag=False, shift_flag=False, scale_flag=False, 
              flip_flag=False, scale_range=[0.95, 1.05], flip_axis=0, 
              shift_range=[-2, -1, 0, 1, 2]):

    if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')
            
    margin = int(np.floor((patch_size-1)/2.0))
    
    input_shape = (batch_size, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (batch_size, num_patches)
    mid_ot_shape = (batch_size, num_region_outputs)
    high_ot_shape = (batch_size, num_subject_outputs)
    
    center_z = np.unique(template_cors[0,:]).tolist()
    center_y = np.unique(template_cors[1,:]).tolist()
    center_x = np.unique(template_cors[2,:]).tolist()
    
    while True:
        if shuffle_flag:
            sample_list = np.array(sample_list)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_list))
            np.take(sample_list, permut, out=sample_list)
            np.take(sample_labels, permut, out=sample_labels)
            sample_list = sample_list.tolist()
            sample_labels = sample_labels.tolist()
            
        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = [np.ones(low_ot_shape, dtype='int8'), 
                   np.ones(mid_ot_shape, dtype='int8'),
                   np.ones(high_ot_shape, dtype='int8')]
                   
        i_batch = 0
        for i_iter in range(len(sample_list)):
            # random flip
            if flip_flag:
                flip_action = np.random.randint(0, 2)
            else:
                flip_action = 0
                
            i_subject = sample_list[i_iter]
                
            img_dir = img_path + 'Img_{0}.nii.gz'
            I = sitk.ReadImage(img_dir.format(i_subject))
            img = np.array(sitk.GetArrayFromImage(I))
            
            # random rescale
            if scale_flag:
                scale = np.random.uniform(scale_range[0], scale_range[1], 3)
                img = nd.interpolation.zoom(img, scale, order=1)
                
            i_patch = 0
            for i_x in center_x:
                for i_z in center_z:
                    for i_y in center_y:
                            
                        idxs_z = np.where(template_cors[0, :]==i_z)[0]
                        idxs_y = np.where(template_cors[1, :]==i_y)[0]
                        idxs_x = np.where(template_cors[2, :]==i_x)[0]
                            
                        idx_cor = [v for v in idxs_z if v in idxs_y if v in idxs_x][0]
                        z_cor = template_cors[0, idx_cor]
                        y_cor = template_cors[1, idx_cor]
                        x_cor = template_cors[2, idx_cor]
                            
                        if shift_flag:
                            x_scor = x_cor + np.random.choice(shift_range)
                            y_scor = y_cor + np.random.choice(shift_range)
                            z_scor = z_cor + np.random.choice(shift_range)
                        else:
                            x_scor, y_scor, z_scor = x_cor, y_cor, z_cor
              
                        img_patch = img[z_scor-margin: z_scor+margin+1, 
                                        y_scor-margin: y_scor+margin+1, 
                                        x_scor-margin: x_scor+margin+1]
                                    
                        if flip_action == 1:
                            if flip_axis == 0:
                                img_patch = img_patch[:, :, ::-1]
                            elif flip_axis == 1:
                                img_patch = img_patch[:, ::-1, :]
                            elif flip_axis == 2:
                                img_patch = img_patch[::-1, :, :]

                        inputs[i_patch][i_batch, 0, :, :, :] = img_patch
                        i_patch += 1
                    
            outputs[0][i_batch,:] = sample_labels[i_iter]*outputs[0][i_batch,:]
            outputs[1][i_batch,:] = sample_labels[i_iter]*outputs[1][i_batch,:]
            outputs[2][i_batch,:] = sample_labels[i_iter]*outputs[2][i_batch,:]
                    
            i_batch += 1
                
            if i_batch == batch_size:
                yield(inputs, outputs)
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = [np.ones(low_ot_shape, dtype='int8'), 
                           np.ones(mid_ot_shape, dtype='int8'),
                           np.ones(high_ot_shape, dtype='int8')]
                i_batch = 0


                    
def tst_data_flow(img_path, sample_idx, sample_lbl, center_cors,
                  template_cors, patch_size, num_chns, num_patches,
                  num_region_outputs, num_subject_outputs):
    
    input_shape = (1, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (1, num_patches)
    mid_ot_shape = (1, num_region_outputs)
    high_ot_shape = (1, num_subject_outputs)
    
    margin = int(np.floor((patch_size-1)/2.0))
    
    img_dir = img_path + 'Img_{0}.hdr'
    I = sitk.ReadImage(img_dir.format(sample_idx))
    img = np.array(sitk.GetArrayFromImage(I))
    
    inputs = []
    for i_input in range(num_patches):
        inputs.append(np.zeros(input_shape, dtype='float32'))
    
    center_z = np.unique(template_cors[0,:]).tolist()
    center_y = np.unique(template_cors[1,:]).tolist()
    center_x = np.unique(template_cors[2,:]).tolist()
    
    i_patch = 0
    for i_x in center_x:
        for i_z in center_z:
            for i_y in center_y:
                idxs_z = np.where(template_cors[0, :]==i_z)[0]
                idxs_y = np.where(template_cors[1, :]==i_y)[0]
                idxs_x = np.where(template_cors[2, :]==i_x)[0]
                        
                idx_cor = [v for v in idxs_z if v in idxs_y if v in idxs_x][0]
                z_cor = center_cors[0, idx_cor, sample_idx-1]
                y_cor = center_cors[1, idx_cor, sample_idx-1]
                x_cor = center_cors[2, idx_cor, sample_idx-1]

                img_patch = img[z_cor-margin: z_cor+margin+1,
                                y_cor-margin: y_cor+margin+1,
                                x_cor-margin: x_cor+margin+1]
                inputs[i_patch][0, 0, :, :, :] = img_patch
                i_patch += 1

    outputs = [sample_lbl * np.ones(low_ot_shape, dtype='float32'),
               sample_lbl * np.ones(mid_ot_shape, dtype='float32'),
               sample_lbl * np.ones(high_ot_shape, dtype='float32')]

    return inputs, outputs
 
                   
def tst_data_flow_template(img_path, sample_idx, sample_lbl, center_cors,
                  template_cors, patch_size, num_chns, num_patches,
                  num_region_outputs, num_subject_outputs):
    
    input_shape = (1, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (1, num_patches)
    mid_ot_shape = (1, num_region_outputs)
    high_ot_shape = (1, num_subject_outputs)
    
    margin = int(np.floor((patch_size-1)/2.0))
    
    img_dir = img_path + 'Img_{0}.hdr'
    I = sitk.ReadImage(img_dir.format(sample_idx))
    img = np.array(sitk.GetArrayFromImage(I))
    
    inputs = []
    for i_input in range(num_patches):
        inputs.append(np.zeros(input_shape, dtype='float32'))
    
    center_z = np.unique(template_cors[0,:]).tolist()
    center_y = np.unique(template_cors[1,:]).tolist()
    center_x = np.unique(template_cors[2,:]).tolist()
    
    i_patch = 0
    for i_x in center_x:
        for i_z in center_z:
            for i_y in center_y:
                idxs_z = np.where(template_cors[0, :]==i_z)[0]
                idxs_y = np.where(template_cors[1, :]==i_y)[0]
                idxs_x = np.where(template_cors[2, :]==i_x)[0]
                        
                idx_cor = [v for v in idxs_z if v in idxs_y if v in idxs_x][0]
                z_cor = template_cors[0, idx_cor]
                y_cor = template_cors[1, idx_cor]
                x_cor = template_cors[2, idx_cor]

                img_patch = img[z_cor-margin: z_cor+margin+1,
                                y_cor-margin: y_cor+margin+1,
                                x_cor-margin: x_cor+margin+1]
                inputs[i_patch][0, 0, :, :, :] = img_patch
                i_patch += 1

    outputs = [sample_lbl * np.ones(low_ot_shape, dtype='float32'),
               sample_lbl * np.ones(mid_ot_shape, dtype='float32'),
               sample_lbl * np.ones(high_ot_shape, dtype='float32')]

    return inputs, outputs