# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:48:33 2017

@author: chlian
"""

import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio


def data_flow(img_path, sample_list, sample_labels, 
              center_cors, patch_idxs, batch_size, 
              patch_size, num_chns, num_patches,
              num_region_outputs, num_subject_outputs,
              shuffle_flag=True, shift_flag=True, scale_flag=False, 
              flip_flag=False, scale_range=[0.95, 1.05], flip_axis=0, 
              shift_range=[-2, -1, 0, 1, 2]):
    
    if flip_axis < 0 or flip_axis > 2:
            raise ValueError('flip axis should be 0 -> x, 1 -> y, or 2 -> z.')
            
    margin = int(np.floor((patch_size-1)/2.0))
    
    input_shape = (batch_size, num_chns, patch_size, patch_size, patch_size)
    low_ot_shape = (batch_size, num_patches)
    mid_ot_shape = (batch_size, num_region_outputs)
    high_ot_shape = (batch_size, num_subject_outputs)
    
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
                
            img_dir = img_path + 'Img_{0}.hdr'
            I = sitk.ReadImage(img_dir.format(i_subject))
            img = np.array(sitk.GetArrayFromImage(I))
    
            # random rescale
            if scale_flag:
                scale = np.random.uniform(scale_range[0], scale_range[1], 3)
                img = nd.interpolation.zoom(img, scale, order=1)
                
            center = center_cors[:, :, i_subject-1]
            center_z = center[2,:].tolist()
            center_y = center[1,:].tolist()
            center_x = center[0,:].tolist()
            for i_patch in range(num_patches):
                x_cor = center_x[patch_idxs[i_patch]]
                y_cor = center_y[patch_idxs[i_patch]]
                z_cor = center_z[patch_idxs[i_patch]]
                    
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
                    
            outputs[0][i_batch, :] = \
                    sample_labels[i_iter] * outputs[0][i_batch, :]
            outputs[1][i_batch, :] = \
                    sample_labels[i_iter] * outputs[1][i_batch, :]
            outputs[2][i_batch, :] = \
                    sample_labels[i_iter] * outputs[2][i_batch, :]
                    
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
                  patch_idxs, patch_size, num_chns, num_patches,
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
    
    center = center_cors[:, :, sample_idx-1]
    center_z = center[2,:].tolist()
    center_y = center[1,:].tolist()
    center_x = center[0,:].tolist()
    for i_patch in range(num_patches):
        x_cor = center_x[patch_idxs[i_patch]]
        y_cor = center_y[patch_idxs[i_patch]]
        z_cor = center_z[patch_idxs[i_patch]]
        img_patch = img[z_cor-margin: z_cor+margin+1,
                        y_cor-margin: y_cor+margin+1,
                        x_cor-margin: x_cor+margin+1]
        inputs[i_patch][0, 0, :, :, :] = img_patch

    outputs = [sample_lbl * np.ones(low_ot_shape, dtype='float32'),
               sample_lbl * np.ones(mid_ot_shape, dtype='float32'),
               sample_lbl * np.ones(high_ot_shape, dtype='float32')]

    return inputs, outputs
