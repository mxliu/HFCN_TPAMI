# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:26:26 2017

@author: chlian

Hierarchical FCN (H-FCN) with anatomical-landmark-based
prior knowledge (i.e., wH-FCN)
"""

import numpy as np
import scipy.io as sio

from keras.models import Model
from keras import backend as K
from keras import layers

from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv3D, Cropping3D, Conv1D
from keras.layers.pooling import MaxPooling3D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Reshape, Permute
from keras import regularizers
from keras.regularizers import Regularizer
from keras.optimizers import SGD, Adam

from loss import binary_cross_entropy, mse


class L2Normalization(layers.Layer):
    
    def comput_output_shape(self, input_shape):
        return tuple(input_shape)
    
    def call(self, inputs):
        inputs = K.l2_normalize(inputs, axis=1)
        return inputs


def acc(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


class StructuredSparse(Regularizer):
    
    def __init__(self, C=0.001):
        self.C = K.cast_to_floatx(C)
    
    def __call__(self, kernel_matrix):
        #const_coeff = np.sqrt(K.int_shape(kernel_matrix)[-1] * K.int_shape(kernel_matrix)[-2])
        return self.C * \
               K.sum(K.sqrt(K.sum(K.sum(K.square(kernel_matrix), axis=-1), axis=-1))) 

    def get_config(self):
        return {'C': float(self.C)}



class HierachicalNet(object):
    
    def __init__(self, 
                 patch_size,
                 num_patches,
                 num_neighbors,
                 neighbor_matrix,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 num_region_features=64,
                 num_subject_features=64,
                 with_bn=True,
                 with_dropout=True,
                 drop_prob=0.5,
                 region_sparse=0.0,
                 subject_sparse=0.0):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.num_patches = num_patches
        
        self.num_neighbors = num_neighbors 
        self.nn_mat = np.array(neighbor_matrix)[:, :num_neighbors]
        
        self.patch_size = patch_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop_prob = drop_prob
            
        self.num_region_features = num_region_features
        self.num_subject_features = num_subject_features
        
        self.region_sparse = region_sparse
        self.subject_sparse = subject_sparse

   
    def get_global_net(self):
        
        embed_net = self.base_net()
        
        # inputs
        inputs = []
        for i_input in range(self.num_patches):
            input_name = 'input_{0}'.format(i_input+1)
            inputs.append(Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size), 
                        name=input_name))
        
        #+++++++++++++++++++++++++++++#                      
        ##   patch-level processing   #
        #+++++++++++++++++++++++++++++# 
        patch_features_list, patch_probs_list = [], []
        for i_input in range(self.num_patches):
            feature_map, class_prob = embed_net(inputs[i_input])
            patch_features_list.append(feature_map)
            patch_probs_list.append(class_prob)
            
        
        patch_outputs = Concatenate(name='patch_outputs', 
                                    axis=1)(patch_probs_list)
                                    
        #+++++++++++++++++++++++++++++++#                      
        ##   region-level processing   ## 
        #+++++++++++++++++++++++++++++++#
        region_features_list, region_probs_list = [], []
        i_region = 1
        for i_input in range(self.num_patches):
            nn_features, nn_probs = [], []
            for i_neighbor in range(self.num_neighbors):
                nn_features.append(patch_features_list[self.nn_mat[i_input, i_neighbor]])
                nn_probs.append(patch_probs_list[self.nn_mat[i_input, i_neighbor]])   
            region_input_features = Concatenate(axis=1)(nn_features)
            region_input_probs = Concatenate(axis=1)(nn_probs)
            region_input_probs = Reshape([self.num_neighbors, 1])(region_input_probs)
            
            in_name = 'region_input_{0}'.format(i_region)
            region_input = Concatenate(name=in_name, 
                                       axis=-1)([region_input_probs, region_input_features])
            conv_name = 'region_conv_{0}'.format(i_region)
            region_feature = Conv1D(filters=self.num_region_features,
                                    kernel_size=self.num_neighbors,
                                    kernel_regularizer=StructuredSparse(self.region_sparse),
                                    padding='valid', 
                                    name=conv_name)(region_input)
            if self.with_bn:
                region_feature = BatchNormalization(axis=-1)(region_feature)
            region_feature = Activation('relu')(region_feature)
            if self.with_dropout:
                region_feature = Dropout(self.drop_prob)(region_feature)

            ot_name = 'region_prob_{0}'.format(i_region)
            region_prob = Dense(units=1, activation='sigmoid',
                                name=ot_name)(Flatten()(region_feature))
                                
            region_features_list.append(region_feature)
            region_probs_list.append(region_prob)
            
            i_region += 1          
            
        region_outputs = Concatenate(name='region_outputs', 
                                     axis=1)(region_probs_list)
        region_features = Concatenate(name='region_features', 
                                      axis=1)(region_features_list)
        region_probs = Reshape([self.num_patches, 1], 
                               name='region_probs')(region_outputs)
          
        region_feat_prob = Concatenate(name='region_features_probs', 
                                      axis=-1)([region_probs, region_features]) 

        #+++++++++++++++++++++++++++++++#                      
        ##   subject-level processing   #
        #+++++++++++++++++++++++++++++++#
        subject_feature = Conv1D(filters=self.num_subject_features,
                                 kernel_size=self.num_patches,
                                 kernel_regularizer=StructuredSparse(self.subject_sparse),
                                 padding='valid', 
                                 name='subject_conv')(region_feat_prob)
        if self.with_bn:
            subject_feature = BatchNormalization(axis=-1)(subject_feature)
        subject_feature = Activation('relu')(subject_feature)
        if self.with_dropout:
            subject_feature = Dropout(self.drop_prob)(subject_feature)
            
        # subject-level units
        subject_units = Flatten(name='subject_level_units')(subject_feature)
        
        #+++++++++++#                      
        #   MODEL   #
        #+++++++++++#
        subject_outputs = Dense(units=1, activation='sigmoid',
                                name='subject_outputs')(subject_units)
        
        outputs = [patch_outputs, region_outputs, subject_outputs] 
        model = Model(inputs=inputs, outputs=outputs)
      
        return model
    
    
            
    def base_net(self):
        
        """ Input with channel first"""
        inputs = Input((self.num_chns, self.patch_size, 
                        self.patch_size, self.patch_size))
                        
        """ 1st convolution"""                
        conv1 = Conv3D(filters=self.feature_depth[0], 
                       kernel_size=[4, 4, 4],
                       padding='valid', 
                       data_format='channels_first')(inputs)
        if self.with_bn:
            conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = Activation('relu')(conv1)

        """ 2nd convolution"""
        conv2 = Conv3D(filters=self.feature_depth[1], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv1)
        if self.with_bn:
            conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = Activation('relu')(conv2)
 
        """ pooling 1"""
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv2)
 
        """ 3rd convolution"""
        conv3 = Conv3D(filters=self.feature_depth[2], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool1)
        if self.with_bn:
            conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = Activation('relu')(conv3)

        """ 4th convolution"""
        conv4 = Conv3D(filters=self.feature_depth[3], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(conv3)
        if self.with_bn:
            conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = Activation('relu')(conv4)
        
        """ pooling 2"""
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), 
                             data_format='channels_first')(conv4)
                             
        """ 5th convolution"""
        conv5 = Conv3D(filters=self.feature_depth[4], 
                       kernel_size=[3, 3, 3],
                       padding='valid', 
                       data_format='channels_first')(pool2)
        if self.with_bn:
            conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = Activation('relu')(conv5)

        """ 6th convolution"""
        conv6 = Conv3D(filters=self.feature_depth[5], 
                       kernel_size=[1, 1, 1],
                       padding='valid', 
                       data_format='channels_first')(conv5)
        if self.with_bn:
            conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = Activation('relu')(conv6)
        if self.with_dropout:
            conv6 = Dropout(self.drop_prob)(conv6)

        feature_map = Reshape((1, self.feature_depth[-1]),
                              name='patch_features')(conv6)

        class_prob = Dense(units=1, activation='sigmoid',
                           name='patch_prob')(Flatten()(feature_map))
        
        model = Model(inputs=inputs, 
                      outputs=[feature_map, class_prob], 
                      name='base_net')

        model.summary()
        
        return model
        
        
        
if __name__ == '__main__':
    
    feature_depth = [32, 32, 64, 64, 128, 32]
    neighbor_matrix = \
    sio.loadmat('files/top_lmks_neighbor_matrix.mat')
    neighbor_matrix = neighbor_matrix['neighbor_matrix'].astype('uint8') - 1
    T = HierachicalNet(patch_size=25, 
                       num_chns=1, 
                       num_outputs=1,
                       num_patches=76,
                       num_neighbors=5,
                       neighbor_matrix=neighbor_matrix,
                       num_region_features=32,
                       num_subject_features=32,
                       feature_depth=feature_depth)
                 
    Net = T.get_global_net()
    Net.summary()

