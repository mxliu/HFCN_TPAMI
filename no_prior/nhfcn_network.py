# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:35:48 2017

@author: chlian

Hierarchical FCN (H-FCN) without anatomical-landmark-based
prior knowledge (i.e., nH-FCN)
"""


import numpy as np

from keras.models import Model
from keras import backend as K
from keras import layers

from keras.layers import Input, Flatten, Add
from keras.layers.convolutional import Conv3D, UpSampling3D, Cropping3D, Conv1D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D
from keras.layers.merge import Concatenate, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Reshape, Permute
from keras import regularizers
from keras.layers.local import LocallyConnected1D, LocallyConnected2D
from keras.regularizers import Regularizer
from keras.optimizers import SGD, RMSprop, Adadelta,Adam

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
    
        return self.C * \
               K.sum(K.sqrt(K.sum(K.sum(K.square(kernel_matrix), axis=-1), axis=-1)))

    def get_config(self):
        return {'C': float(self.C)}



class HierachicalNet(object):
    
    def __init__(self, 
                 patch_size,
                 num_patches,
                 num_patches_per_axis,
                 num_chns, 
                 num_outputs, 
                 feature_depth,
                 num_region_features=64,
                 num_subject_features=16,
                 with_bn=False,
                 with_dropout=False,
                 keep_prob=0.5,
                 region_sparse=0.0,
                 subject_sparse=0.0,
                 predict_sparse=0.0,
                 sharing_weights=False):
        self.num_chns = num_chns
        self.num_outputs = num_outputs
        self.num_patches = num_patches
        self.num_per_axis = num_patches_per_axis
        self.patch_size = patch_size
        self.feature_depth = feature_depth
        self.with_bn = with_bn
        self.with_dropout = with_dropout
        if with_dropout:
            self.keep_prob = keep_prob
            
        self.num_region_features = num_region_features
        self.num_subject_features = num_subject_features
        
        self.sharing_weights = sharing_weights

        self.region_sparse = region_sparse
        self.subject_sparse = subject_sparse
        self.predict_sparse = predict_sparse


    def get_global_net(self, optimizer='Adam', metrics=[acc]):
        
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
        
        patch_outputs = Concatenate(axis=-1, 
                                  name='patch_outputs')(patch_probs_list)
        patch_features_flatten = Concatenate(axis=-1, 
                                  name='merged_features')(patch_features_list)
        patch_probs = Reshape([1]+self.num_per_axis, 
                              name='patch_probs')(patch_outputs)
        patch_features = Reshape([self.feature_depth[-1]]+self.num_per_axis,
                                  name='patch_features')(patch_features_flatten)
          
        patch_feat_prob = Concatenate(name='merge_patch_outputs', 
                                      axis=1)([patch_probs, patch_features])

        #+++++++++++++++++++++++++++++++#                      
        ##   region-level processing   ## 
        #+++++++++++++++++++++++++++++++#
        if not self.sharing_weights:   
            region_features_list = []
            region_probs_list = []
            i_region = 1
            for cor_x in range(int(self.num_per_axis[2]-2)+1):
                for cor_z in range(int(self.num_per_axis[0]-2)+1):
                    for cor_y in range(int((self.num_per_axis[1]-2))+1):
                    
                        margin = ((cor_z*1, self.num_per_axis[0]-2-cor_z*1), 
                              (cor_y*1, self.num_per_axis[1]-2-cor_y*1), 
                              (cor_x*1, self.num_per_axis[2]-2-cor_x*1))
                        crop_input = Cropping3D(margin, 
                                     data_format='channels_first')(patch_feat_prob)
                    
                        in_name = 'region_conv_{0}'.format(i_region)
                        crop_feature = Conv3D(filters=self.num_region_features, 
                                              kernel_size=[2, 2, 2],
                                              kernel_regularizer=StructuredSparse(self.region_sparse),
                                              padding='valid',
                                              data_format='channels_first',
                                              name=in_name)(crop_input)
                        if self.with_bn:
                            crop_feature = BatchNormalization(axis=1)(crop_feature)
                        crop_feature = Activation('relu')(crop_feature)
                        if self.with_dropout:
                            crop_feature = Dropout(self.keep_prob)(crop_feature)
                        
                        ot_name = 'region_prob_{0}'.format(i_region)
                        crop_prob = Dense(units=1, activation='sigmoid',
                                          name=ot_name)(Flatten()(crop_feature)) 
                    
                        crop_feature = \
                        Reshape((self.num_region_features, -1))(crop_feature)
                        
                        region_features_list.append(crop_feature)
                        region_probs_list.append(crop_prob)
                    
                        i_region += 1
        
          
            region_features = Concatenate(name='region_features', 
                                       axis=-1)(region_features_list)
            region_outputs = Concatenate(name='region_outputs',
                                        axis=-1)(region_probs_list)
            region_probs = Reshape((1, -1))(region_outputs)
        
            region_feat_prob = Concatenate(name='merge_region_outputs', axis=1)(
                                        [region_probs, region_features])
            region_feat_prob = Reshape((self.num_region_features+1, 
                                        self.num_per_axis[0]-1, 
                                        self.num_per_axis[1]-1, 
                                        self.num_per_axis[2]-1), 
                                        name='reshape_region_outputs')(region_feat_prob)
                                    
        else:
            region_features = Conv3D(filters=self.num_region_features, 
                                     kernel_size=[2, 2, 2],
                                     kernel_regularizer=StructuredSparse(self.region_sparse),
                                     strides=[1, 1, 1],
                                     padding='valid', 
                                     activation='relu',
                                     data_format='channels_first',
                                     name='region_features')(patch_feat_prob)
            
            if self.with_bn:
                region_features = BatchNormalization(axis=1)(region_features)
            region_features = Activation('relu')(region_features)
            if self.with_dropout:
                region_features = Dropout(self.keep_prob)(region_features)
                            
            region_features_shaped = \
                Reshape((self.num_region_features, -1))(region_features)
            region_features_shaped = Permute((2,1))(region_features_shaped)
                      
            region_probs = LocallyConnected1D(filters=1, kernel_size=1, 
                                        data_format='channels_first',
                                        activation='sigmoid')(region_features_shaped)                          
            region_outputs = Flatten(name='region_outputs')(region_probs)
            
            region_probs = Reshape((1, self.num_per_axis[0]-1, 
                                self.num_per_axis[1]-1, self.num_per_axis[2]-1), 
                                name='region_probs')(Permute((2,1))(region_probs))
            region_feat_prob = Concatenate(name='merge_region_outputs', axis=1)(
                                           [region_probs, region_features])
        
        #+++++++++++++++++++++++++++++++#                      
        ##   subject-level processing   #
        #+++++++++++++++++++++++++++++++#
        subject_features = Conv3D(filters=self.num_subject_features, 
                                  kernel_size=(self.num_per_axis[0]-1,
                                               self.num_per_axis[1]-1, 
                                               self.num_per_axis[2]-1),
                                  kernel_regularizer=StructuredSparse(self.subject_sparse),
                                  padding='valid',
                                  data_format='channels_first',
                                  name='subject_conv')(region_feat_prob)
        if self.with_bn:
            subject_features = BatchNormalization(axis=1)(subject_features)
        subject_features = Activation('relu')(subject_features)
        if self.with_dropout:
            subject_features = Dropout(self.keep_prob)(subject_features)

        subject_units = Flatten(name='subject_level_units')(subject_features)
        
        
        # merge units
        """
        total_units = Concatenate(name='merged_units',
                                  axis=1)([patch_units, region_units, subject_units])
        """
        subject_outputs = Dense(units=1, activation='sigmoid',
                                kernel_regularizer=regularizers.l1(self.predict_sparse),
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
            conv6 = Dropout(self.keep_prob)(conv6)
         
            
        feature_map = Reshape((self.feature_depth[-1], 1),
                              name='patch_features')(conv6)
        
        class_prob = Dense(units=1, activation='sigmoid',
                           name='patch_prob')(Flatten()(feature_map))

        model = Model(inputs=inputs, 
                      outputs=[feature_map, class_prob], 
                      name='base_net')

        model.summary()
        
        return model
        
        
        
if __name__ == '__main__':
    
    feature_depth = [32, 64, 64, 128, 128, 64]
    
    T = HierachicalNet(patch_size=25, 
                       num_chns=1, 
                       num_outputs=1,
                       num_patches=120,
                       num_region_features=64,
                       num_subject_features=64,
                       num_patches_per_axis = [4, 6, 5],
                       feature_depth=feature_depth, 
                       sharing_weights=True)
                 
    net = T.get_global_net()
    
    
    
