# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:47:33 2017

@author: chlian
"""

from keras import backend as K



# F_{\beta}-Measure loss
def fmeasure_loss(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    alpha = 1.0 # alpha = beta**2

    tp = K.sum(y_pred_f * y_true_f)

    FL = 1- ((1.0 + alpha) * tp + K.epsilon()) / \
         (alpha * K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    return FL
    
    
    
def triplet_loss(margin):
    def _triplet_loss(y_true, y_pred):
        
        anchors = y_pred[0::3]
        positives = y_pred[1::3]
        negatives = y_pred[2::3]
        
        pos_diff = K.sum(K.square(anchors - positives), axis=1)
        neg_diff = K.sum(K.square(anchors - negatives), axis=1)
        loss = K.mean(K.maximum(pos_diff - neg_diff + margin, 0), axis=-1)
        
        return loss + 0 * K.sum(y_true)  
        
    return _triplet_loss



# the following function seems to provide a nice gradient function.
# The code from: https://github.com/maciejkula/triplet_recommendations_keras
def bpr_triplet_loss(y_true, y_pred):
    
    anchors = y_pred[0::3]
    positives = y_pred[1::3]
    negatives = y_pred[2::3]
    
    # BPR loss
    loss = K.mean(1.0 - K.sigmoid(K.sum(anchors * positives, axis=-1) -
                                  K.sum(anchors * negatives, axis=-1)), 
                  axis=-1)

    return loss - 0 * K.sum(y_true)



def binary_cross_entropy(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def mse(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)
