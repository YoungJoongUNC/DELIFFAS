#############################################################################################
# import
#############################################################################################

import tensorflow as tf
import numpy as np
import pdb
#############################################################################################
# img2mse
#############################################################################################

def img2mse(x, y):
    return  tf.reduce_mean(tf.square(x - y))

def img2mseL1(x, y):
    return  tf.reduce_mean(tf.abs(x - y))

def img2ssim(x, y):
    '''
    :param x: pred [B,H,W,C] value range [0,1]
    :param y: gt [B,H,W,C] value range [0,1]
    '''

    # range [-1,1]
    raw_ssim = tf.image.ssim(x, y, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    # range [0,1]
    ssim_score = 0.5 * (raw_ssim + 1.0)
    return tf.reduce_mean(ssim_score)

def maskedimg2mseL1(x, y, FGpixels):
    '''
    nbatch is distributed batch (self.args.N_batch / GPUs)
    :param x: predictions [nbatch, # FG pixels MAX, 3]
    :param y: GT [nbatch, # FG pixels MAX, 3]
    :param FGpixels: [nbatch, 1]
    :return: masked L1 loss (scalar)
    '''
    sum = tf.reduce_sum(tf.abs(x - y), axis=[1, 2])[:,None] # shape [nbatch,]->[nbatch,1]
    sum = sum / tf.cast(FGpixels,dtype=tf.float32)

    # finally, average across batches
    return tf.reduce_mean(sum)

#############################################################################################
# mse2psnr
#############################################################################################

def mse2psnr(x): return -10.*tf.math.log(x)/tf.math.log(10.)

#############################################################################################
# to8b
#############################################################################################

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

#############################################################################################
# variance loss
#############################################################################################

def varianceLoss(depthValues, weights, expectedDepth):

    # depthValues  --> [rays, samples]
    # weights      --> [rays, samples]
    # meanDepth    --> [rays]

    expectedDepth = tf.reshape(expectedDepth, [-1, 1])
    expectedDepth = tf.tile(expectedDepth, [1, tf.shape(depthValues)[1]])

    varianceSquaredPerRay = tf.square(tf.reduce_sum(tf.square(depthValues-expectedDepth) * weights, 1))

    return tf.reduce_mean(varianceSquaredPerRay)

