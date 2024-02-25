
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications.vgg19 import preprocess_input

########################################################################################################################
# Create the VGG Model
########################################################################################################################

def createVGGLossModel(resolutionU, resolutionV):

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', pooling= 'avg', input_shape = [resolutionV,resolutionU,3])
    tf.print(vgg.summary())
    vgg.trainable = False
    content_layers = 'block3_conv1'
    lossModel = models.Model([vgg.input], vgg.get_layer(content_layers).output, name = 'vggL')

    tf.print(lossModel.summary())
    
    return lossModel

########################################################################################################################
# Loss
########################################################################################################################

def lossVGG(X,Y, lossModel):

    Xt = preprocess_input(X*255)
    Yt = preprocess_input(Y*255)

    vggX = lossModel(Xt)
    vggY = lossModel(Yt)

    return tf.reduce_mean(tf.square(vggY-vggX))