
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import CustomTFOperators.CppPath as CppPath

########################################################################################################################
# Load custom operators
########################################################################################################################

customOperators = tf.load_op_library(CppPath.getCustomOperatorPath())

########################################################################################################################
# PerspectiveNPointGpu class
########################################################################################################################

class PerspectiveNPointGpu:

    def __init__(self, cameraFilePath = '', numberOfBatches=0, numberOfMarkers=0, preds2D = None, predsConf = None, markers3D = None, backPropGradients = None,  camerasUsed = None, nodeName=''):

        self.cameraFilePath = cameraFilePath

        self.numberOfBatches = numberOfBatches

        self.numberOfMarkers = numberOfMarkers

        self.preds2D = preds2D

        self.predsConf = predsConf

        self.markers3D = markers3D

        self.backPropGradients = backPropGradients

        self.camerasUsed = camerasUsed

        self.nodeName = nodeName

        self.PerspectiveNPointOperator = None

        if(cameraFilePath != ''
                and numberOfMarkers != 0
                and numberOfBatches != 0
                and preds2D is not None
                and predsConf is not None
                and markers3D is not None
                and backPropGradients is not None
                and camerasUsed is not None
                and nodeName != ''):

            self.PerspectiveNPointOperator = customOperators.perspective_n_point_gpu(preds2D,
                                                                                     predsConf,
                                                                                     markers3D,
                                                                                     camera_file_path = cameraFilePath,
                                                                                     number_of_batches_pnp = numberOfBatches,
                                                                                     number_of_markers_pnp = numberOfMarkers,
                                                                                     backprop_gradient = backPropGradients,
                                                                                     cameras_used = camerasUsed,
                                                                                     name = nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the perspective N point operator!')

    # return the node for building a tf graph
    def getNode(self):

        return self.PerspectiveNPointOperator[0]

########################################################################################################################
# Register Gradient
########################################################################################################################

@ops.RegisterGradient("PerspectiveNPointGpu")
def perspective_n_point_gpu_grad(op, grad, grad1, grad2):

    predictions2DZeroGrad = tf.zeros(tf.shape(op.inputs[0]), tf.float32)

    predictionsConfidenceZeroGrad = tf.zeros(tf.shape(op.inputs[1]), tf.float32)

    markerGlobalSpaceGrad = customOperators.perspective_n_point_gpu_grad(grad,
                                                                         op.outputs[1],
                                                                         op.outputs[2],
                                                                         op.get_attr('camera_file_path'),
                                                                         op.get_attr('number_of_batches_pnp'),
                                                                         op.get_attr('number_of_markers_pnp'),
                                                                         op.get_attr('backprop_gradient'))

    return predictions2DZeroGrad, predictionsConfidenceZeroGrad, markerGlobalSpaceGrad

########################################################################################################################
#
########################################################################################################################
