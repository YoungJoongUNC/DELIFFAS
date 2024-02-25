
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
# MultiViewSilhouetteLossGpu class
########################################################################################################################

class MultiViewSilhouetteLossGpu:

    def __init__(self,
                 pointsImageSpace = None,
                 normalsImageSpace = None,
                 boundaryVertices = None ,
                 dtImages=None,
                 cropsMultiView=None,
                 nodeName=''):

        self.pointsImageSpace = pointsImageSpace
        self.normalsImageSpace = normalsImageSpace
        self.boundaryVertices = boundaryVertices
        self.dtImages = dtImages
        self.cropsMultiView = cropsMultiView
        self.nodeName = nodeName

        self.multiViewSilhouetteLossGpuOperator = None

        if(pointsImageSpace is not None and normalsImageSpace is not None and boundaryVertices is not None and dtImages is not None and cropsMultiView is not None and nodeName != ''):

            self.multiViewSilhouetteLossGpuOperator = customOperators.multi_view_silhouette_loss_gpu(pointsImageSpace,
                                                                                                   normalsImageSpace,
                                                                                                   boundaryVertices,
                                                                                                   dtImages,
                                                                                                   cropsMultiView,
                                                                                                   name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the multi view silhouette loss operator!')

    # return the node for building a tf graph
    def getModelToData(self):
        return self.multiViewSilhouetteLossGpuOperator[0]

    # return the node for building a tf graph
    def getDataToModel(self):
        return self.multiViewSilhouetteLossGpuOperator[1]

########################################################################################################################
# CustomTFOperators gradients
########################################################################################################################

@ops.RegisterGradient("MultiViewSilhouetteLossGpu")
def multi_view_silhouette_loss_gpu_grad(op, gradientDTLoss, gradientDTLoss1, gradImage,gradClosestVertex):

    #take the outputs from the custom gradient layer
    outputGrads = customOperators.multi_view_silhouette_loss_gpu_grad(gradientDTLoss,gradientDTLoss1, op.outputs[2], op.outputs[3], op.inputs[4])

    #determine the zero gradient stuff
    normalsImageSpaceZeroGradient = tf.zeros(tf.shape(op.inputs[1]), tf.float32)
    isBoundaryZeroGradient = tf.zeros(tf.shape(op.inputs[2]), tf.bool)
    distanceTransformZeroGradient = tf.zeros(tf.shape(op.inputs[3]), tf.float32)
    multiViewCropsZeroGradient = tf.zeros(tf.shape(op.inputs[4]), tf.float32)

    return outputGrads, normalsImageSpaceZeroGradient, isBoundaryZeroGradient, distanceTransformZeroGradient, multiViewCropsZeroGradient

########################################################################################################################
#
########################################################################################################################