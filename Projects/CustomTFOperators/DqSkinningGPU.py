
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
# DqSkinningGpu class
########################################################################################################################

class DqSkinningGpu:

    def __init__(self, characterFilePath = '', dofs = None, skinningWeights = None, miniBatchSize = 0, nodeName=''):

        self.characterFilePath = characterFilePath

        self.miniBatchSize = miniBatchSize

        self.nodeName = nodeName

        self.dqSkinningGpuOperator = None

        if(characterFilePath != '' and dofs is not None and miniBatchSize != 0  and nodeName != '' and skinningWeights is not None):

            self.dqSkinningGpuOperator = customOperators.dq_skinning_gpu(dofs=dofs,
                                                                         skinning_weights=skinningWeights,
                                                                         character_file_path_skinning=characterFilePath,
                                                                         mini_batch_size_skinning=miniBatchSize,
                                                                         name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the dq skinning operator!')

    # return the node for building a tf graph
    def getNode(self):

        return [self.dqSkinningGpuOperator[0],self.dqSkinningGpuOperator[1]]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("DqSkinningGpu")
def dq_skinning_gpu_grad(op, grad1, grad2, grad3, grad4, grad5, grad6):

    dofsGrad, skinningWeightsGrad = customOperators.dq_skinning_gpu_grad(grad1,
                                                    op.outputs[0],
                                                    op.outputs[2],
                                                    op.outputs[3],
                                                    op.outputs[4],
                                                    op.outputs[5],
                                                    character_file_path_skinning_grad=op.get_attr('character_file_path_skinning'),
                                                    mini_batch_size_skinning_grad=op.get_attr('mini_batch_size_skinning'))

    return dofsGrad, skinningWeightsGrad

########################################################################################################################
#
########################################################################################################################
