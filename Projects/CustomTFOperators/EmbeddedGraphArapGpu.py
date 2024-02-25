
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
# EmbeddedGraphArapGpu class
########################################################################################################################

class EmbeddedGraphArapGpu:

    def __init__(self,
                 characterFilePath = '',
                 graphFilePath = '',
                 numberOfBatches = 0,
                 maxNumberOfNodeConnections =0,
                 T = None,
                 A = None,
                 refinement = None,
                 nodeName=''):

        self.characterFilePath = characterFilePath
        self.graphFilePath = graphFilePath
        self.numberOfBatches = numberOfBatches
        self.maxNumberOfNodeConnections = maxNumberOfNodeConnections
        self.T = T
        self.A = A
        self.refinement = refinement
        self.nodeName = nodeName

        self.EmbeddedGraphArapGpuOperator = None

        if(characterFilePath != '' and graphFilePath != '' and numberOfBatches != 0 and T is not None and A is not None and nodeName != '' and maxNumberOfNodeConnections != 0 and refinement is not None):

            self.EmbeddedGraphArapGpuOperator = customOperators.embedded_graph_arap_gpu(T,
                                                                                A,
                                                                                character_file_path_eg = characterFilePath,
                                                                                graph_file_path = graphFilePath,
                                                                                number_of_batches_eg= numberOfBatches,
                                                                                max_number_of_node_connections=maxNumberOfNodeConnections,
                                                                                refinement=refinement)

        else:

            raise ValueError('Invalid argument during the construction of the embedded graph arap operator!')

    # return the node for building a tf graph
    def getNode(self):

        return [self.EmbeddedGraphArapGpuOperator[0],self.EmbeddedGraphArapGpuOperator[1]]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("EmbeddedGraphArapGpu")
def embedded_graph_arap_gpu_grad(op, nodeArapLoss, connectionWeightGrad, AGrad):

    nodesTGrad, nodesRGrad = customOperators.embedded_graph_arap_gpu_grad(nodeArapLoss,
                                                                               op.outputs[2],
                                                                               character_file_path_eg_grad = op.get_attr('character_file_path_eg'),
                                                                               graph_file_path_grad =op.get_attr('graph_file_path'),
                                                                               number_of_batches_eg_grad=op.get_attr('number_of_batches_eg'),
                                                                               max_number_of_node_connections_grad =op.get_attr('max_number_of_node_connections'),
                                                                               refinement_grad = op.get_attr('refinement') )

    return nodesTGrad, nodesRGrad

########################################################################################################################
#
########################################################################################################################
