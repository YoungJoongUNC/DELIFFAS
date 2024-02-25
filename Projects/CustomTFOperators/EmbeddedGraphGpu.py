
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
# EmbeddedGraphGpu class
########################################################################################################################

class EmbeddedGraphGpu:

    def __init__(self,
                 characterFilePath = '',
                 graphFilePath = '',
                 numberOfBatches = 0,
                 deltaT = None,
                 deltaR = None,
                 skinnedT=None,
                 skinnedR=None,
                 displacements=None,
                 refinement = None,
                 nodeName=''):

        self.characterFilePath = characterFilePath
        self.graphFilePath = graphFilePath
        self.numberOfBatches = numberOfBatches
        self.deltaT = deltaT
        self.deltaR = deltaR
        self.skinnedT = skinnedT
        self.skinnedR = skinnedR
        self.displacements = displacements
        self.refinement = refinement
        self.nodeName = nodeName

        self.embeddedGraphGpuOperator = None

        if(characterFilePath != ''
                and graphFilePath != ''
                and numberOfBatches != 0
                and deltaT is not None
                and deltaR is not None
                and skinnedT is not None
                and skinnedR is not None
                and displacements is not None
                and nodeName != ''and refinement is not None):

            self.embeddedGraphGpuOperator = customOperators.embedded_graph_gpu( deltaT,
                                                                                deltaR,
                                                                                skinnedT,
                                                                                skinnedR,
                                                                                displacements,
                                                                                character_file_path_eg = characterFilePath,
                                                                                graph_file_path = graphFilePath,
                                                                                number_of_batches_eg= numberOfBatches,
                                                                                refinement=refinement)

        else:

            raise ValueError('Invalid argument during the construction of the embedded graph operator!')

    # return the node for building a tf graph
    #outputs deformed vertices | deformed normals | deformed markers | deformed graph nodes
    def getNode(self):

        return [self.embeddedGraphGpuOperator[0],self.embeddedGraphGpuOperator[1], self.embeddedGraphGpuOperator[2],self.embeddedGraphGpuOperator[3]]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("EmbeddedGraphGpu")
def embedded_graph_gpu_grad(op, deformedVerticesGrad, deformedNormalsGrad, deformedMarkersGrad, deformedGraphNodes, deltaAGrad, skinnedAGrad):

    nodesTGrad, nodesRGrad, nodesSkinnedTGrad, nodesSkinnedRGrad, displacementsGrad = customOperators.embedded_graph_gpu_grad(deformedVerticesGrad,
                                                                               deformedMarkersGrad,
                                                                               op.outputs[4],
                                                                               op.outputs[5],
                                                                               character_file_path_eg_grad = op.get_attr('character_file_path_eg'),
                                                                               graph_file_path_grad = op.get_attr('graph_file_path'),
                                                                               refinement_grad = op.get_attr('refinement')
                                                                               )

    return nodesTGrad, nodesRGrad, nodesSkinnedTGrad, nodesSkinnedRGrad, displacementsGrad

########################################################################################################################
#
########################################################################################################################
