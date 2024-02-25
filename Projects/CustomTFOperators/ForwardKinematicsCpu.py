
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
# ForwardKinematicsCpu class
########################################################################################################################

class ForwardKinematicsCpu:

    def __init__(self, skeletonFilePath='', dofs=None, numberOfBatches =0, numberOfThreads=0, nodeName=''):

        self.skeletonFilePath = skeletonFilePath
        self.dofs = dofs
        self.nodeName = nodeName
        self.numberOfBatches = numberOfBatches
        self.forwardKinematicsOperator = None

        if(skeletonFilePath != '' and dofs is not None and numberOfBatches != 0 and numberOfThreads != 0and nodeName != ''):

            self.forwardKinematicsOperator = customOperators.forward_kinematics_cpu(dofs,
                                                                                    skeleton_file_path=skeletonFilePath,
                                                                                    number_of_batches_fk=numberOfBatches,
                                                                                    number_of_threads_fk=numberOfThreads,
                                                                                    name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the forward kinematics operator!')

    # return the node for building a tf graph
    def getNode(self):

        return self.forwardKinematicsOperator[0]

########################################################################################################################
# Register Gradient
########################################################################################################################

@ops.RegisterGradient("ForwardKinematicsCpu")
def forward_kinematics_cpu_grad(op, gradMarkerPos, grad1, grad2, grad3):

            dofGradient = customOperators.forward_kinematics_cpu_grad(gradMarkerPos,
                                                                      op.outputs[1],
                                                                      op.outputs[2],
                                                                      op.outputs[3],
                                                                      op.get_attr('skeleton_file_path'),
                                                                      op.get_attr('number_of_batches_fk'),
                                                                      op.get_attr('number_of_threads_fk'))

            return dofGradient

########################################################################################################################
#
########################################################################################################################
