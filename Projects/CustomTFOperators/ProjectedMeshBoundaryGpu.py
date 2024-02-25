
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
# ProjectedMeshBoundaryGpu class
########################################################################################################################

class ProjectedMeshBoundaryGpu:

    def __init__(self,
                 pointsGlobalSpace=None,
                 extrinsics=None,
                 intrinsics=None,
                 meshFilePath = '',
                 useGapDetections = False,
                 numberOfBatches = None,
                 numberOfCameras = None,
                 renderU = None,
                 renderV=None,
                 nodeName=''):

        self.pointsGlobalSpace = pointsGlobalSpace
        self.extrinsics =extrinsics
        self.intrinsics =intrinsics
        self.meshFilePath = meshFilePath
        self.useGapDetections = useGapDetections
        self.numberOfBatches = numberOfBatches
        self.numberOfCameras = numberOfCameras
        self.renderU = renderU
        self.renderV = renderV
        self.nodeName = nodeName


        self.projectedMeshBoundaryGpuOperator = None

        if( meshFilePath != ''
                and pointsGlobalSpace is not None
                and numberOfBatches is not None
                and extrinsics is not None
                and intrinsics is not None
                and nodeName != ''):

            self.projectedMeshBoundaryGpuOperator = customOperators.projected_mesh_boundary_gpu(points_global_space = pointsGlobalSpace,
                                                                                                extrinsics = extrinsics,
                                                                                                intrinsics = intrinsics,
                                                                                                mesh_file_path_boundary_check = meshFilePath,
                                                                                                use_gap_detection = useGapDetections,
                                                                                                number_batches =numberOfBatches,
                                                                                                number_cameras = numberOfCameras,
                                                                                                render_u = renderU,
                                                                                                render_v = renderV,
                                                                                                name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the projected mesh boundary operator!')

    # return the node for building a tf graph
    def getNode(self):

        return self.projectedMeshBoundaryGpuOperator

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("ProjectedMeshBoundaryGpu")
def projected_mesh_boundary_gpu_grad(op, grad):

    # determine the zero gradient stuff
    pointsGlobalSpaceZeroGrad = tf.zeros(tf.shape(op.inputs[0]), tf.float32)

    return pointsGlobalSpaceZeroGrad, tf.zeros(tf.shape(op.inputs[1]), tf.float32),tf.zeros(tf.shape(op.inputs[2]), tf.float32)

########################################################################################################################
#
########################################################################################################################
