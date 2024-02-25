
########################################################################################################################
# Imports
########################################################################################################################

import sys
sys.path.append("../")

import tensorflow as tf
import data.test_SH_tensor as test_SH_tensor
import CudaRendererGPU as CudaRenderer
from AdditionalUtils import CameraReader, OBJReader
import cv2 as cv
import numpy as np
import os

#######################################

print('Create output directory ...')
os.makedirs('output_test_gradient_vertex_positions', exist_ok=True)

#######################################

print('Setup data ...')

numberOfBatches             = 1
renderResolutionU           = 256
renderResolutionV           = 256
shadingMode_attr            = 'shadeless'
albedoMode_attr             = 'vertexColor'

objreader = OBJReader.OBJReader('data/cone.obj', segWeightsFlag=False)
cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)

inputVertexPositions = objreader.vertexCoordinates
inputVertexPositions = np.asarray(inputVertexPositions)
inputVertexPositions = inputVertexPositions.reshape([1, objreader.numberOfVertices, 3])
inputVertexPositions = np.tile(inputVertexPositions, (numberOfBatches, 1, 1))

inputVertexColors = objreader.vertexColors
inputVertexColors = np.asarray(inputVertexColors)
inputVertexColors = inputVertexColors.reshape([1, objreader.numberOfVertices, 3])
inputVertexColors = np.tile(inputVertexColors, (numberOfBatches, 1, 1))

inputTexture = objreader.textureMap
inputTexture = np.asarray(inputTexture)
inputTexture = inputTexture.reshape([1, objreader.texHeight, objreader.texWidth, 3])
inputTexture = np.tile(inputTexture, (numberOfBatches, 1, 1, 1))

inputSHCoeff = test_SH_tensor.getSHCoeff(numberOfBatches, cameraReader.numberOfCameras)

VertexPosConst = tf.constant(inputVertexPositions, dtype=tf.float32)
VertexColorConst = tf.constant(inputVertexColors, dtype=tf.float32)
VertexTextureConst = tf.constant(inputTexture, dtype=tf.float32)
SHCConst = tf.constant(inputSHCoeff, dtype=tf.float32)

#######################################

print('Create GT data ...')

rendererTarget = CudaRenderer.CudaRendererGpu(
                                faces_attr                   = objreader.facesVertexId,
                                texCoords_attr               = objreader.textureCoordinates,
                                numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                numberOfCameras_attr         = cameraReader.numberOfCameras,
                                renderResolutionU_attr       = renderResolutionU,
                                renderResolutionV_attr       = renderResolutionV,
                                albedoMode_attr              = albedoMode_attr,
                                shadingMode_attr             = shadingMode_attr,
                                vertexPos_input              = VertexPosConst,
                                vertexColor_input            = VertexColorConst,
                                texture_input                = VertexTextureConst,
                                shCoeff_input                = SHCConst,
                                targetImage_input            = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                                extrinsics_input             = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                                intrinsics_input             = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                                nodeName                     = 'target'
                            )
target = rendererTarget.getRenderBufferTF()
targetCV = rendererTarget.getRenderBufferOpenCV(0, 0)
cv.imwrite('output_test_gradient_vertex_positions/target.jpg', targetCV*255.0)

#######################################

print('Optimize vertex positions ...')

offset = tf.constant([0.0, 0.0, 0.0])
offset = tf.reshape(offset, [1, 1, 3])
offset = tf.tile(offset, [numberOfBatches, objreader.numberOfVertices, 1])
VertexPosition_rnd = tf.Variable(VertexPosConst * 0.6 , dtype=tf.float32)

opt = tf.keras.optimizers.SGD(learning_rate=1.0)

for i in range(1000):

    with tf.GradientTape() as tape:
        tape.watch(VertexPosition_rnd)

        # render image
        renderer = CudaRenderer.CudaRendererGpu(
            faces_attr                  = objreader.facesVertexId,
            texCoords_attr              = objreader.textureCoordinates,
            numberOfVertices_attr       = len(objreader.vertexCoordinates),
            numberOfCameras_attr        = cameraReader.numberOfCameras,
            renderResolutionU_attr      = renderResolutionU,
            renderResolutionV_attr      = renderResolutionV,
            albedoMode_attr             = albedoMode_attr,
            shadingMode_attr            = shadingMode_attr,
            vertexPos_input             = VertexPosition_rnd,
            vertexColor_input           = VertexColorConst,
            texture_input               = VertexTextureConst,
            shCoeff_input               = SHCConst,
            targetImage_input           = target,
            extrinsics_input            = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
            intrinsics_input            = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
            nodeName                    = 'train'
        )
        output = renderer.getRenderBufferTF()

        # image loss
        difference = (output - target)
        foregroundPixels = tf.math.count_nonzero(difference)
        loss = 1000000.0  * tf.nn.l2_loss((difference) ) / (float(foregroundPixels))

    # apply gradient
    Color_Grad = tape.gradient(loss, VertexPosition_rnd)
    opt.apply_gradients(zip([Color_Grad], [VertexPosition_rnd]))

    # print loss
    print('Iteration:', i, 'Loss:', loss.numpy())

    # output images
    if i % 10 == 0:
        outputCV = renderer.getRenderBufferOpenCV(0, 0)
        cv.imwrite('output_test_gradient_vertex_positions/iter_' + str(i) + '.jpg', outputCV * 255.0)