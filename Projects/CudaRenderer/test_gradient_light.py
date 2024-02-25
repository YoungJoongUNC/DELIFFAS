
########################################################################################################################
# Imports
########################################################################################################################

import sys
sys.path.append("../")

import tensorflow as tf
import data.test_mesh_tensor as test_mesh_tensor
import data.test_SH_tensor as test_SH_tensor
import CudaRendererGPU as CudaRenderer
import cv2 as cv
import numpy as np
from AdditionalUtils import CameraReader, OBJReader
import os

#######################################

print('Create output directory ...')
os.makedirs('output_test_gradient_light', exist_ok=True)

#######################################

print('Setup data ...')

numberOfBatches = 3
renderResolutionU = 1024
renderResolutionV = 1024

cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)
objreader = OBJReader.OBJReader('data/magdalena.obj', segWeightsFlag = False)

inputVertexPositions = test_mesh_tensor.getGTMesh()
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

VertexPosConst          = tf.constant(inputVertexPositions,     dtype=tf.float32)
VertexColorConst        = tf.constant(inputVertexColors,        dtype=tf.float32)
VertexTextureConst      = tf.constant(inputTexture,             dtype=tf.float32)
SHCConst                = tf.constant(inputSHCoeff,             dtype=tf.float32)

#######################################

print('Create GT data ...')

rendererTarget = CudaRenderer.CudaRendererGpu(
                                    faces_attr                   = objreader.facesVertexId,
                                    texCoords_attr               = objreader.textureCoordinates,
                                    numberOfVertices_attr        = len(objreader.vertexCoordinates),
                                    numberOfCameras_attr         = cameraReader.numberOfCameras,
                                    renderResolutionU_attr       = renderResolutionU,
                                    renderResolutionV_attr       = renderResolutionV,
                                    albedoMode_attr              = 'vertexColor',
                                    shadingMode_attr             = 'shaded',

                                    vertexPos_input              = VertexPosConst,
                                    vertexColor_input            = VertexColorConst,
                                    texture_input                = VertexTextureConst,
                                    shCoeff_input                = SHCConst,
                                    targetImage_input            = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                                    extrinsics_input             = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                                    intrinsics_input             = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                                    nodeName                     = 'target')

target = rendererTarget.getRenderBufferTF()

targetCV = rendererTarget.getRenderBufferOpenCV(0, 0)
cv.imwrite('output_test_gradient_light/target.jpg', targetCV*255.0)

#######################################

print('Optimize lighting ...')

SHC_rnd = tf.Variable(SHCConst+tf.random.uniform([1,1, 27],0, 0.5) )

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(300):
    with tf.GradientTape() as g:
        g.watch(SHC_rnd)

        # render the current iteration
        renderer = CudaRenderer.CudaRendererGpu(
            faces_attr                  = objreader.facesVertexId,
            texCoords_attr              = objreader.textureCoordinates,
            numberOfVertices_attr       = len(objreader.vertexCoordinates),
            numberOfCameras_attr        = cameraReader.numberOfCameras,
            renderResolutionU_attr      = renderResolutionU,
            renderResolutionV_attr      = renderResolutionV,
            albedoMode_attr             = 'vertexColor',
            shadingMode_attr            = 'shaded',
            vertexPos_input             = VertexPosConst,
            vertexColor_input           = VertexColorConst,
            texture_input               = VertexTextureConst,
            shCoeff_input               = SHC_rnd,
            targetImage_input           = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
            extrinsics_input            = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
            intrinsics_input            = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
            nodeName                    = 'train')
        output = renderer.getRenderBufferTF()

        # define the loss
        Loss=tf.nn.l2_loss(target-output)

    # apply gradient
    SHC_Grad=g.gradient(Loss,SHC_rnd)
    opt.apply_gradients(zip([SHC_Grad], [SHC_rnd]))

    # print loss
    print('Iteration:', i, 'Loss:', Loss.numpy())

    # output images
    if i % 10 == 0:
        outputCV = renderer.getRenderBufferOpenCV(0, 0)
        cv.imwrite('output_test_gradient_light/iter_' + str(i) + '.jpg', outputCV*255.0)