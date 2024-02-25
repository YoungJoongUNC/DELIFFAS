
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
os.makedirs('output_test_gradient_texture', exist_ok=True)

#######################################

print('Setup data ...')

numberOfBatches = 1
renderResolutionU = 1024
renderResolutionV = 1024
imageFilterSize = 1
textureFilterSize = 1

cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)
objreader = OBJReader.OBJReader('data/magdalena.obj', segWeightsFlag=False)

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
                                    albedoMode_attr              = 'textured',
                                    shadingMode_attr             = 'shadeless',
                                    image_filter_size_attr       = imageFilterSize,
                                    texture_filter_size_attr     = textureFilterSize,
                                    vertexPos_input              = VertexPosConst,
                                    vertexColor_input            = VertexColorConst,
                                    texture_input                = VertexTextureConst,
                                    shCoeff_input                = SHCConst,
                                    targetImage_input            = tf.zeros([numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
                                    extrinsics_input             = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
                                    intrinsics_input             = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
                                    nodeName                     = 'target')

target = rendererTarget.getRenderBufferTF()
targetCV = rendererTarget.getRenderBufferOpenCV(0, 0)
cv.imwrite('output_test_gradient_texture/target.jpg', targetCV*255.0)

#######################################

print('Optimize texture ...')

VertexTextureRand = tf.Variable(tf.ones(VertexTextureConst.shape))

opt = tf.keras.optimizers.SGD(learning_rate=100.0)

for i in range(250):

    with tf.GradientTape() as tape:
        tape.watch(VertexTextureRand)

        # render the current result
        renderer = CudaRenderer.CudaRendererGpu(
            faces_attr                  = objreader.facesVertexId,
            texCoords_attr              = objreader.textureCoordinates,
            numberOfVertices_attr       = len(objreader.vertexCoordinates),
            numberOfCameras_attr        = cameraReader.numberOfCameras,
            renderResolutionU_attr      = renderResolutionU,
            renderResolutionV_attr      = renderResolutionV,
            albedoMode_attr             = 'textured',
            shadingMode_attr            = 'shadeless',
            image_filter_size_attr      = imageFilterSize,
            texture_filter_size_attr    = textureFilterSize,
            vertexPos_input             = VertexPosConst,
            vertexColor_input           = VertexColorConst,
            texture_input               = VertexTextureRand,
            shCoeff_input               = SHCConst,
            targetImage_input           = target,
            extrinsics_input            = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
            intrinsics_input            = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
            nodeName                    = 'render')
        output = renderer.getRenderBufferTF()

        # loss
        Loss1 = (output-target) * (output-target)
        Loss = tf.reduce_sum(Loss1) / (float(cameraReader.numberOfCameras) * float(objreader.numberOfVertices))

    # apply gradient
    Color_Grad = tape.gradient(Loss,VertexTextureRand)
    opt.apply_gradients(zip([Color_Grad],[VertexTextureRand]))

    # print loss
    print('Iteration:', i, 'Loss:', Loss.numpy())

    # output images
    if i % 10 == 0:
        outputCV = renderer.getRenderBufferOpenCV(0, 0)
        cv.imwrite('output_test_gradient_texture/iter_' + str(i) + '.jpg', outputCV * 255.0)

# safe textures
print('Safe optimized and target texture ...')

textureTarget = cv.cvtColor(VertexTextureConst[0,:,:,:].numpy(), cv.COLOR_RGB2BGR)
cv.imwrite('output_test_gradient_texture/texture_target.jpg', textureTarget * 255.0)

textureCV = cv.cvtColor(VertexTextureRand[0,:,:,:].numpy(), cv.COLOR_RGB2BGR)
cv.imwrite('output_test_gradient_texture/texture_optimized.jpg', textureCV * 255.0)















    