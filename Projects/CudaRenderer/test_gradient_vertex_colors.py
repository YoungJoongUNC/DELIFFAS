
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
from AdditionalUtils import CameraReader, OBJReader
import numpy as np
import os

#######################################

print('Create output directory ...')
os.makedirs('output_test_gradient_vertex_colors', exist_ok=True)

#######################################

print('Setup data ...')

numberOfBatches = 3
renderResolutionU = 1024
renderResolutionV = 1024

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
cv.imwrite('output_test_gradient_vertex_colors/target.jpg', targetCV*255.0)

#######################################

print('Optimize vertex colors ...')

VertexColor_rnd = tf.Variable(tf.zeros(VertexColorConst.shape))

opt = tf.keras.optimizers.SGD(learning_rate=100.0)

for i in range(250):

    with tf.GradientTape() as tape:
        tape.watch(VertexColor_rnd)

        # render the current result
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
            vertexColor_input           = VertexColor_rnd,
            texture_input               = VertexTextureConst,
            shCoeff_input               = SHCConst,
            targetImage_input           = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3]),
            extrinsics_input            = [cameraReader.extrinsics, cameraReader.extrinsics, cameraReader.extrinsics],
            intrinsics_input            = [cameraReader.intrinsics, cameraReader.intrinsics, cameraReader.intrinsics],
            nodeName                    = 'train')
        output = renderer.getRenderBufferTF()

        # loss
        Loss1 = (output-target) * (output-target)
        Loss = tf.reduce_sum(Loss1) / (float(cameraReader.numberOfCameras) * float(objreader.numberOfVertices))

    #apply gradient
    Color_Grad = tape.gradient(Loss,VertexColor_rnd)
    opt.apply_gradients(zip([Color_Grad],[VertexColor_rnd]))

    # print loss
    print('Iteration:', i, 'Loss:', Loss.numpy())

    # output images
    if i % 10 == 0:
        outputCV = renderer.getRenderBufferOpenCV(0, 0)
        cv.imwrite('output_test_gradient_vertex_colors/iter_' + str(i) + '.jpg', outputCV * 255.0)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    