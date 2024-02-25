
########################################################################################################################
# Imports
########################################################################################################################

import sys
sys.path.append("../")

import data.test_SH_tensor as test_SH_tensor
import CudaRendererGPU as CudaRenderer
import cv2 as cv
import numpy as np
from AdditionalUtils import CameraReader, OBJReader
import tensorflow as tf
import os

#######################################

print('Create output directory ...')
os.makedirs('output_test_render', exist_ok=True)

#######################################

print('Setup data ...')

# here we artificially added 2 batches although each contains the same data.
# it is just do demonstrate the shaping of data
numberOfBatches     = 2
renderResolutionU   = 256
renderResolutionV   = 256

cameraReader = CameraReader.CameraReader('data/cameras.calibration', renderResolutionU, renderResolutionV)
objreader = OBJReader.OBJReader('data/cone.obj', segWeightsFlag = False)

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

VertexPosConst      = tf.constant(inputVertexPositions, dtype=tf.float32)
VertexColorConst    = tf.constant(inputVertexColors, dtype=tf.float32)
VertexTextureConst  = tf.constant(inputTexture, dtype=tf.float32)
SHCConst            = tf.constant(inputSHCoeff, dtype=tf.float32)
camExtrinsics       = tf.constant([cameraReader.extrinsics,cameraReader.extrinsics])
camIntrinsics       = tf.constant([cameraReader.intrinsics,cameraReader.intrinsics])
targetImage         = tf.zeros( [numberOfBatches, cameraReader.numberOfCameras, renderResolutionV, renderResolutionU, 3])

#######################################

print('---------- Input shapes ------------')

tf.print('Vertex positions')
tf.print(tf.shape(VertexPosConst))

tf.print('Vertex color')
tf.print(tf.shape(VertexColorConst))

tf.print('Texture')
tf.print(tf.shape(VertexTextureConst))

tf.print('Spherical harmonics')
tf.print(tf.shape(SHCConst))

tf.print('Camera extrinsics')
tf.print(tf.shape(camExtrinsics))

tf.print('Camera intrinsics')
tf.print(tf.shape(camIntrinsics))

tf.print('Target image')
tf.print(tf.shape(targetImage))

print('------------------------------------')

print('Define render function ...')
def renderModes(albedoMode, shadingMode, normalMap = False):
    renderer = CudaRenderer.CudaRendererGpu(
                                            faces_attr                  = objreader.facesVertexId,
                                            texCoords_attr              = objreader.textureCoordinates,
                                            numberOfVertices_attr       = len(objreader.vertexCoordinates),
                                            numberOfCameras_attr        = cameraReader.numberOfCameras,
                                            renderResolutionU_attr      = renderResolutionU,
                                            renderResolutionV_attr      = renderResolutionV,
                                            albedoMode_attr             = albedoMode,
                                            shadingMode_attr            = shadingMode,
                                            image_filter_size_attr      = 1,
                                            texture_filter_size_attr    = 1,
                                            compute_normal_map_attr     = normalMap,

                                            vertexPos_input             = VertexPosConst,
                                            vertexColor_input           = VertexColorConst,
                                            texture_input               = VertexTextureConst,
                                            shCoeff_input               = SHCConst,
                                            targetImage_input           = targetImage,
                                            extrinsics_input            = camExtrinsics,
                                            intrinsics_input            = camIntrinsics,

                                            nodeName                    = 'test')
    return renderer

#######################################

print('render ...')

# output images for batch 1 and camera 0
batch  = 1
camera = 0

# albedo modes which support shaded and shadeless
albedoSet1 = ['vertexColor', 'textured']
shadingSet1 = ['shaded', 'shadeless']

# albedo modes which only support shadeless (those are also non-differentiable)
albedoSet2 = ['normal', 'foregroundMask', 'lighting', 'depth', 'position', 'uv']
shadingSet2 = ['shadeless']

# render first set of options
for albedo in albedoSet1:
    for shading in shadingSet1:
        renderer = renderModes(albedo, shading)
        render = renderer.getRenderBufferOpenCV(batch,camera)
        savePath = 'output_test_render/test_'+shading+'_'+albedo+'.jpg'
        print('Safe to ', savePath)
        cv.imwrite(savePath, render * 255.0)

# render second set of options
for albedo in albedoSet2:
    for shading in shadingSet2:
        renderer = renderModes(albedo, shading)
        render = renderer.getRenderBufferOpenCV(batch,camera)
        savePath = 'output_test_render/test_'+shading+'_'+albedo+'.jpg'
        print('Safe to ', savePath)
        cv.imwrite(savePath, render * 255.0)

# render the global normals into a texture
renderer = renderModes('vertexColor', 'shaded', True)
render = renderer.getNormalMapOpenCV(batch)
savePath = 'output_test_render/normalMap.jpg'
print('Safe to ', savePath)
cv.imwrite(savePath, render * 255.0)

#done
print('done.')

#######################################