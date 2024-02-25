
#############################################################################################
#############################################################################################

import sys
import pdb
sys.path.append("../")
sys.path.append("../DeepDynamicCharacters")

#############################################################################################
#############################################################################################

import CudaRenderer.CudaRendererGPU as CudaRenderer
import numpy as np
import tensorflow as tf

#############################################################################################
#############################################################################################

def getSHCoeff(numBatches, numCams):
    shCoeff = np.array([0.7, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.7, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.7, 0, 0, -0.5, 0, 0, 0, 0, 0])
    shCoeff = shCoeff.reshape([1, 1, 27])
    shCoeff = np.tile(shCoeff,(numBatches , numCams,1))
    return shCoeff

#############################################################################################
#############################################################################################

def renderDepth(cameraId, objreader, cameraReader, meshInstance, jellyOffset):

    # vertex color
    inputVertexColors = np.asarray(objreader.vertexColors)
    inputVertexColors = inputVertexColors.reshape([1, objreader.numberOfVertices, 3])
    VertexColorConst = tf.constant(inputVertexColors, dtype=tf.float32)

    # texture
    inputTexture = np.asarray(objreader.textureMap)
    inputTexture = inputTexture.reshape([1, objreader.texHeight, objreader.texWidth, 3])
    VertexTextureConst = tf.constant(inputTexture, dtype=tf.float32)

    # light
    inputSHCoeff = getSHCoeff(1, 1)
    SHCConst = tf.constant(inputSHCoeff, dtype=tf.float32)

    renderer = render(tf.constant(cameraId), objreader, cameraReader, tf.constant(meshInstance, dtype=tf.float32), VertexColorConst, VertexTextureConst, SHCConst)


    if jellyOffset != 0.0:

        renderImg = renderer.getRenderBufferTF() * renderer.getModelMaskTF() + (1.0 - renderer.getModelMaskTF()) * 100000.0
        renderImg = renderImg[0, :, :, :, 0:1]
        renderImgEroded = tf.nn.erosion2d(value=renderImg, filters=tf.zeros([9, 9, 1]), strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1], name=None)

        mask = tf.less(renderImgEroded, 100000.0)
        mask = tf.cast(mask, tf.float32)
        renderImgEroded = renderImgEroded * mask

        renderImg = renderer.getRenderBufferTF() * renderer.getModelMaskTF()
        renderImg = renderImg[0, :, :, :, 0:1]
        renderImgDilated = tf.nn.dilation2d(input=renderImg, filters=tf.zeros([9, 9, 1]), strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1], name=None)

        near = renderImgEroded * 0.001 - mask * jellyOffset
        near = (near.numpy())[0, :, :, 0:1]
        near = np.array(near)
        far = renderImgDilated * 0.001 + mask * jellyOffset
        far = (far.numpy())[0, :, :, 0:1]
        far = np.array(far)

    elif jellyOffset == 0.0:
        renderImg = renderer.getRenderBufferTF() * renderer.getModelMaskTF() + (1.0 - renderer.getModelMaskTF()) * 100000.0

        renderImgEroded = renderImg[0, :, :, :, 0:1]

        mask = tf.less(renderImgEroded, 100000.0)
        mask = tf.cast(mask, tf.float32)
        renderImgEroded = renderImgEroded * mask

        renderImg = renderer.getRenderBufferTF() * renderer.getModelMaskTF()

        renderImgDilated = renderImg[0, :, :, :, 0:1]

        near = renderImgEroded * 0.001
        near = (near.numpy())[0, :, :, 0:1]
        near = np.array(near)
        far = renderImgDilated * 0.001
        far = (far.numpy())[0, :, :, 0:1]
        far = np.array(far)
    depthImage = np.concatenate([near, far], 2)

    return depthImage

#

#############################################################################################
#############################################################################################

def render(c, objreader,cameraReader, vP, vC, T, SH):

    camExtrinsics = tf.reshape(cameraReader.extrinsics, [cameraReader.numberOfCameras, 12])
    camExtrinsics = tf.gather(camExtrinsics, [c])
    camExtrinsics = tf.reshape(camExtrinsics, [1, 12])

    camIntrinsics = tf.reshape(cameraReader.intrinsics, [cameraReader.numberOfCameras, 9])
    camIntrinsics = tf.gather(camIntrinsics, [c])
    camIntrinsics = tf.reshape(camIntrinsics, [1, 9])

    renderer = CudaRenderer.CudaRendererGpu(
                                            faces_attr                  = objreader.facesVertexId,
                                            texCoords_attr              = objreader.textureCoordinates,
                                            numberOfVertices_attr       = len(objreader.vertexCoordinates),
                                            numberOfCameras_attr        = 1,
                                            renderResolutionU_attr      = cameraReader.width,
                                            renderResolutionV_attr      = cameraReader.height,
                                            albedoMode_attr             = 'depth',
                                            shadingMode_attr            = 'shadeless',
                                            image_filter_size_attr      = 1,
                                            texture_filter_size_attr    = 1,
                                            vertexPos_input             = vP,
                                            vertexColor_input           = vC,
                                            texture_input               = T,
                                            shCoeff_input               = SH,
                                            targetImage_input           = tf.zeros( [1, 1, cameraReader.height, cameraReader.width, 3]),
                                            extrinsics_input            = camExtrinsics,
                                            intrinsics_input            = camIntrinsics,
                                            nodeName                    = 'test')

    return renderer