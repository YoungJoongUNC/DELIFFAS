#############################################################################################
#############################################################################################

import os
import numpy as np
import imageio
import tensorflow as tf
import AdditionalUtils.FileChecker as FileChecker
from Utils.model_main import \
    depth_render as DepthRenderer

import pdb
import cv2


# from Utils import depth_render as DepthRenderer

#############################################################################################
#############################################################################################

class NerfDataloader:

    #####################

    def __init__(self,
                 cameraReader=None,
                 objReader=None,
                 startFrame=-1000000,
                 N_rand=-1,
                 meshSequence=None,
                 all_rays_d=None,
                 all_rays_o=None,
                 depthDir='',
                 queueSize=-1,
                 useGTTex=False,
                 texDir='',
                 dataDir='',
                 imageDir='',
                 fgMaskDir='',
                 render_only=False,
                 calibName='',
                 useDepth=False,
                 i_train_camera=None,
                 iteration=-1,
                 dilateMask=False,
                 numViewsPerBatch=-1,
                 i_frames=None,
                 training=False,
                 quantitative_eval=False,
                 supervise_img_mask_fg=False,
                 N_batch=1,
                 shuffle_buffer=1000,
                 inference=False
                 ):

        self.N_rand = N_rand
        self.cameraReader = cameraReader
        self.objReader = objReader
        self.meshSequence = meshSequence
        self.all_rays_d = all_rays_d
        self.all_rays_o = all_rays_o
        self.startFrame = startFrame
        self.depthDir = depthDir
        self.queueSize = queueSize
        self.useGTTex = useGTTex
        self.texDir = texDir
        self.dataDir = dataDir
        self.imageDir = imageDir
        self.fgMaskDir = fgMaskDir
        self.render_only = render_only
        self.calibName = calibName
        self.useDepth = useDepth
        self.i_train_camera = i_train_camera
        self.iteration = iteration
        self.dilateMask = dilateMask
        self.numViewsPerBatch = numViewsPerBatch
        self.i_frames = i_frames
        self.training = training
        self.quantitative_eval = quantitative_eval
        self.N_batch = N_batch
        self.supervise_img_mask_fg = supervise_img_mask_fg
        self.shuffle_buffer = shuffle_buffer
        self.inference = False

    #####################

    def frameCheck(self, frame):
        if frame < self.startFrame:
            return self.startFrame
        else:
            return frame

    #####################

    def load_data(self, frameInt, frameString, createData=False, cameraId=-1):

        frameStr = frameString.numpy()
        frameStr = frameStr.decode()

        batch_rays_FG_d = []
        batch_rays_FG_o = []
        nearFar_FG = []
        target_s_FG = []
        target_s_mask = []


        # sample views
        cI = 0
        while cI < self.numViewsPerBatch:


            if cameraId == -1:
                camera_i = np.random.choice(self.i_train_camera)
            else:
                camera_i = cameraId



            if (
                    not self.render_only and not createData) or self.quantitative_eval:
                imgName = FileChecker.checkFilePath(
                    os.path.join(self.imageDir, str(frameStr).zfill(6),
                                 'image_alp_c_' + str(camera_i).zfill(
                                     3) + '_f_' + str(frameStr).zfill(
                                     6) + '.png'))

                img = imageio.imread(imgName)
                img = (np.array(img,
                                dtype=np.float32) / 255.0)

            # depth

            depthImage = DepthRenderer.renderDepth(cameraId=camera_i,
                                                   objreader=self.objReader,
                                                   cameraReader=self.cameraReader,
                                                   meshInstance=self.meshSequence[
                                                                frameInt - self.startFrame:frameInt - self.startFrame + 1] * 1000.0,
                                                   jellyOffset=0.03)
            near = depthImage[..., 0:1]
            far = depthImage[..., 1:2]

            depthImage = np.concatenate([near, far], 2)

            # FG
            dilationMask = depthImage[..., -1] != 0.0  #
            whereForeground = np.where(dilationMask)

            if self.inference:
                depthOrigImage = DepthRenderer.renderDepth(cameraId=camera_i,
                                                           objreader=self.objReader,
                                                           cameraReader=self.cameraReader,
                                                           meshInstance=self.meshSequence[
                                                                        frameInt - self.startFrame:frameInt - self.startFrame + 1] * 1000.0,
                                                           jellyOffset=0.0)
                nearOrig = depthOrigImage[..., 0:1]
                farOrig = depthOrigImage[..., 1:2]

                depthOrigImage = np.concatenate([nearOrig, farOrig],
                                                2)


                origMask = depthOrigImage[..., -1] != 0.0
                kernel = np.ones((3, 3), np.uint8)
                origMask = origMask.astype(np.float32)

                origMaskDilated = cv2.dilate(origMask, kernel)
                origMaskRefined = cv2.erode(origMaskDilated,
                                            np.ones((5, 5), np.uint8))
                origMaskFinal = origMaskRefined.astype(np.bool)

                riceMask = origMask + (1.0 - origMaskRefined)
                riceMask = np.clip(riceMask, 0.0, 1.0)
                riceMask = 1.0 - riceMask
                missingIdx = np.where(riceMask)


            # White
            if not self.render_only or self.quantitative_eval:
                image = img[:, :, :3]

                imageMask = img[:, :, 3:]



            numForegroundPixels = len(whereForeground[0])


            if numForegroundPixels <= self.N_rand and createData:
                numForegroundPixels = 4096 * 2
                whereForeground = (np.full((4096 * 2), 0, dtype=np.int32),
                                   np.full((4096 * 2), 0, dtype=np.int32))


            if numForegroundPixels > self.N_rand:

                offset = len(whereForeground[0]) % 1024
                if offset != 0:


                    tiledOffset1 = np.tile(whereForeground[0][0:1],(1024 - offset))
                    tiledOffset2 = np.tile(whereForeground[1][0:1],(1024 - offset))
                    whereForeground1 = np.concatenate((whereForeground[0], tiledOffset1), 0)
                    whereForeground2 = np.concatenate((whereForeground[1], tiledOffset2), 0)

                    whereForeground = (whereForeground1,whereForeground2)
                    numForegroundPixels= len(whereForeground[0])


                select_inds_FG = whereForeground

                nearFar_FG.append(depthImage[select_inds_FG])

                if (not self.render_only) or self.quantitative_eval:
                    target_s_FG.append(image[select_inds_FG])

                cI = cI + 1
            else:
                continue




        meshPoints_0 = self.meshSequence[frameInt - self.startFrame]  # [4852, 3]

        frameIdx_1 = self.startFrame if ((frameInt.numpy() - 1) <= self.startFrame) else (frameInt.numpy() - 1)
        meshPoints_1 = self.meshSequence[frameIdx_1 - self.startFrame]  # [4852, 3]

        frameIdx_2 = self.startFrame if ((frameInt.numpy() - 2) <= self.startFrame) else (frameInt.numpy() - 2)
        meshPoints_2 = self.meshSequence[frameIdx_2 - self.startFrame]  # [4852, 3]

        meshPoints = np.concatenate([meshPoints_0, meshPoints_1, meshPoints_2],-1) # [4852, 9]

        meshPoints = tf.constant(meshPoints)



        nearFar_FG = np.asarray(nearFar_FG, dtype=np.float32).reshape((-1, 2))


        nearFar_FG = tf.constant(nearFar_FG)

        frame = tf.reshape(tf.constant(frameInt), [1, 1])

        camera_i = tf.reshape(tf.constant(camera_i, dtype=tf.int32),
                              [1, 1])




        if (not self.render_only) or self.quantitative_eval:

            target_s_FG = np.asarray(target_s_FG, dtype=np.float32).reshape(
                (-1, 3))


            target_s_FG = tf.constant(target_s_FG)

            target_s_mask = np.asarray(target_s_mask, dtype=np.float32).reshape(
                (-1, 1))
            target_s_mask = tf.constant(target_s_mask)



        else:
            target_s_mask = tf.zeros([1], dtype=tf.float32)
            target_s_FG = tf.zeros([1], dtype=tf.float32)


            image = tf.zeros([1], dtype=tf.float32)

        returnDict = {}

        returnDict['meshPoints'] = meshPoints
        returnDict['frame_i'] = frame
        returnDict['camera_i'] = camera_i
        returnDict['numRays'] = tf.reshape(numForegroundPixels, [1, 1])
        returnDict['whereForeground'] = whereForeground
        returnDict['target_s'] = target_s_FG
        returnDict['image'] = image


        return returnDict

    #####################

    def load_tf_record_dataset(self):

        def py_func(frameInt, frameString):
            d = self.load_data(frameInt, frameString)
            return list(d.values())

        def parse_dataset(frameInt, frameString):

            data = tf.py_function(py_func, [frameInt, frameString],
                                  [tf.float32, tf.int32, tf.int32, tf.int32,
                                   tf.int32, tf.float32, tf.float32])

            return data

        
        frameStr = []
        for i in self.i_frames:
            frameStr.append(str(i))

        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.constant(self.i_frames), tf.constant(frameStr)))
        dataset = dataset.map(parse_dataset, num_parallel_calls=64)


        if self.training:
            bufferSize = self.shuffle_buffer
        else:
            bufferSize = 200

        if not self.render_only:
            dataset = dataset.shuffle(buffer_size=bufferSize, seed=1)

        dataset = dataset.repeat()


        dataset = dataset.prefetch(200)


        dataset = dataset.batch(self.N_batch)

        return dataset

#############################################################################################
#############################################################################################
