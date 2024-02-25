#############################################################################################
# Imports
#############################################################################################

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys

sys.path.append("../")

from Utils.arg_parser import config_parser
from Utils.ray_sample import *
from Utils.convert_utils import *

# when not evaluating the speed, use this line
from Utils.model_main.load_data import \
    NerfDataloader
# when evaluating the speed, use this line
# from Utils.model_main.load_data_measure_eval_time import \
#    NerfDataloader

from Utils.model_main.create_model import *

import AdditionalUtils.SlurmTimeLimitConverter as SlurmTimeLimitConverter
import AdditionalUtils.CheckGPU as CheckGPU
import AdditionalUtils.FileChecker as FileChecker
import AdditionalUtils.CreateMeshTensor as CreateMeshTensor
import AdditionalUtils.OBJReader as OBJReader
import AdditionalUtils.CameraReader as CameraReader
from AdditionalUtils.PrintFormat import *
import CudaRenderer.CudaRendererGPU as Renderer
import matplotlib.pyplot as plt
import CustomTFOperators.VGGLoss19 as VGGLoss

import imageio
import time
import tensorflow as tf
import tensorflow_addons as tfa

import CustomTFOperators.GlobalToUVSpaceGpu as GlobalToUVSpaceGpu
import pdb
import cv2






class Nerf:

    #############################################################################################
    # Render rays
    #############################################################################################

    def render_rays(self, out_uv, in_uv, outNormalFeature, origNormalFeature):


        raw_rgb = self.network_query_fn(out_uv, in_uv, outNormalFeature,
                                        origNormalFeature,
                                        self.modelList[0])
        rgb_map = tf.math.sigmoid(raw_rgb[..., :3])

        ret = {}
        ret['rgb_map'] = rgb_map

        return ret

    def batchify_rays(self, whereFG, jellyUVResult, normalFeaturesMap,
                      outerNormalFeaturesMap, isTrain,
                      num_rays):


        all_ret = {}

        nbatch = int(whereFG.shape[0] / 3)

        # Note: when not evaluating the speed, use this following four lines
        if num_rays[0][0] > 50000:
            chunk = 40960
        else:
            chunk = whereFG.shape[1]  # whereFG [3, 1500000, 2]

        # when evaluating the speed, use the following line
        # chunk = whereFG.shape[1]

        feat_map_sh = normalFeaturesMap.shape[1]



        for i in range(0, whereFG.shape[1], chunk):
            rawSampledUV = tf.gather_nd(batch_dims=1, params=jellyUVResult,
                                        indices=tf.cast(
                                            whereFG[:, i: i + chunk, :],
                                            dtype=tf.int32))

            sampledUV = rawSampledUV
            sampledVU = tf.concat([sampledUV[..., 1:], sampledUV[..., :1]],
                                  axis=-1)

            orig_vu = sampledVU[2 * nbatch:3 * nbatch, :, :]
            out_1st_vu = sampledVU[:nbatch, :, :]
            out_2nd_vu = sampledVU[nbatch:2 * nbatch, :, :]

            is_v_zero = (orig_vu[:, :, 0] == 0.0)
            is_u_zero = (orig_vu[:, :, 1] == 0.0)
            is_vu_zero = is_v_zero & is_u_zero


            valid_hit = (is_vu_zero == False)[ ..., None]
            valid_hit = tf.cast(valid_hit, tf.float32)
            invalid_hit = (1.0 - valid_hit)



            outFirstFeature = tfa.image.interpolate_bilinear(
                outerNormalFeaturesMap,
                out_1st_vu * feat_map_sh)
            outSecondFeature = tfa.image.interpolate_bilinear(
                outerNormalFeaturesMap,
                out_2nd_vu * feat_map_sh)

            origFeature = tfa.image.interpolate_bilinear(normalFeaturesMap,
                                                         (
                                                                     orig_vu * feat_map_sh))


            orig_uv = rawSampledUV[2 * nbatch:3 * nbatch, :, :]
            out_1st_uv = rawSampledUV[:nbatch, :, :]
            out_2nd_uv = rawSampledUV[nbatch:2 * nbatch, :, :]

            ret = self.render_rays(out_uv=out_1st_uv,
                                   in_uv=valid_hit * orig_uv + invalid_hit * out_2nd_uv,
                                   outNormalFeature=outFirstFeature,
                                   origNormalFeature=valid_hit * origFeature + invalid_hit * outSecondFeature)


            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])


        all_ret = {k: tf.concat(all_ret[k], 1) for k in all_ret}

        return all_ret

    #############################################################################################
    # Render
    #############################################################################################

    @tf.function
    def renderWithTFFunction(self, camera_i, whereFG, meshPoints,
                             num_rays,
                             isTrain):

        return self.render(camera_i=camera_i, whereFG=whereFG,
                           meshPoints=meshPoints, num_rays=num_rays,
                           isTrain=False)

    def getRenderer(self, vertForUV, renderNameUV, camera_i, mode, njelly=1):


        global renderResolutionU_attr
        nbatch = camera_i.shape[0]
        nbatchjelly = nbatch * njelly


        camExtrinsics = tf.gather(self.extrinsics, camera_i, axis=0,
                                  batch_dims=1)

        camExtrinsics = tf.tile(camExtrinsics,
                                [njelly, 1, 1, 1])
        camExtrinsics = tf.reshape(camExtrinsics,
                                   [nbatchjelly, -1])

        camIntrinsics = tf.gather(self.intrinsics, camera_i, axis=0,
                                  batch_dims=1)

        camIntrinsics = tf.tile(camIntrinsics,
                                [njelly, 1, 1, 1])
        camIntrinsics = tf.reshape(camIntrinsics,
                                   [nbatchjelly, -1])


        if mode == 'uv':
            albedoMode_attr = 'uv'
            compute_normal_map_attr = False
            enable_depth_peel_attr = True
            renderResolutionU_attr = self.args.resolutionU
            renderResolutionV_attr = self.args.resolutionV

        elif mode == 'normal':
            albedoMode_attr = 'normal'
            compute_normal_map_attr = True
            enable_depth_peel_attr = False
            renderResolutionU_attr = 1024  # 64
            renderResolutionV_attr = 1024  # 64

        if mode == 'position':
            albedoMode_attr = 'position'
            compute_normal_map_attr = False
            enable_depth_peel_attr = True
            renderResolutionU_attr = self.args.resolutionU
            renderResolutionV_attr = self.args.resolutionV

        renderer = Renderer.CudaRendererGpu(
            faces_attr=self.renderFaces,
            texCoords_attr=self.renderTexCoords,
            numberOfVertices_attr=self.numberOfVertices,
            numberOfCameras_attr=self.args.num_views_per_batch,
            renderResolutionU_attr=renderResolutionU_attr,
            renderResolutionV_attr=renderResolutionV_attr,
            albedoMode_attr=albedoMode_attr,
            shadingMode_attr='shadeless',
            image_filter_size_attr=1,
            texture_filter_size_attr=1,
            compute_normal_map_attr=compute_normal_map_attr,
            enable_depth_peel_attr=enable_depth_peel_attr,
            vertexPos_input=vertForUV,
            vertexColor_input=tf.zeros([nbatchjelly, 4852, 3]),
            texture_input=tf.zeros([nbatchjelly, 1024, 1024, 3]),
            shCoeff_input=tf.zeros(
                [nbatchjelly, self.args.num_views_per_batch, 27],
                dtype=tf.float32),
            targetImage_input=tf.zeros(
                [nbatchjelly, self.args.num_views_per_batch,
                 self.args.resolutionV,
                 self.args.resolutionU, 3], dtype=tf.float32),
            extrinsics_input=camExtrinsics,
            intrinsics_input=camIntrinsics,
            nodeName=renderNameUV)


        return renderer

    def fill_missing_pixel(self, uvmap):


        tmp_uvmap = uvmap.numpy()

        tmp_uvmap1 = tmp_uvmap[0, 0, :, :, 0]
        tmp_uvmap1 = tmp_uvmap1 != 0.0

        tmp_uvmap2 = tmp_uvmap[0, 0, :, :, 1]
        tmp_uvmap2 = tmp_uvmap2 != 0.0
        origMask = tmp_uvmap1 * tmp_uvmap2

        origMask = origMask.astype(np.float32)

        origMaskDilated = cv2.dilate(origMask,
                                     np.ones((3, 3), np.uint8))
        origMaskRefined = cv2.erode(origMaskDilated,
                                    np.ones((5, 5), np.uint8))  # (940, 1285)

        riceMask = origMask + (1.0 - origMaskRefined)  # unique [0.0, 1.0, 2.0]

        riceMask = np.clip(riceMask, 0.0, 1.0)
        riceMask = 1.0 - riceMask  # (940, 1285)
        missingIdx = np.where(riceMask)
        nMissing = len(missingIdx[0])

        if nMissing > 0:
            for idx in range(nMissing):
                nValid = 0
                values = np.zeros([3], np.float32)
                if missingIdx[0][idx] + 1 < 940:
                    upPixel = tmp_uvmap[0, 0][
                        missingIdx[0][idx] + 1, missingIdx[1][idx]]
                    values = values + upPixel
                    nValid = nValid + np.all(upPixel[:2] != 0.0).astype(int)
                if missingIdx[0][idx] - 1 >= 0:
                    downPixel = tmp_uvmap[0, 0][
                        missingIdx[0][idx] - 1, missingIdx[1][idx]]
                    values = values + downPixel
                    nValid = nValid + np.all(downPixel[:2] != 0.0).astype(int)
                if missingIdx[1][idx] + 1 < 1285:
                    rightPixel = tmp_uvmap[0, 0][
                        missingIdx[0][idx], missingIdx[1][idx] + 1]
                    values = values + rightPixel
                    nValid = nValid + np.all(rightPixel[:2] != 0.0).astype(int)
                if missingIdx[1][idx] - 1 >= 0:
                    leftPixel = tmp_uvmap[0, 0][
                        missingIdx[0][idx], missingIdx[1][idx] - 1]
                    values = values + leftPixel
                    nValid = nValid + np.all(leftPixel[:2] != 0.0).astype(int)

                if nValid != 0:
                    tmp_uvmap[0, 0][missingIdx[0][idx], missingIdx[1][
                        idx]] = values / nValid

            uvmap = tf.constant(tmp_uvmap)

        return uvmap

    #############################################################################################
    # Render with tf function
    #############################################################################################

    def render(self, camera_i, whereFG, meshPoints, num_rays,
               isTrain=False):


        nbatch = camera_i.shape[0]


        whereFG = tf.transpose(whereFG, perm=[0, 2, 1])
        whereFG = tf.cast(whereFG, dtype=tf.float32)
        whereFG = tf.tile(whereFG, [3, 1, 1])

        # scale from meter to mm

        meshPoints = meshPoints * 1000.0
        original_jelly = meshPoints[..., :3]  # time step t
        original_jelly_1 = meshPoints[..., 3:6]  # time step t-1
        original_jelly_2 = meshPoints[..., 6:9]  # time step t-2



        origUVRenderer = self.getRenderer(original_jelly, 'CudaOrigUVRenderer',
                                          camera_i, mode='uv')
        origUVResult = origUVRenderer.getRenderBufferTF()

        origUVResult = tf.reshape(origUVResult,
                                  [nbatch * 1, self.args.num_views_per_batch,
                                   self.args.resolutionV, self.args.resolutionU,
                                   3])

        origUVRenderer_1 = self.getRenderer(original_jelly_1,
                                            'CudaOrigUVRenderer_1', camera_i,
                                            mode='uv')
        origUVResult_1 = origUVRenderer_1.getRenderBufferTF()

        origUVResult_1 = tf.reshape(origUVResult_1,
                                    [nbatch * 1, self.args.num_views_per_batch,
                                     self.args.resolutionV,
                                     self.args.resolutionU,
                                     3])

        origUVRenderer_2 = self.getRenderer(original_jelly_2,
                                            'CudaOrigUVRenderer_2', camera_i,
                                            mode='uv')
        origUVResult_2 = origUVRenderer_2.getRenderBufferTF()

        origUVResult_2 = tf.reshape(origUVResult_2,
                                    [nbatch * 1, self.args.num_views_per_batch,
                                     self.args.resolutionV,
                                     self.args.resolutionU,
                                     3])

        if self.args.inference:
            origUVResult = self.fill_missing_pixel(origUVResult)
            origUVResult_1 = self.fill_missing_pixel(origUVResult_1)
            origUVResult_2 = self.fill_missing_pixel(origUVResult_2)

        # vertex normal before normalization
        rawVertexNormals = origUVRenderer.getVertexNormal()
        rawVertexNormals = tf.reshape(rawVertexNormals, [nbatch * 1, -1, 3])
        # normalize vertex normals
        vertexNormals = tf.math.l2_normalize(rawVertexNormals, axis=2)[0]

        rawVertexNormals_1 = origUVRenderer_1.getVertexNormal()
        rawVertexNormals_1 = tf.reshape(rawVertexNormals_1, [nbatch * 1, -1, 3])
        # normalize vertex normals
        vertexNormals_1 = tf.math.l2_normalize(rawVertexNormals_1, axis=2)[0]

        rawVertexNormals_2 = origUVRenderer_2.getVertexNormal()
        rawVertexNormals_2 = tf.reshape(rawVertexNormals_2, [nbatch * 1, -1, 3])
        # normalize vertex normals
        vertexNormals_2 = tf.math.l2_normalize(rawVertexNormals_2, axis=2)[0]

        jelly_weight = 30

        # make outer jelly
        outer_jelly = original_jelly + jelly_weight * vertexNormals
        outer_jelly_1 = original_jelly_1 + jelly_weight * vertexNormals_1
        outer_jelly_2 = original_jelly_2 + jelly_weight * vertexNormals_2

        # make inner jelly
        if self.args.use_inner_jelly:

            inner_jelly = original_jelly + 10 * vertexNormals  # [1, 4852, 3]
            inner_jelly_1 = original_jelly_1 + 10 * vertexNormals_1  # [1, 4852, 3]
            inner_jelly_2 = original_jelly_2 + 10 * vertexNormals_2  # [1, 4852, 3]




        jellyUVRenderer = self.getRenderer(outer_jelly, 'CudaJellyUVRenderer',
                                           camera_i, mode='uv', njelly=1)
        jellyUVResult = jellyUVRenderer.getRenderBufferTF()
        jellyUVSecondResult = jellyUVRenderer.getRenderSecondBufferTF()

        if self.args.inference:
            jellyUVResult = self.fill_missing_pixel(jellyUVResult)
            jellyUVSecondResult = self.fill_missing_pixel(jellyUVSecondResult)


        jellyUVResult = tf.reshape(jellyUVResult,
                                   [nbatch * 1, self.args.num_views_per_batch,
                                    self.args.resolutionV,
                                    self.args.resolutionU, 3])

        jellyUVSecondResult = tf.reshape(jellyUVSecondResult,
                                         [nbatch * 1,
                                          self.args.num_views_per_batch,
                                          self.args.resolutionV,
                                          self.args.resolutionU, 3])
        if self.args.use_inner_jelly:
            innerUVRenderer = self.getRenderer(inner_jelly,
                                               'CudaInnerUVRenderer',
                                               camera_i, mode='uv')
            innerUVResult = innerUVRenderer.getRenderBufferTF()


            innerUVResult = tf.reshape(innerUVResult,
                                       [nbatch * 1,
                                        self.args.num_views_per_batch,
                                        self.args.resolutionV,
                                        self.args.resolutionU,
                                        3])



        if self.args.use_inner_jelly:
            jellyUVResult = tf.concat(
                [jellyUVResult, jellyUVSecondResult, innerUVResult],
                axis=0)
        else:
            jellyUVResult = tf.concat(
                [jellyUVResult, jellyUVSecondResult, origUVResult],
                axis=0)


        jellyUVResult = tf.reshape(jellyUVResult,
                                   [nbatch * 3, self.args.resolutionV,
                                    self.args.resolutionU, 3])

        jellyUVResult = jellyUVResult[...,
                        : 2]


        temporal_jelly = tf.concat(
            [original_jelly, original_jelly_1, original_jelly_2], axis=0)

        temporal_outer_jelly = tf.concat(
            [outer_jelly, outer_jelly_1, outer_jelly_2], axis=0)

        jellyNormalRenderer = self.getRenderer(temporal_jelly,
                                               'CudaJellyNormalRenderer',
                                               camera_i, mode='normal',
                                               njelly=3)

        outerjellyNormalRenderer = self.getRenderer(temporal_outer_jelly,
                                                    'CudaOuterJellyNormalRenderer',
                                                    camera_i, mode='normal',
                                                    njelly=3)


        jellyNormalMap = jellyNormalRenderer.getNormalMap()
        outerJellyNormalMap = outerjellyNormalRenderer.getNormalMap()


        jellyNormalMap_0 = jellyNormalMap[
                           :nbatch]
        jellyNormalMap_1 = jellyNormalMap[
                           nbatch:2 * nbatch]
        jellyNormalMap_2 = jellyNormalMap[2 * nbatch:]


        temporalJellyNormalMap = tf.concat(
            [jellyNormalMap_0, jellyNormalMap_1, jellyNormalMap_2], -1)


        normalFeaturesMap = self.featureExtractor_orig(temporalJellyNormalMap)

        outerJellyNormalMap_0 = outerJellyNormalMap[
                                :nbatch]
        outerJellyNormalMap_1 = outerJellyNormalMap[
                                nbatch:2 * nbatch]
        outerJellyNormalMap_2 = outerJellyNormalMap[
                                2 * nbatch:]


        temporalOuterJellyNormalMap = tf.concat(
            [outerJellyNormalMap_0, outerJellyNormalMap_1,
             outerJellyNormalMap_2], -1)


        outerNormalFeaturesMap = self.featureExtractor_out(
            temporalOuterJellyNormalMap)

        all_ret = self.batchify_rays(whereFG=whereFG,
                                     jellyUVResult=jellyUVResult,
                                     normalFeaturesMap=normalFeaturesMap,
                                     outerNormalFeaturesMap=outerNormalFeaturesMap,
                                     isTrain=isTrain,
                                     num_rays=num_rays
                                     )


        return all_ret

    #############################################################################################
    # training step
    #############################################################################################

    def trainStep(self, camera_i, whereFG, meshPoints, target_s,
                  num_rays):

        with tf.GradientTape(persistent=True) as tape:


            ret_dict = self.render(camera_i=camera_i,
                                   whereFG=whereFG,
                                   meshPoints=meshPoints, num_rays=num_rays,
                                   isTrain=True)

            rgb = ret_dict['rgb_map']

            # Compute MSE loss between predicted and true RGB.

            # image mask area
            img_loss = img2mseL1(rgb, target_s) / float(self.numGpus)
            img_mse_loss = img2mse(rgb, target_s) / float(self.numGpus)

            psnr = mse2psnr(img_mse_loss)

            # Compute GAN loss
            nbatch = whereFG.shape[0]  # nbatch is distributed batch (self.args.N_batch / GPUs
            indices = tf.transpose(whereFG, perm=[0, 2,
                                                  1])  # [nbatch, 2, # FG pixel MAX]

            if self.args.fg_loss_only:
                new_H = 512
                new_W = 512
            else:
                new_H = self.args.resolutionV
                new_W = self.args.resolutionU

            genImgList = []
            gtImgList = []

            if self.args.fg_loss_only:
                h_min = tf.reduce_min(indices[..., 0])
                h_max = tf.reduce_max(indices[..., 0])
                h_len = h_max - h_min + 1  # int32


                w_min = tf.reduce_min(indices[..., 1])
                w_max = tf.reduce_max(indices[..., 1])
                w_len = w_max - w_min + 1  # int32



                new_length = tf.math.maximum(h_len, w_len)

                h_offset = tf.cast((new_H - h_len) / 2, tf.int32)
                w_offset = tf.cast((new_W - w_len) / 2, tf.int32)
                indices = tf.concat([indices[..., 0:1] - h_min + h_offset,
                                     indices[..., 1:] - w_min + w_offset],
                                    axis=-1)

            for idx in range(nbatch):
                genImg = tf.ones([new_H, new_W, 3])

                genImg = tf.tensor_scatter_nd_update(genImg, indices[idx],
                                                     rgb[idx])
                genImgList.append(genImg)

                gtImg = tf.ones([new_H, new_W, 3])
                gtImg = tf.tensor_scatter_nd_update(gtImg, indices[idx],
                                                    target_s[idx])
                gtImgList.append(gtImg)

            genImg = tf.stack(genImgList, axis=0)
            gtImg = tf.stack(gtImgList, axis=0)

            if self.args.use_adversarial_loss:
                disc_real_output = self.discriminator([gtImg],
                                                      training=True)
                disc_generated_output = self.discriminator([genImg],
                                                           training=True)

                gen_gan_loss = self.discriminatorClass.generator_loss(
                    disc_generated_output)
                disc_loss = self.discriminatorClass.discriminator_loss(
                    disc_real_output, disc_generated_output)

                gen_gan_loss = gen_gan_loss / float(self.numGpus)
                disc_loss = (disc_loss / float(self.numGpus))

            ssim_score = img2ssim(genImg, gtImg)
            ssim_loss = (1.0 - ssim_score)
            ssim_loss = ssim_loss / float(self.numGpus)

            loss = self.args.w_img_FG_loss * img_loss

            if self.args.use_adversarial_loss:
                loss += self.args.w_adversarial_loss * gen_gan_loss

            if self.args.use_perceptual_loss:
                perceptual_loss = VGGLoss.lossVGG(genImg, gtImg,
                                                  self.vggLossModel)
                perceptual_loss = perceptual_loss / float(self.numGpus)
                loss += self.args.w_perceptual_loss * perceptual_loss

            if self.args.use_ssim_loss:
                loss += self.args.w_ssim_loss * ssim_loss

        '''
        Adding the loss should be done inside the gradient tape
        '''
        gradients = tape.gradient(loss, self.gradVars)
        self.optimizer.apply_gradients(zip(gradients, self.gradVars))
        if self.args.use_adversarial_loss:
            discriminator_gradients = tape.gradient(disc_loss, self.discVars)
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discVars))

        loss_dict = {}
        loss_dict['img_loss'] = img_loss
        loss_dict['psnr'] = psnr
        loss_dict['ssim_loss'] = ssim_loss
        if self.args.use_adversarial_loss:
            loss_dict['gen_gan_loss'] = gen_gan_loss
            loss_dict['disc_loss'] = disc_loss
        if self.args.use_perceptual_loss:
            loss_dict['perceptual_loss'] = perceptual_loss


        return loss_dict

    #############################################################################################
    # runTrain
    #############################################################################################

    # @tf.function
    def runTrain(self, camera_i, whereFG, meshPoints, target_s,
                 num_rays):

        def value_fn(ctx):

            multiplier = int(self.N_batch / ctx.num_replicas_in_sync)
            start = multiplier * ctx.replica_id_in_sync_group
            end = multiplier * (ctx.replica_id_in_sync_group + 1)

            return camera_i[start:end, :], \
                   whereFG[start:end, :, :], \
                   meshPoints[start:end, :, :], \
                   target_s[start:end, :, :], \
                   num_rays[start:end, :]

        camera_i, whereFG, meshPoints, target_s, num_rays = self.mirrored_strategy.experimental_distribute_values_from_function(
            value_fn)

        loss_dict = \
            self.mirrored_strategy.run(self.trainStep, args=(
                camera_i, whereFG, meshPoints, target_s, num_rays,))

        final_loss_dict = {}

        lossF = loss_dict['img_loss']
        psnrF = loss_dict['psnr']
        ssim_loss = loss_dict['ssim_loss']

        lossF = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, lossF,
                                              axis=None)
        psnrF = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, psnrF,
                                              axis=None)
        ssim_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                  ssim_loss, axis=None)

        final_loss_dict['img_loss'] = lossF
        final_loss_dict['psnr'] = psnrF
        final_loss_dict['ssim_loss'] = ssim_loss

        if self.args.use_adversarial_loss:
            gen_gan_loss = loss_dict['gen_gan_loss']
            disc_loss = loss_dict['disc_loss']
            gen_gan_loss = self.mirrored_strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                gen_gan_loss, axis=None)
            disc_loss = self.mirrored_strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                disc_loss, axis=None)
            final_loss_dict['gen_gan_loss'] = gen_gan_loss
            final_loss_dict['disc_loss'] = disc_loss

        if self.args.use_perceptual_loss:
            perceptual_loss = loss_dict['perceptual_loss']
            perceptual_loss = self.mirrored_strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                perceptual_loss, axis=None)
            final_loss_dict['perceptual_loss'] = perceptual_loss


        return final_loss_dict

    #############################################################################################
    # init
    #############################################################################################

    def __init__(self, args):

        ###################################

        print1Level('Parse argument!')


        self.args = args

        self.slurmStartTime = time.time()
        self.jobTimeLimit = SlurmTimeLimitConverter.convertTime(
            self.args.timeLimit)
        self.meshfilepath = self.args.meshfilepath  # '/HPS/RTMPC4/work/DeepDynamicCharacters/Datasets/VladNew/vlad.obj'
        self.charFilePath = self.args.charFilePath  # '/HPS/RTMPC4/work/DeepDynamicCharacters/Datasets/VladNew/vlad.character'
        self.positional_pose = self.args.positional_pose
        self.use_viewdirs = self.args.use_viewdirs
        self.N_samples = self.args.N_samples
        self.perturb = self.args.perturb
        self.raw_noise_std = self.args.raw_noise_std
        self.chunk = self.args.chunk
        self.N_rand = self.args.N_rand
        self.numGpus = CheckGPU.print_gpu_usage()  # 1
        self.expname = self.args.expname
        self.basedir = self.args.basedir
        self.storepath = None
        self.N_batch = self.args.N_batch

        print2Level('Number of gpus: ' + str(self.numGpus))

        ###################################

        print1Level('Fix seed!')

        if self.args.random_seed is not None:
            print2Level('Fixing random seed ' + str(self.args.random_seed))
            np.random.seed(self.args.random_seed)
            tf.compat.v1.set_random_seed(self.args.random_seed)

        ###################################

        print1Level('Create log dir and copy the config file!')
        print2Level('Basedir: ' + str(self.basedir))
        print2Level('Experiment Name: ' + self.expname)

        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)
        f = os.path.join(self.basedir, self.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.args.config is not None:
            f = os.path.join(self.basedir, self.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())

        ###################################

        print1Level('Create camera reader!')
        # '/HPS/RTMPC4/work/DeepDynamicCharacters/Datasets/VladNew/cameras.calibration'
        if self.args.calib in ['rotation_steps_100', 'rotation', 'rotation_x',
                               'rotation_z']:
            camPath = os.path.join(self.args.calibdir,
                                   self.args.calib + '.calibration')
        else:
            camPath = os.path.join(self.args.datadir,
                                   self.args.calib + '.calibration')

        self.cameraReader = CameraReader.CameraReader(camPath,
                                                      self.args.resolutionU,
                                                      self.args.resolutionV)
        self.number_of_cameras = self.cameraReader.numberOfCameras
        self.extrinsics = tf.reshape(self.cameraReader.extrinsics,
                                     [self.number_of_cameras, 3,
                                      4])
        self.intrinsics = tf.reshape(self.cameraReader.intrinsics,
                                     [self.number_of_cameras, 3,
                                      3])



        print1Level('Bodypart networks')
        if self.args.multiNet:
            file = open(self.args.datadir + '/segmentation.txt', 'r')
            meshLabelset = set()

            for line in file:
                meshLabelset.add(int(line.split()[0]))
            meshLabelset.add(-1000)

            self.meshLabelset = meshLabelset
            self.numberOfSmallNetworks = len(meshLabelset)
            print2Level('Number of different body part labels ' + str(
                self.numberOfSmallNetworks))
            print2Level('Labels are: ' + str(meshLabelset))
        else:
            print2Level('No multinet selected')
            self.meshLabelset = None
            self.numberOfSmallNetworks = 1

        ###################################

        print1Level('Load mesh sequence!')

        meshFilePath = FileChecker.checkFilePath(self.args.mesh_seq_path)
        if meshFilePath is not None:
            print2Level('Meshes found under ' + meshFilePath)
            self.meshSequence = CreateMeshTensor.createMeshSequenceTensor(
                inputMeshesFile=meshFilePath,
                startFrame=self.args.train_start_frame,
                numSamples=self.args.train_end_frame - self.args.train_start_frame,
                scale=0.001)

        else:
            return

        ###################################

        print1Level('Load mesh!')

        self.objreader = OBJReader.OBJReader(self.meshfilepath)
        self.numberOfVertices = self.objreader.numberOfVertices  # 4852
        self.renderFaces = self.objreader.facesVertexId  # 29190
        self.renderTexCoords = self.objreader.textureCoordinates  # 58380

        vc = tf.constant(self.objreader.vertexColors, dtype=tf.float32)
        self.vertexColor = tf.reshape(vc, [1, self.numberOfVertices,
                                           3])  # [1, 4852, 3] # min : 0.13, max: 1.0

        vt = tf.constant(self.objreader.textureMap, dtype=tf.float32)
        self.vertexTexture = tf.reshape(vt, [1, self.objreader.texHeight,
                                             self.objreader.texWidth,
                                             3])  # [1, 1024, 1024, 3]

        ###################################

        print1Level('Precompute all rays!')

        ###################################

        print1Level('View Split!')
        self.i_train_camera = self.args.train_cameras
        self.i_val_camera = self.args.validation_cameras
        self.i_test_camera = self.args.test_cameras

        print2Level('Train cameras: ' + str(self.i_train_camera))
        print2Level('Validation cameras: ' + str(self.i_val_camera))
        print2Level('Test cameras: ' + str(self.i_test_camera))

        ####################################

        print1Level('Frame!')
        self.i_frames = range(self.args.train_start_frame,
                              self.args.train_end_frame, 10)
        print2Level(str(self.i_frames))

        ####################################

        print1Level('Data loader!')
        print('--Init normal dataset', flush=True)

        self.nerfDataLoader = NerfDataloader(cameraReader=self.cameraReader,
                                             objReader=self.objreader,
                                             startFrame=self.args.train_start_frame,
                                             N_rand=self.N_rand,
                                             meshSequence=self.meshSequence,
                                             all_rays_d=None,
                                             all_rays_o=None,
                                             depthDir=self.args.depthdir,
                                             queueSize=self.args.queue_size,
                                             useGTTex=self.args.use_gt_tex,
                                             texDir=self.args.texdir,
                                             dataDir=self.args.datadir,
                                             imageDir=self.args.imagedir,
                                             fgMaskDir=self.args.fgmaskdir,
                                             render_only=self.args.render_only,
                                             calibName=self.args.calib,
                                             useDepth=self.args.depth,
                                             i_train_camera=self.i_train_camera,
                                             iteration=self.args.iteration,
                                             dilateMask=self.args.dilate_mask,
                                             numViewsPerBatch=self.args.num_views_per_batch,
                                             i_frames=self.i_frames,
                                             training=True,
                                             supervise_img_mask_fg=self.args.supervise_img_mask_fg,
                                             N_batch=self.args.N_batch,
                                             shuffle_buffer=self.args.shuffle_buffer)

        self.nerfDataLoaderTest = NerfDataloader(cameraReader=self.cameraReader,
                                                 objReader=self.objreader,
                                                 startFrame=self.args.train_start_frame,
                                                 N_rand=self.N_rand,
                                                 meshSequence=self.meshSequence,
                                                 all_rays_d=None,
                                                 all_rays_o=None,
                                                 depthDir=self.args.depthdir,
                                                 queueSize=self.args.queue_size,
                                                 useGTTex=self.args.use_gt_tex,
                                                 texDir=self.args.texdir,
                                                 dataDir=self.args.datadir,
                                                 imageDir=self.args.imagedir,
                                                 fgMaskDir=self.args.fgmaskdir,
                                                 render_only=self.args.render_only,
                                                 calibName=self.args.calib,
                                                 useDepth=self.args.depth,
                                                 i_train_camera=self.i_test_camera,
                                                 iteration=self.args.iteration,
                                                 dilateMask=False,
                                                 numViewsPerBatch=1,
                                                 i_frames=self.i_frames,
                                                 training=False,
                                                 quantitative_eval=self.args.quantitative_eval,
                                                 N_batch=1,
                                                 shuffle_buffer=self.args.shuffle_buffer,
                                                 inference=self.args.inference)

        if not self.args.render_only:

            dataset = self.nerfDataLoader.load_tf_record_dataset()
            self.datasetIterator = iter(dataset)

            datasetTest = self.nerfDataLoaderTest.load_tf_record_dataset()
            self.datasetTestIterator = iter(datasetTest)

        ###################################

        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with  self.mirrored_strategy.scope():

            ###################################

            print1Level('Create nerf model!')

            self.textureFeatureSize = self.args.textureFeatureSize  # 32

            network_query_fn, modelList, start, grad_vars, disc_vars, models, embedMotion_fn, discriminatorClass = create_nerf(
                args=self.args,
                textureFeatureSize=self.textureFeatureSize,  # 32
                positionalPose=self.positional_pose,  # True
                numberOfSmallNetworks=self.numberOfSmallNetworks)  # 1

            self.network_query_fn = network_query_fn
            self.modelList = modelList

            self.featureExtractor_out = models['featureExtractor_out']
            self.featureExtractor_orig = models['featureExtractor_orig']

            if self.args.use_adversarial_loss:
                self.discriminator = models['discriminator']
                self.discriminatorClass = discriminatorClass
            if self.args.use_perceptual_loss:
                if self.args.fg_loss_only:
                    new_H = 512
                    new_W = 512
                else:
                    new_H = self.args.resolutionV
                    new_W = self.args.resolutionU

                self.vggLossModel = VGGLoss.createVGGLossModel(
                    resolutionU=new_W,
                    resolutionV=new_H)
            self.start = start

            self.models = models
            self.gradVars = grad_vars  # len 127
            if self.args.use_adversarial_loss:
                self.discVars = disc_vars

            ###################################

            print1Level('Create optimizer!')

            lrate = self.args.lrate
            if self.args.lrate_decay > 0:
                lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                                       decay_steps=self.args.lrate_decay * 1000,
                                                                       decay_rate=0.1)

            self.optimizer = tf.keras.optimizers.Adam(lrate)


            if self.args.use_adversarial_loss:
                self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                                        beta_1=0.5)

            ###################################

    #############################################################################################
    # Train
    #############################################################################################

    def train(self):

        ###################################

        print1Level('Create summary writer!')
        writer = tf.summary.create_file_writer(
            os.path.join(self.basedir, 'summaries', self.expname))
        writer.set_as_default()

        ###################################

        print1Level('Start training!')
        N_iters = 4000000 + self.args.iteration * 10000000

        ###################################

        best_psnr = 23.5


        for i in range(self.start, N_iters):

            time0 = time.time()

            ###################################
            # Load data

            meshPoints, frame_i, camera_i, num_rays, whereFG, target_s, image = next(
                self.datasetIterator)


            # reshape data
            frame_i = tf.reshape(frame_i, [self.N_batch, 1])
            camera_i = tf.reshape(camera_i, [self.N_batch, 1])
            num_rays = tf.reshape(num_rays, [self.N_batch, 1])



            ###################################
            # Train step

            with self.mirrored_strategy.scope():

                loss_dict = \
                    self.runTrain(camera_i, whereFG, meshPoints,
                                  target_s, num_rays)
                loss = loss_dict['img_loss']
                psnr = loss_dict['psnr']
                ssim_loss = loss_dict['ssim_loss']

                if self.args.use_adversarial_loss:
                    gen_gan_loss = loss_dict['gen_gan_loss']
                    disc_loss = loss_dict['disc_loss']
                if self.args.use_perceptual_loss:
                    perceptual_loss = loss_dict['perceptual_loss']


            ###################################
            # Save weights

            def save_weights(net, prefix, i):
                path = os.path.join(self.basedir, self.expname,
                                    '{}_{:07d}.npy'.format(prefix, i))
                np.save(path, net.get_weights())
                print('saved weights at', path, flush=True)


            slurmTimeout = time.time() - self.slurmStartTime
            terminateToTime = slurmTimeout > (self.jobTimeLimit - 15 * 60)


            if i % self.args.i_weights == 0 or terminateToTime:  # after 10,000

                for k in self.models:
                    save_weights(self.models[k], k, i)

                if terminateToTime:
                    return

            ###################################
            # Rest is logging

            if i % self.args.i_print == 0 or i < 10:  # after 100

                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('psnr', psnr, step=i)
                tf.summary.scalar('ssim_loss', ssim_loss, step=i)
                if self.args.use_adversarial_loss:
                    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=i)
                    tf.summary.scalar('disc_loss', disc_loss, step=i)
                if self.args.use_perceptual_loss:
                    tf.summary.scalar('perceptual_loss', perceptual_loss,
                                      step=i)  #

            ###################################
            # validation

            if i % self.args.i_img == 0 or i == 1:

                self.storepath = '/HPS/RTMPC4/nobackup/test.obj'

                # log variables
                for n in range(0, self.numberOfSmallNetworks):
                    for variable in self.models[
                        'model_' + str(n)].trainable_weights:
                        tf.summary.histogram('nerf_backbone/' + variable.name,
                                             variable, step=i)

                meshPoints, frame_i, camera_i, num_rays, whereFG, target_s, target \
                    = next(self.datasetTestIterator)


                frame_i = tf.reshape(frame_i, [1, 1])
                camera_i = tf.reshape(camera_i, [1, 1])
                num_rays = tf.reshape(num_rays, [1, 1])
                target = target[0]

                ###################################
                # render test

                self.perturb = False
                self.raw_noise_std = 0.0


                ret_dict = self.render(
                    camera_i=camera_i,
                    whereFG=whereFG,
                    meshPoints=meshPoints,
                    num_rays=num_rays,
                    isTrain=True)



                self.perturb = self.args.perturb
                self.raw_noise_std = self.args.raw_noise_std
                rgb = ret_dict['rgb_map']  # [1, #FGpixels,3]



                #####################
                # test mse and psnr
                frame_i = frame_i[0, 0].numpy()
                camera_i = camera_i[0, 0].numpy()

                img_loss = img2mse(rgb, target_s) / float(self.numGpus)
                psnr = mse2psnr(img_loss)

                #####################
                # combine FG with white background
                rgbFinal = np.ones(
                    [self.args.resolutionV, self.args.resolutionU, 3],
                    np.float32)

                rgbFinal[tuple(whereFG[0])] = rgb[0]


                #####################
                # evaluate

                testimgdir = os.path.join(self.basedir, self.expname,
                                          'tboard_val_imgs')
                if i == 0:
                    os.makedirs(testimgdir, exist_ok=True)

                imageio.imwrite(os.path.join(testimgdir,
                                             'iter_' + str(i) + '_c_' + str(
                                                 camera_i) + '_f_' + str(
                                                 frame_i) + '.png'),
                                to8b(rgbFinal))

                tf.summary.image('error', to8b(
                    tf.abs(rgbFinal[tf.newaxis] - target[tf.newaxis])), step=i)
                tf.summary.image('rgb', to8b(rgbFinal)[tf.newaxis], step=i)
                tf.summary.scalar('img_loss_holdout', img_loss, step=i)
                tf.summary.scalar('psnr_holdout', psnr, step=i)
                '''
                if self.args.use_ssim_loss:
                    tf.summary.scalar('ssim_holdout', ssim_loss, step=i)
                '''
                tf.summary.image('rgb_holdout', target[tf.newaxis], step=i)

                self.storepath = None

                print('TEST',
                      ' ~~ ',
                      self.expname,
                      ' ~~ ',
                      ' Image: ',
                      str(frame_i).zfill(7),
                      ' ~~ ',
                      ' Camera: ',
                      str(camera_i).zfill(7),
                      ' ~~ ',
                      ' Loss: ',
                      '{:.6f}'.format(img_loss),
                      ' ~~ ',
                      ' PSNR: ',
                      '{:.6f}'.format(psnr),
                      flush=True)
                '''
                #' Gen LR: ',
                #'{:.6f}'.format(self.optimizer.lr),
                #' Disc LR: ',
                #'{:.6f}'.format(self.discriminator_optimizer.lr),
                '''

            ##################################
            # Report iteration time
            elapsed_time = time.time() - time0

            print(self.expname,
                  ' ~~ ',
                  ' Iter: ',
                  str(i).zfill(7),
                  ' ~~ ',
                  ' Loss: ',
                  '{:.6f}'.format(loss.numpy()),
                  ' ~~ ',
                  ' PSNR: ',
                  '{:.6f}'.format(psnr.numpy()),
                  ' ~~ ',
                  'Time passed ' + '{:.2f}'.format(
                      slurmTimeout) + '/' + '{:.2f}'.format(
                      self.jobTimeLimit - 15 * 60),
                  ' ~~ ',
                  'Iter time {:.02f}'.format(elapsed_time), flush=True)


    #############################################################################################
    # Test
    #############################################################################################

    def test(self):

        print('---Render only!', flush=True)
        createData = self.args.create_pc  # False
        quantitative_eval = self.args.quantitative_eval  # False

        ##############
        # create save dir


        if self.args.testsavedir != '':
            testsavedir = os.path.join(self.args.testsavedir,
                                       'pc_' + str(self.args.iteration))
        else:

            seqSplitName = self.args.texdir.split('/')[-1]
            if self.args.use_gt_tex:
                testsavedir = os.path.join(self.basedir, self.expname,
                                           'renderonly_{}_{:09d}'.format(
                                               'path',
                                               self.start),
                                           self.args.calib,
                                           str(self.args.resolutionU) + 'x' + str(
                                               self.args.resolutionV),
                                           seqSplitName, 'useGTTex')
            else:
                testsavedir = os.path.join(self.basedir, self.expname,
                                           'renderonly_{}_{:09d}'.format('path',
                                                                         self.start),
                                           self.args.calib,
                                           str(self.args.resolutionU) + 'x' + str(
                                               self.args.resolutionV),
                                           self.args.test_mode)
        print('Save to ', testsavedir)

        os.makedirs(testsavedir, exist_ok=True)

        ##############
        # slurm camera Id and frame distribution

        if createData:
            startcam = (
                    self.args.slurmArrayID % self.cameraReader.numberOfCameras)
            endcam = (
                             self.args.slurmArrayID % self.cameraReader.numberOfCameras) + 1
            camlist = range(startcam, endcam)
            frameRange = range(self.args.train_start_frame,
                               self.args.train_end_frame)
            logFile = open(os.path.join(testsavedir,
                                        'logFile_' + str(startcam).zfill(
                                            3) + '.log'), 'w')
        else:
            if self.i_test_camera != None:  # None
                camlist = self.i_test_camera
            else:
                if self.args.moving_cam:  # True
                    # self.cameraReader.numberOfCameras 360
                    camWindow = int(float(
                        self.cameraReader.numberOfCameras) / 30.0) + 1  # 13
                    '''
                    startcam = self.args.slurmArrayID * camWindow  # self.args.slurmArrayID 0 # startcam0
                    endcam = (
                                     self.args.slurmArrayID + 1) * camWindow  # endcam 13
                    '''
                    startcam = 0
                    endcam = self.cameraReader.numberOfCameras
                    camlist = range(startcam, endcam)  # range(0,13)

                else:
                    startcam = 0
                    endcam = self.cameraReader.numberOfCameras
                    camlist = range(startcam, endcam)

            if not self.args.moving_cam:
                if quantitative_eval:

                    startFrame = self.args.train_start_frame
                    endFrame = self.args.train_end_frame
                    frameRange = range(self.args.train_start_frame,
                                       self.args.train_end_frame, 1)
                else:

                    startFrame = self.args.train_start_frame
                    endFrame = self.args.train_end_frame
                    frameRange = range(self.args.train_start_frame,
                                       self.args.train_end_frame, 10)

            else:
                startFrame = self.args.train_start_frame
                endFrame = self.args.train_end_frame
                frameRange = range(self.args.train_start_frame,
                                   self.args.train_end_frame)

            print('start / end frame: ', startFrame, endFrame, flush=True)

        print('camlist ', camlist, flush=True)
        print('slurmArrayID', self.args.slurmArrayID, flush=True)

        ##############
        # start rendering

        if quantitative_eval:
            psnr_list = []

        print('  Start rendering', testsavedir, flush=True)

        total_rendered_frame = len(frameRange) * len(camlist)
        total_run_time = 0.0

        for f in frameRange:
            for c in camlist:
                if self.args.moving_cam and not self.args.bullet_time:
                    c = int(f % len(camlist))
                print('Processing: ' + str(c).zfill(3) + ' ' + str(f).zfill(6),
                      flush=True)

                slurmTimeout = time.time() - self.slurmStartTime
                terminateToTime = slurmTimeout > (self.jobTimeLimit - 4 * 60)

                print('Time passed ' + str(slurmTimeout) + '/' + str(
                    self.jobTimeLimit - 4 * 60) + ' seconds ' + str(
                    terminateToTime))

                if terminateToTime:
                    print('TIME LIMIT REACHED!')
                    return
                else:
                    if f == frameRange[0] and (createData or len(
                            camlist) >= 1) and not self.args.moving_cam:
                        print('Create dir ', str(c).zfill(3), flush=True)
                        os.makedirs(os.path.join(testsavedir, str(c).zfill(3)),
                                    exist_ok=True)

                    #############################################
                    # camera index


                    if (not self.args.moving_cam) or quantitative_eval:

                        if self.args.moving_cam and self.args.bullet_time:
                            pdb.set_trace()
                            imageFilename = os.path.join(testsavedir,
                                                         'f{:06d}_c{}.png'.format(
                                                             f,
                                                             str(c).zfill(3)))
                        else:
                            imageFilename = os.path.join(testsavedir,
                                                         str(c).zfill(3),
                                                         '{:06d}.png'.format(f))
                        depthFile = os.path.join(testsavedir, str(c).zfill(3),
                                                 'depthMap_' + str(f) + '.obj')
                        if quantitative_eval:
                            gtFilename = os.path.join(testsavedir,
                                                      str(c).zfill(3),
                                                      '{:06d}_gt.png'.format(f))
                    else:
                        if self.args.moving_cam:
                            imageFilename = os.path.join(testsavedir,
                                                         'movingCam_f_' + str(
                                                             f).zfill(
                                                             6) + '_c_' + str(
                                                             c).zfill(
                                                             6) + '.png')
                            depthFile = os.path.join(testsavedir,
                                                     'movingCam_c_' + str(
                                                         c) + '_f_' + str(
                                                         f).zfill(6) + '.obj')
                        else:
                            imageFilename = os.path.join(testsavedir,
                                                         'image_f_' + str(
                                                             f).zfill(
                                                             6) + '.png')
                            depthFile = os.path.join(testsavedir,
                                                     'depthMap_c_' + str(
                                                         c) + '_f_' + str(
                                                         f).zfill(6) + '.obj')

                    proceedWithComputing = True

                    if os.path.isfile(depthFile) and createData:

                        print(str(c).zfill(3) + ' ' + str(f).zfill(
                            6) + ' already exists', flush=True)

                        sanityCheckPC = True
                        sanityCheckPCPassed = True

                        if sanityCheckPC:

                            file = open(depthFile, 'r')
                            lineCount = 0
                            for line in file:
                                lineCount = lineCount + 1
                                length = len(line.split())
                                if length != 8:
                                    print(
                                        'This file does not contain 8 entries in every line!')
                                    logFile.write(
                                        str(c).zfill(3) + ' ' + str(f).zfill(
                                            6) + ' : File does not contain 8 entries in every line \n')
                                    sanityCheckPCPassed = False

                            if lineCount != 4096 * 2:
                                print(
                                    'Line count is wrong! It should be 4096*2 but it is ',
                                    lineCount)
                                logFile.write(
                                    str(c).zfill(3) + ' ' + str(f).zfill(
                                        6) + ' : Line count is wrong! It should be 4096*2 but it is  ' + str(
                                        lineCount) + '\n')
                                sanityCheckPCPassed = False

                        if sanityCheckPCPassed:
                            proceedWithComputing = False

                    if proceedWithComputing:

                        #############################################
                        # load depth

                        returnDictTest = self.nerfDataLoaderTest.load_data(
                            tf.constant(f), tf.constant(str(f)),
                            createData=createData, cameraId=c)



                        camera_i = returnDictTest['camera_i']  # [1,1]
                        whereFG = returnDictTest[
                            'whereForeground']  # [2, # FG pixels]
                        meshPoints = returnDictTest['meshPoints']  # [4852, 3]
                        num_rays = returnDictTest['numRays']  # [1,1]



                        # meshPoints is in meter scale
                        #############################################
                        # render

                        timeRun = time.time()
                        # if createData:
                        if self.args.render_with_tf_function and not self.args.inference:
                            ret_dict = self.renderWithTFFunction(
                                camera_i=camera_i,
                                whereFG=tf.constant(whereFG)[None],
                                meshPoints=meshPoints[None],
                                num_rays=num_rays,
                                isTrain=False)
                        else:

                            ret_dict = self.render(
                                camera_i=camera_i,
                                whereFG=tf.constant(whereFG)[None],
                                meshPoints=meshPoints[None],
                                num_rays=num_rays,
                                isTrain=False)




                        tmp_run_time = time.time() - timeRun
                        print('Run time {:.02f}'.format(tmp_run_time))
                        total_run_time += tmp_run_time

                        #############################################
                        # write colored depth map

                        rgb = ret_dict['rgb_map']  # [1,#FG, 3]

                        if self.args.write_pc or createData:
                            resultPoints = ret_dict['resultPoints'].numpy()[
                                           :numRays, :]
                            acc_map = ret_dict['acc_map'].numpy()[:numRays]

                            file = open(depthFile, 'w')
                            for p in range(0, rgb.shape[0]):
                                file.write('v '
                                           + str(resultPoints[p, 0]) + str(' ')
                                           + str(resultPoints[p, 1]) + str(' ')
                                           + str(resultPoints[p, 2]) + str(' ')
                                           + str(rgb[p, 0]) + str(' ')
                                           + str(rgb[p, 1]) + str(' ')
                                           + str(rgb[p, 2]) + str(' ')
                                           + str(acc_map[p]) + str('\n'))
                            file.close()

                        #############################################
                        # combine FG with white background

                        if not createData:

                            rgbFinal = np.ones(
                                [self.args.resolutionV, self.args.resolutionU,
                                 3], np.float32)


                            rgbFinal[tuple(whereFG)] = rgb[0]

                            rgb8 = to8b(rgbFinal)

                            imageio.imwrite(imageFilename, rgb8)
                            if quantitative_eval:
                                target_s = returnDictTest['target_s']
                                target_s_mask = returnDictTest['target_s_mask']

                                # Compute MSE loss between predicted and true RGB.

                                rgb = tf.constant(
                                    rgb)
                                target_s = target_s

                                img_loss = img2mseL1(rgb, target_s) / float(
                                    self.numGpus)
                                psnr = mse2psnr(img_loss)
                                psnr_list.append(psnr)


                            #############################################




                    # end file exists check
                # end time check passed

                if self.args.moving_cam and not self.args.bullet_time:
                    break
            # end camera loop  #
        # end frame loop
        print('Total average run time {:.05f}'.format(
            total_run_time / total_rendered_frame))

        if quantitative_eval:
            psnr_sum = tf.add_n(psnr_list)
            psnr_sum = tf.add_n(psnr_list)
            final_psnr = psnr_sum / len(psnr_list)
            print('Final PSNR score is {}'.format(final_psnr.numpy()))

        if createData:
            logFile.close()

        return
