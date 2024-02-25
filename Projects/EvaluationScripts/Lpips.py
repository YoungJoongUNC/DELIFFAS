
###########################################
# imports
###########################################

import sys, glob, imageio
import skimage.metrics
import numpy as np
import torch
import cv2
import os
from lpips_pytorch import LPIPS
from multiprocessing import Pool, cpu_count
from functools import partial
import shutil

###########################################

tmpsrc = '/HPS/mhaberma/nobackup/metric_err/rec'
tmptgt = '/HPS/mhaberma/nobackup/metric_err/gt'
tmperr = '/HPS/mhaberma/nobackup/metric_err/'

###########################################

num_proc = cpu_count()

###########################################

def mae(imageA, imageB):

    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])

    return err

###########################################

def mse(imageA, imageB):

    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2,2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])

    return err,errImage

###########################################

def func(x, g_files, t_files, mask, endFrame):

    H = 600
    W = 496

    psnrs, ssims, mses, maes  = [], [], [], []
    for i in range(len(g_files)):
        if i % num_proc == x:

            frame = int(t_files[i][-9:-4])
            camera  = int(t_files[i][-13:-10])

            if frame >= 100 and frame < endFrame:
                g = imageio.imread(g_files[i]).astype('float32') / 255.
                t = imageio.imread(t_files[i]).astype('float32') / 255.
                g = cv2.resize(g, (t.shape[1], t.shape[0]))

                # g = cv2.blur(g,(13, 13))#todo

                h0, w0 = h, w = g.shape[0], g.shape[1]

                #---------------

                maskName = os.path.join(mask, str(camera), 'image_c_' + str(camera) + '_f_' + str(frame) + '.jpg')

                imgMask = imageio.imread(maskName)
                imgMask = (np.array(imgMask, dtype=np.float32) / 255.0)

                imgMask[np.where(imgMask > 0.5)] = 1.0
                imgMask[np.where(imgMask <= 0.5)] = 0.0

                kernel = np.ones((5, 5), np.uint8)
                imgMask = cv2.erode(imgMask,kernel)
                imgMask = cv2.resize(imgMask, (t.shape[1], t.shape[0]))

                imgMask = np.reshape(imgMask, (imgMask.shape[0], imgMask.shape[1], 1))

                g = g * imgMask + (1.0 - imgMask)
                t = t * imgMask + (1.0 - imgMask)

                # ---------------

                ii, jj = np.where(~(t == 1).all(-1)) # all background pixel coordinates

                try:
                    # bounds for V direction
                    hmin, hmax = np.min(ii), np.max(ii)
                    uu = (H - (hmax + 1 - hmin)) // 2
                    vv = H - (hmax - hmin) - uu
                    if hmin - uu < 0:
                        hmin, hmax = 0, H
                    elif hmax + vv > h:
                        hmin, hmax = h - H, h
                    else:
                        hmin, hmax = hmin - uu, hmax + vv

                    # bounds for U direction
                    wmin, wmax = np.min(jj), np.max(jj)
                    uu = (W - (wmax + 1 - wmin)) // 2
                    vv = W - (wmax - wmin) - uu
                    if wmin - uu < 0:
                        wmin, wmax = 0, W
                    elif wmax + vv > w:
                        wmin, wmax = w - W, w
                    else:
                        wmin, wmax = wmin - uu, wmax + vv

                except ValueError:
                    print(f"target is empty {i}")
                    continue

                #crop images
                g = g[hmin: hmax, wmin: wmax]
                t = t[hmin: hmax, wmin: wmax]

                h, w = g.shape[0], g.shape[1]

                assert (h == H) and (w == W), f"error {hmin} {hmax} {wmin} {wmax} {h0} {w0} {uu} {vv}"

                mseValue, errImg= mse(g, t)
                maeValue = mae(g, t)
                psnr= 10 * np.log10((1 ** 2) / mseValue)
                errImg = (errImg*255.0).astype(np.uint8)
                errImg = cv2.applyColorMap(errImg, cv2.COLORMAP_JET)
                cv2.imwrite(tmperr + str(frame) +'_'+ str(camera) +'_error.png', errImg)
                imageio.imsave("{}/{}_{}_source.png".format(tmpsrc, frame, camera), (g * 255).astype('uint8'))
                imageio.imsave("{}/{}_{}_target.png".format(tmptgt, frame, camera), (t * 255).astype('uint8'))

                psnrs += [psnr]
                ssims += [skimage.metrics.structural_similarity(g, t, channel_axis=2, data_range=1)]
                maes += [maeValue]
                mses += [mseValue]

    return np.asarray(psnrs), np.asarray(ssims), np.asarray(mses), np.asarray(maes)

###########################################

def evaluateErr(output, target, mask, endFrame):

    g_files = sorted(glob.glob(output + f'/*/*.png'))
    t_files = [f'{target}/{g.split("/")[-2]}_{g.split("/")[-1][1:-4]}.png' for g in g_files]

    ###########################################

    with Pool(num_proc) as p:
        results = p.map(partial(func, g_files=g_files, t_files=t_files, mask=mask, endFrame=endFrame), range(num_proc))

    ###########################################
    # PSNR & SSIM

    psnr  = np.concatenate([r[0] for r in results]).mean()
    print(f"PSNR {psnr}", flush=True)
    ssim  = np.concatenate([r[1] for r in results]).mean()
    print(f"SSIM {ssim}", flush=True)
    mse = np.concatenate([r[2] for r in results]).mean()
    print(f"MSE {mse}", flush=True)
    mae = np.concatenate([r[3] for r in results]).mean()
    print(f"MAE {mae}", flush=True)

    ###########################################
    # LPIPS

    lpips = LPIPS(net_type='alex', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmpsrc + '/*_source.png'))
    t_files = sorted(glob.glob(tmptgt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)

    print(f"LPIPS Alex {lpips}", flush=True)

    ###########

    lpips = LPIPS(net_type='vgg', version='0.1')
    if torch.cuda.is_available():
        lpips = lpips.cuda()

    g_files = sorted(glob.glob(tmpsrc + '/*_source.png'))
    t_files = sorted(glob.glob(tmptgt + '/*_target.png'))

    lpipses = []
    for i in range(len(g_files)):
        g = imageio.imread(g_files[i]).astype('float32') / 255.
        t = imageio.imread(t_files[i]).astype('float32') / 255.
        g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3,2,0,1) - 1
        t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3,2,0,1) - 1
        if torch.cuda.is_available():
            g = g.cuda()
            t = t.cuda()
        lpipses += [lpips(g, t).item()]
    lpips = np.mean(lpipses)

    print(f"LPIPS VGG {lpips}", flush=True)

    ###########################################
    # FID

    os.system('python -m pytorch_fid --device cuda {} {}'.format(tmpsrc, tmptgt))

######################################################################################
######################################################################################
#parameters
######################################################################################
######################################################################################

###########################################
# training (Novel view synthesis task)
#
#target = 'PATH TO GT RGB'
#mask = 'PATH TO GT MASK'
#output=[
#'PATH TO YOUR PREDICTED IMAGES'
#]
# endFrame = 17000

###########################################
# testing (Novel pose synthesis task)

target = 'PATH TO GT RGB'
mask = 'PATH TO GT MASK'

output=[
'PATH TO YOUR PREDICTED IMAGES'
]

endFrame = 7000

######################################################################################
######################################################################################

for o in output:

    print('###############################################', flush=True)
    print(o, flush=True)

    os.mkdir(tmptgt)
    os.mkdir(tmpsrc)

    evaluateErr(o, target, mask, endFrame)

    shutil.rmtree(tmptgt)
    shutil.rmtree(tmpsrc)


######################################################################################
######################################################################################