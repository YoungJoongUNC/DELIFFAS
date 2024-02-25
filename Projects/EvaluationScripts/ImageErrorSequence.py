
# import the necessary packages
from skimage import measure
import numpy as np
import cv2

###################################################################

def mse(imageA, imageB):

    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, 2)
    errImage = errImage / (3.0)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float( imageA.shape[0] * imageA.shape[1] *imageA.shape[2])

    return err,errImage

###################################################################

def compare_images(imageA, imageB):

    MSE,errImage = mse(imageA, imageB)
    SSIM = measure.compare_ssim(imageA, imageB, multichannel=True)

    return MSE, SSIM,errImage

###################################################################

def compute_sequence_img_error(originalPath, resultPath, storePath, start, end, offset, cameras):

    sumMSE = 0.0
    sumSSIM = 0.0
    counter = 0

    for c in cameras:
        for f in range(start, end, offset):
            originalFile = originalPath + 'images/' + str(c) + '/image_c_' + str(c) + '_f_' + str(f) + '.jpg'
            maskFile = originalPath + 'foregroundSegmentation/' + str(c) + '/image_c_' + str(c) + '_f_' + str(f) + '.jpg'
            resultFile = resultPath  + 'render_c_' + str(c) + '_f_' + str(f) + '.jpg'

            # load the images
            original = cv2.imread(originalFile)
            mask = cv2.imread(maskFile)
            original = original * (mask/255.0)
            original = original.astype('uint8')

            result = cv2.imread(resultFile)

            # compare the images
            MSE, SSIM, errImage  = compare_images(original, result)

            sumMSE = sumMSE + MSE
            sumSSIM = sumSSIM + SSIM
            counter = counter + 1.0

            # write error image
            errImage = errImage.astype(np.uint8)
            errImage = cv2.applyColorMap(errImage, cv2.COLORMAP_JET)

            cv2.imwrite(storePath + 'error_c_' + str(c) + '_f_' + str(f) + '.jpg', errImage)

            print('Frame: ', f, ' MSE: ', MSE, ' SSIM: ', SSIM)

    return sumMSE / counter, sumSSIM/counter

###################################################################


# ORIGINAL_PATH = '/HPS/RTMPC3/work/DeepDynamicCharacters/Dataset/OlekDesert/testing/'
# RESULT_PATH = '/HPS/RTMPC3/work/DeepDynamicCharacters/Dataset/OlekDesert/results/tensorboardLogDeepDynamicCharacters/3642x3667x3668x3976x3978x3992/snapshot_iter_240000/3993_test_0_6900/rendersStaticLight/'
# START = 100
# END = 6000
# OFFSET = 100
# CAMERAS = [46]

ORIGINAL_PATH = '/HPS/RTMPC2/work/DeepCap/Dataset/Magdalena/training/'
RESULT_PATH = '/HPS/RTMPC2/work/DeepCap/Dataset/Magdalena/results/tensorboardLogDeepDynamicCharacters/3413x3415x3423x3597x3598/snapshot_iter_359999/3621_train_100_25900/render/'
STORE_PATH = '/HPS/RTMPC2/work/DeepCap/Dataset/Magdalena/results/tensorboardLogDeepDynamicCharacters/3413x3415x3423x3597x3598/snapshot_iter_359999/3621_train_100_25900/error/'
START = 100
END = 25800
OFFSET = 100
CAMERAS = [0]

avgMSE, avgSSIM = compute_sequence_img_error(ORIGINAL_PATH,RESULT_PATH,STORE_PATH,START,END,OFFSET,CAMERAS)

print('avgMSE: ', avgMSE, ' avgSSIM: ', avgSSIM)