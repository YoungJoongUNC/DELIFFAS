# import the necessary packages

from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2

#############################

def mse(imageA, imageB):

    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2,2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])

    return err,errImage

#############################

def compare_images(imageA, imageB, title):

    m ,errImageA= mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB, multichannel=True)

    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.8f" % (m, s))

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(errImageA, cmap = 'jet')
    plt.axis("off")

    plt.show()

    return errImageA

# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("D:/original.jpg")
mask = cv2.imread("D:/mask.jpg")
original = original * (mask/255.0)
original = original.astype('uint8')

light = cv2.imread("D:/light.jpg")
albedo = cv2.imread("D:/albedo.jpg")

original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
light= cv2.cvtColor(light, cv2.COLOR_BGR2RGB)
albedo=cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("light", light), ("albedo", albedo)

# loop over the images
for (i, (name, image)) in enumerate(images):
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image)
	plt.axis("off")

# show the figure
plt.show()

# compare the images
errImageAlbedo = compare_images(original, albedo, "Original vs. Albedo")
errImageLight  = compare_images(original, light, "Original vs. light")

errImageAlbedo = errImageAlbedo.astype(np.uint8)
errImageLight = errImageLight.astype(np.uint8)

errImageAlbedo = cv2.applyColorMap(errImageAlbedo, cv2.COLORMAP_JET)
errImageLight = cv2.applyColorMap(errImageLight, cv2.COLORMAP_JET)

cv2.imwrite('D:/errImgAlbedo.jpg', errImageAlbedo)
cv2.imwrite('D:/errImgLight.jpg', errImageLight)