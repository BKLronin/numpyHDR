import cv2
import numpy as np
import skimage

def gamma_correction(image, gamma):
    adjust = skimage.exposure.rescale_intensity(image)
    adjust = skimage.exposure.adjust_gamma(adjust,gamma)
    return adjust

def fusionhdr(images: list):
    # Load the set of images

    img_array = []
    for img in images:
        img_array.append(cv2.imread(img))

    # Create Mertens object and set its parameters
    mertens = cv2.createMergeMertens()
    mertens.setExposureWeight(0.9)
    mertens.setSaturationWeight(0.5)

    # Merge the images using the Mertens algorithm
    fused_image = mertens.process(img_array)
    fused = np.clip(fused_image, 0, 1)

    # Save the fused image

    return fused

list = ['hdr/webcam20_3_2023_ev1.jpg', 'hdr/webcam20_3_2023_ev0.jpg','hdr/webcam20_3_2023_ev-1.jpg']

#TODO add image open
hdr_image = fusionhdr(list)
hdr_image = gamma_correction(hdr_image,0.7)
cv2.imwrite('fused_image2.jpg', hdr_image * 255)
