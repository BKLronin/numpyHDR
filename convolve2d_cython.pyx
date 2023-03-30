# Import necessary packages
import cython
import numpy as np
cimport numpy as np


# Declare types for function arguments and variables
cpdef np.ndarray[np.float64_t, ndim=2] convolve2d(np.ndarray[np.float64_t, ndim=2] image,
                                                  np.ndarray[np.float64_t, ndim=2] kernel):

    cdef int image_height, image_width, kernel_height, kernel_width, pad_height, pad_width, row, col
    cdef np.ndarray[np.float64_t, ndim=2] padded_image, convolved_image, patch, product

    # Get the dimensions of the input image and kernel
    # Get the dimensions of the input image and kernel
    image_height, image_width = int(image.shape[0]), int(image.shape[1])
    kernel_height, kernel_width = int(kernel.shape[0]), int(kernel.shape[1])

    # Compute the padding needed to handle boundary effects
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output image
    convolved_image = np.zeros((image_height, image_width), dtype=np.float64)

    # Loop over each pixel in the output image and compute the convolved value
    for row in range(image_height):
        for col in range(image_width):
            # Extract the patch centered at the current pixel
            patch = padded_image[row : row + kernel_height, col : col + kernel_width]

            # Compute the element-wise product of the patch and the flipped kernel
            product = patch * np.flip(kernel, axis=(0, 1))

            # Compute the sum of the element-wise products
            convolved_value = np.sum(product)

            # Store the convolved value in the output image
            convolved_image[row, col] = convolved_value

    return convolved_image