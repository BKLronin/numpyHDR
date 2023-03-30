import numpy as np

try:
    import convolve2d_cython
    available = True
    print("Using compiled Cython Convolve")
except ImportError:
    available = False
    print("Using normal Numpy Convolve")



'''Numpy and PIL implementation of a Mertens Fusion alghoritm
Usage: Instantiate then set attributes:
    input_image = List containing path strings including .jpg Extension
    output_path = String ot Output without jpg ending
    compress_quality = 0-100 Jpeg compression level defaults to 75

    Run function sequence() to start processing.
    Example:

        hdr = numpyHDR.NumpyHDR()

        hdr.input_image = photos/EV- stages/
        hdr.compress_quality = 50
        hdr.output_path = photos/result/
        hdr.sequence()
    returns: Nothing    
'''

def simple_clip(fused,gamma):
    # Apply gamma correction
    #fused = np.clip(fused, 0, 1)
    fused = np.power(fused, 1.0 / gamma)
    #hdr_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    fused = (255.0 * fused).astype(np.uint8)
    #fused = Image.fromarray(fused)

    return fused
def convolve2d(image, kernel):
    # Get the dimensions of the input image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Compute the padding needed to handle boundary effects
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Define generators for row and column indices
    row_indices = range(image_height)
    col_indices = range(image_width)

    # Define a generator expression to generate patches centered at each pixel
    patches = (
        padded_image[
            row : row + kernel_height, col : col + kernel_width
        ]
        for row in row_indices
        for col in col_indices
    )

    # Define a generator expression to generate element-wise products of patches and flipped kernels
    products = (
        patch * np.flip(kernel, axis=(0, 1))
        for patch in patches
    )

    # Define a generator expression to generate convolved values
    convolved_values = (
        product.sum()
        for product in products
    )

    # Reshape the convolved values into an output image
    convolved_image = np.array(list(convolved_values)).reshape((image_height, image_width))

    return convolved_image

def mask(img, center=50, width=20, threshold=0.2):
    '''Mask with sigmoid smooth'''
    mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
    mask = np.where(img > threshold, mask, 1)  # Apply threshold to the mask
    mask =  img * mask
    #plot_histogram(mask, title="mask")
    return mask

def highlightsdrop(img, center=0.7, width=0.2, threshold=0.6, amount=0.08):
    '''Mask with sigmoid smooth targets bright sections'''
    mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
    mask = np.where(img > threshold, mask, 0)  # Apply threshold to the mask
    mask = mask.reshape((img.shape))
    print(np.max(mask))
    img_adjusted  = img - (mask * amount)   # Adjust the image with a user-specified amount
    img_adjusted = np.clip(img_adjusted, 0, 1)

    return img_adjusted

def shadowlift(img, center=0.2, width=0.1, threshold=0.2, amount= 0.05):
    '''Mask with sigmoid smooth targets bright sections'''
    mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
    mask = np.where(img < threshold, mask, 0)  # Apply threshold to the mask
    mask = mask.reshape((img.shape))
    print(np.max(mask))
    img_adjusted  = (mask * amount) + img   # Adjust the image with a user-specified amount
    img_adjusted = np.clip(img_adjusted, 0, 1)

    return img_adjusted
def blur(image, amount=1):
    # Define a kernel for sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    # Apply the kernel to each channel of the image using convolution
    #blurred = convolve2d(image, kernel)
    kernel = kernel.astype(np.float64)
    #image= image.astype(np.float64)
    if available:
        blurred = convolve2d_cython.convolve2d(image, kernel)
    else:
        blurred = convolve2d(image, kernel)
    # Add the original image to the sharpened image with a weight of the sharpening amount
    sharpened = image + amount * (image - blurred)

    sharpened = np.clip(sharpened, 0, 1)

    # Crop the output image to match the input size
    #sharpened = sharpened.reshape(image.shape)

    return sharpened



def mertens_fusion(stack, gamma:float =1, contrast_weight:float =1 ,blurred: bool = False) -> bytearray:
    """Fuse multiple exposures into a single HDR image using the Mertens algorithm.

    Args:
        image_paths: A list of paths to input images.
        gamma: The gamma correction value to apply to the input images.
        contrast_weight: The weight of the local contrast term in the weight map computation.
        blurred: Helps making transitions for the weights smoother but increases provessing time x2

    Returns:
        The fused HDR image.
    """

    images = []
    for array in stack:
        #Incoming arrays in 255 er range
        img = np.array(array).astype(np.float64) / 255.0
        img = np.power(img, gamma)
        images.append(img)

    # Compute the weight maps for each input image based on the local contrast.
    weight_maps = []
    kernel = np.array([[1, 2, 1], [2, -11, 2], [1, 2, 1]])

    for img in images:
        threshold_h = .99
        threshold_l = .1
        # Apply thresholding to filter out overexposed portions of the image
        img = np.where(img > threshold_h, 0.99, img)
        gray = np.dot(img, [0.2989, 0.5870, 0.1140])
        if blurred:
            gray = blur(gray, 1)
        #kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        kernel = kernel.astype(np.float64)
        if available:
            laplacian = np.abs(convolve2d_cython.convolve2d(gray, kernel))
        else:
            laplacian = np.abs(convolve2d(gray, kernel))
        weight = np.power(laplacian, contrast_weight)
        weight_maps.append(weight)

    # Normalize the weight maps.
    total_weight = sum(weight_maps)
    weight_maps = [w / total_weight for w in weight_maps]

    # Compute the fused HDR image by computing a weighted sum of the input images.
    fused = np.zeros(images[0].shape, dtype=np.float64)
    for i, img in enumerate(images):
        fused += weight_maps[i][:, :, np.newaxis] * img

    return fused

def compress_dynamic_range(image):
    '''Compress dynamic range based on percentile'''
    # Find the 1st and 99th percentiles of the image
    p1, p99 = np.percentile(image, (0, 99))

    # Calculate the range of the image
    img_range = p99 - p1

    # Calculate the compression factor required to fit the image into 8-bit range
    c = 1 / img_range

    # Subtract the 1st percentile from the image and clip it to the [0, 1] range
    new_image = np.clip((image - p1) * c, 0, 1)

    return new_image

def compress_dynamic_range_histo(image, new_min=0.01, new_max=0.99):
    """Compress the dynamic range of an image using histogram stretching.

    Args:
        image: A numpy array representing an image.
        new_min: The minimum value of the new range.
        new_max: The maximum value of the new range.
    Returns:
        The compressed image.
    """
    # Calculate the histogram of the image.
    hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 1))

    # Calculate the cumulative distribution function (CDF) of the histogram.
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # normalize to [0, 1]

    # Interpolate the CDF to get the new pixel values.
    new_pixels = np.interp(image.ravel(), bins[:-1], cdf * (new_max - new_min) + new_min)

    # Reshape the new pixel values to the shape of the original image.
    new_image = new_pixels.reshape((image.shape[0], image.shape[1], image.shape[2]))

    return new_image

def process(stack, gain: float = 1, weight: float = 1, gamma: float = 1, post: bool = True, blurred: bool = True):
    '''Processes the stack that contains a list of arrays form the camera into a PIL compatible clipped output array
    Args:
        stack : input list with arrays
        gain : low value low contrast, high value high contrast and brightness
        weight: How much the extracted portions of each image gets allpied to to the result "HDR effect intensity"
        gamma: Post fusion adjustment of the gamma.
        post: Enable or disable effects applied after the fusion True or False, default True
            shadowlift = slightly lifts the shadows
            Args:
                center: position of the filter dropoff
                width: range of the gradient, softness
                threshold: sets the threshhold form 0 to 1 0.1= lowest blacks....
                amount: How much the shadows should be lifted. Values under 0.1 seem to be good.
            returns:
                Hdr image with lifted blacks clipped to 0,1 range

            compress dynamic range:
                Tries to fit the image better into the available range. Less loggy image.
    Returns:
        HDR Image as PIL compatible array.

    '''

    hdr_image = mertens_fusion(stack ,gain, weight, blurred)
    if post == True:
        #hdr_image = self.highlightsdrop(hdr_image)
        hdr_image = shadowlift(hdr_image)
        hdr_image = compress_dynamic_range(hdr_image)
        #hdr_image = self.compress_dynamic_range_histo(hdr_image)

    hdr_image = simple_clip(hdr_image,gamma)
    return hdr_image



