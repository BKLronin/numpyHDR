import numpy as np
from PIL import Image


#import matplotlib.pyplot as plt

class NumpyHDR:
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

    def __init__(self):
        self.input_image: list = []
        self.output_path: str = '/'
        self.compress_quality: int = 75

    def plot_histogram(self, image, title="Histogram", bins=256):
        """Plot the histogram of an image.

        Args:
            image: A numpy array representing an image.
            title: The title of the plot.
            bins: The number of bins in the histogram.
        """
        fig, ax = plt.subplots()
        ax.hist(image.ravel(), bins=bins, color='gray', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        plt.show()
    ### Experimental functions above this line. chatGPT sketches

    def simple_clip(self, fused,gamma):
        # Apply gamma correction
        #fused = np.clip(fused, 0, 1)
        fused = np.power(fused, 1.0 / gamma)
        #hdr_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
        fused = (255.0 * fused).astype(np.uint8)
        #fused = Image.fromarray(fused)

        return fused

    def convolve2d(self, image, kernel):
        """Perform a 2D convolution on the given image with the given kernel.

        Args:
            image: The input image to convolve.
            kernel: The kernel to convolve the image with.

        Returns:
            The convolved image.
        """
        # Get the dimensions of the image and kernel.
        image_height, image_width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape[:2]

        # Compute the amount of padding to add to the image.
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the image with zeros.
        padded_image = np.zeros(
            (image_height + 2 * pad_height, image_width + 2 * pad_width),
            dtype=np.float32,
        )
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

        # Flip the kernel horizontally and vertically.
        flipped_kernel = np.flipud(np.fliplr(kernel))

        # Convolve the padded image with the flipped kernel.
        convolved_image = np.zeros_like(image, dtype=np.float32)
        for row in range(image_height):
            for col in range(image_width):
                patch = padded_image[
                    row : row + kernel_height, col : col + kernel_width
                ]
                product = patch * flipped_kernel
                convolved_image[row, col] = product.sum()

        return convolved_image

    def mask(self, img, center=50, width=20, threshold=0.2):
        '''Mask with sigmoid smooth'''
        mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
        mask = np.where(img > threshold, mask, 1)  # Apply threshold to the mask
        mask =  img * mask
        #plot_histogram(mask, title="mask")
        return mask

    def highlightsdrop(self, img, center=0.7, width=0.2, threshold=0.6, amount=0.08):
        '''Mask with sigmoid smooth targets bright sections'''
        mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
        mask = np.where(img > threshold, mask, 0)  # Apply threshold to the mask
        mask = mask.reshape((img.shape))
        print(np.max(mask))
        img_adjusted  = img - (mask * amount)   # Adjust the image with a user-specified amount
        img_adjusted = np.clip(img_adjusted, 0, 1)

        return img_adjusted

    def shadowlift(self, img, center=0.2, width=0.1, threshold=0.2, amount= 0.05):
        '''Mask with sigmoid smooth targets bright sections'''
        mask = 1 / (1 + np.exp((center - img) / width))  # Smooth gradient mask
        mask = np.where(img < threshold, mask, 0)  # Apply threshold to the mask
        mask = mask.reshape((img.shape))
        print(np.max(mask))
        img_adjusted  = (mask * amount) + img   # Adjust the image with a user-specified amount
        img_adjusted = np.clip(img_adjusted, 0, 1)

        return img_adjusted

    def mertens_fusion(self, image_paths, gamma=2.2, contrast_weight=0.2):
        """Fuse multiple exposures into a single HDR image using the Mertens algorithm.

        Args:
            image_paths: A list of paths to input images.
            gamma: The gamma correction value to apply to the input images.
            contrast_weight: The weight of the local contrast term in the weight map computation.

        Returns:
            The fused HDR image.
        """
        # Load the input images and convert them to floating-point format.
        images = []
        for path in image_paths:
            img = Image.open(path)
            img = img.resize((1280, 720))
            img = np.array(img).astype(np.float32) / 255.0
            img = np.power(img, gamma)

            images.append(img)

        # Compute the weight maps for each input image based on the local contrast.
        weight_maps = []

        for img in images:
            gray = np.dot(img, [0.2989, 0.5870, 0.1140])
            kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])
            laplacian = np.abs(self.convolve2d(gray, kernel))
            weight = np.power(laplacian, contrast_weight)
            weight_maps.append(weight)

        # Normalize the weight maps.
        total_weight = sum(weight_maps)
        weight_maps = [w / total_weight for w in weight_maps]

        # Compute the fused HDR image by computing a weighted sum of the input images.
        fused = np.zeros(images[0].shape, dtype=np.float32)
        for i, img in enumerate(images):
            fused += weight_maps[i][:, :, np.newaxis] * img
        #print(fused)

        return fused

    def compress_dynamic_range(self, image):
        # Find the 1st and 99th percentiles of the image
        p1, p99 = np.percentile(image, (0, 99))

        # Calculate the range of the image
        img_range = p99 - p1

        # Calculate the compression factor required to fit the image into 8-bit range
        c = 1 / img_range

        # Subtract the 1st percentile from the image and clip it to the [0, 1] range
        new_image = np.clip((image - p1) * c, 0, 1)

        return new_image

    def compress_dynamic_range_histo(self, image, new_min=0.01, new_max=0.99):
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

    def open_image(filename):
        # Open the image file in binary mode
        with open(filename, 'rb') as f:
            # Read the binary data from the file
            binary_data = f.read()

        # Convert the binary data to a 1D numpy array of uint8 type
        image_array = np.frombuffer(binary_data, dtype=np.uint8)

        # Reshape the 1D array into a 2D array with the correct image shape
        # (Assuming a 3-channel RGB image with shape (height, width))
        height = int.from_bytes(binary_data[16:20], byteorder='big')
        width = int.from_bytes(binary_data[20:24], byteorder='big')
        image_array = image_array[24:].reshape((height, width, 3))

        return image_array

    def sequence(self, gain: float = 0.8, weight: float = 0.5, gamma: float = 1, post: bool = True):
        '''gain setting : the higher the darker, good range from 0.4- 1.0'''
        print(self.input_image)
        hdr_image = self.mertens_fusion(self.input_image ,gain, weight)

        if post == True:

            #hdr_image = self.highlightsdrop(hdr_image)
            hdr_image = self.shadowlift(hdr_image)
            hdr_image = self.compress_dynamic_range(hdr_image)
            #hdr_image = self.compress_dynamic_range_histo(hdr_image)

        hdr_image = self.simple_clip(hdr_image,gamma)
        image = Image.fromarray(hdr_image)
        image.save(f"{self.output_path}_hdr.jpg", quality=self.compress_quality)


