class NumpyUtility:
    def __init__(self):
        '''Utilitys made with chatGPT for experimentation, few are working'''

    def compress_dynamic_range(image):
        # Find the 1st and 99th percentiles of the image
        p1, p99 = np.percentile(image, (1, 99))

        # Calculate the range of the image
        img_range = p99 - p1

        # Calculate the compression factor required to fit the image into 8-bit range
        c = 1 / img_range

        # Subtract the 1st percentile from the image and clip it to the [0, 1] range
        new_image = np.clip((image - p1) * c, 0, 1)

        # Convert the image to uint8 format
        new_image = new_image.astype(np.uint8)

        return new_image

    def adjust_luminance(image, mask, amount):
        # Convert the image to LAB color space
        rgb_image = image.astype(np.float32) / 255.0
        xyz_image = np.dot(rgb_image, np.array([[0.412453, 0.357580, 0.180423],
                                                [0.212671, 0.715160, 0.072169],
                                                [0.019334, 0.119193, 0.950227]]))
        xyz_image = np.clip(xyz_image, 0, 1)
        lab_image = np.zeros_like(xyz_image)
        lab_image[..., 0] = 116.0 * np.power(xyz_image[..., 1], 1 / 3.0) - 16.0
        lab_image[..., 1] = 500.0 * (np.power(xyz_image[..., 0], 1 / 3.0) - np.power(xyz_image[..., 1], 1 / 3.0))
        lab_image[..., 2] = 200.0 * (np.power(xyz_image[..., 1], 1 / 3.0) - np.power(xyz_image[..., 2], 1 / 3.0))

        # Apply the luminance adjustment to the masked area
        lab_image[..., 0][mask == 1] = np.clip(lab_image[..., 0][mask == 1] * (1 + amount), 0, 100)

        # Convert the image back to RGB color space
        xyz_image[..., 1] = np.power((lab_image[..., 0] + 16.0) / 116.0, 3)
        xyz_image[..., 0] = np.power((lab_image[..., 0] + 16.0) / 116.0 + lab_image[..., 1] / 500.0, 3)
        xyz_image[..., 2] = np.power((lab_image[..., 0] + 16.0) / 116.0 - lab_image[..., 2] / 200.0, 3)
        rgb_image = np.dot(xyz_image, np.array([[3.240479, -1.537150, -0.498535],
                                                [-0.969256, 1.875992, 0.041556],
                                                [0.055648, -0.204043, 1.057311]]))

        # Convert the image back to the range [0, 255]
        rgb_image = np.clip(rgb_image, 0, 1) * 255.0
        rgb_image = rgb_image.astype(np.uint8)

        return rgb_image

    def gaussian_filter(mask, sigma):
        """
        Apply Gaussian filtering to a binary mask.

        Parameters:
        mask (numpy.ndarray): Binary mask to apply filtering on.
        sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
        numpy.ndarray: Binary mask after Gaussian filtering.
        """
        # Create a Gaussian kernel with the given sigma
        ksize = int(2 * np.ceil(2 * sigma) + 1)
        kernel = np.zeros((ksize, ksize))
        for i in range(ksize):
            for j in range(ksize):
                kernel[i, j] = np.exp(-((i - ksize // 2) ** 2 + (j - ksize // 2) ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)

        # Apply convolution with the kernel
        filtered = np.zeros_like(mask, dtype='float32')
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                roi = mask[max(i - ksize // 2, 0):min(i + ksize // 2 + 1, mask.shape[0]),
                      max(j - ksize // 2, 0):min(j + ksize // 2 + 1, mask.shape[1])]
                filtered[i, j] = np.sum(roi * kernel[:roi.shape[0], :roi.shape[1]])

        return filtered

    def binary_opening(mask, kernel):
        """
        Perform binary morphological opening on a binary mask using a structuring element.

        Parameters:
        mask (numpy.ndarray): Binary mask to perform opening on.
        kernel (numpy.ndarray): Structuring element used for opening.

        Returns:
        numpy.ndarray: Binary mask after morphological opening.
        """
        # Create padding on all sides of the mask
        pad_width = [(kernel.shape[i] // 2, kernel.shape[i] // 2) for i in range(kernel.ndim)]
        mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)

        # Apply morphological erosion using the kernel
        eroded = np.zeros_like(mask_padded)
        for i in range(mask_padded.ndim):
            eroded = np.maximum(eroded, np.apply_along_axis(np.roll, i, mask_padded, shift=-kernel.shape[i] // 2))
        eroded = eroded[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

        # Create padding on all sides of the eroded mask
        pad_width = [(kernel.shape[i] // 2, kernel.shape[i] // 2) for i in range(kernel.ndim)]
        eroded_padded = np.pad(eroded, pad_width, mode='constant', constant_values=0)

        # Apply morphological dilation using the kernel
        dilated = np.zeros_like(eroded_padded)
        for i in range(eroded_padded.ndim):
            dilated = np.minimum(dilated, np.apply_along_axis(np.roll, i, eroded_padded, shift=kernel.shape[i] // 2))
        dilated = dilated[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

        return dilated

    def binary_closing(mask, kernel):
        """
        Perform binary morphological closing on a binary mask using a structuring element.

        Parameters:
        mask (numpy.ndarray): Binary mask to perform closing on.
        kernel (numpy.ndarray): Structuring element used for closing.

        Returns:
        numpy.ndarray: Binary mask after morphological closing.
        """
        # Create padding on all sides of the mask
        pad_width = [(kernel.shape[i] // 2, kernel.shape[i] // 2) for i in range(kernel.ndim)]
        mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)

        # Apply morphological dilation using the kernel
        dilated = np.zeros_like(mask_padded)
        for i in range(mask_padded.ndim):
            dilated = np.minimum(dilated, np.apply_along_axis(np.roll, i, mask_padded, shift=kernel.shape[i] // 2))
        dilated = dilated[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

        # Create padding on all sides of the dilated mask
        pad_width = [(kernel.shape[i] // 2, kernel.shape[i] // 2) for i in range(kernel.ndim)]
        dilated_padded = np.pad(dilated, pad_width, mode='constant', constant_values=0)

        # Apply morphological erosion using the kernel
        eroded = np.zeros_like(dilated_padded)
        for i in range(dilated_padded.ndim):
            eroded = np.maximum(eroded, np.apply_along_axis(np.roll, i, dilated_padded, shift=-kernel.shape[i] // 2))
        eroded = eroded[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

        return eroded

    def create_shadow_mask(image, threshold=80, range_width=50):
        lab_image = np.apply_along_axis(lambda x: np.dot([0.2126, 0.7152, 0.0722], x), 2, image).astype('float64')
        luminance_range = np.max(lab_image) - np.min(lab_image)
        if luminance_range < range_width:
            range_width = luminance_range
        threshold_min = np.min(lab_image) + range_width / 2
        threshold_max = np.max(lab_image) - range_width / 2
        if threshold < threshold_min:
            threshold = threshold_min
        elif threshold > threshold_max:
            threshold = threshold_max
        mask = np.logical_and(lab_image >= threshold - range_width / 2, lab_image <= threshold + range_width / 2)
        if np.sum(mask) == 0:
            return np.zeros_like(mask).astype(float)
        else:
            center_line = np.nanmean(np.where(mask, lab_image, np.nan), axis=0)
            x = np.arange(center_line.shape[0])
            slope = np.zeros(center_line.shape)
            slope[1:-1] = (center_line[2:] - center_line[:-2]) / 2
            slope[0] = slope[1]
            slope[-1] = slope[-2]
            intercept = center_line - slope * x
            x = np.arange(image.shape[1])  # x-coordinates of pixels
            y = np.arange(image.shape[0])  # y-coordinates of pixels
            print(y)
            x, y = np.meshgrid(x, y)  # create 2D arrays of x- and y-coordinates

            # compute distances from each pixel to the shadow line
            dist = np.abs((y[:, :, np.newaxis] - slope[np.newaxis, np.newaxis, :] * x[:, :, np.newaxis]
                           - intercept[np.newaxis, np.newaxis, :]) / np.sqrt(1 + slope[np.newaxis, np.newaxis, :] ** 2))
            print(dist)

            # sigma = np.nanmedian(dist)/0.6745
            # mask = np.exp(-0.5*(dist/sigma)**2)

            return dist

    def highpass_mask(img, cutoff, order=1):
        # Calculate the Fourier transform of the image
        fft_img = np.fft.fft2(img)

        # Shift the zero-frequency component to the center of the spectrum
        fft_img = np.fft.fftshift(fft_img)

        # Construct a highpass filter in the Fourier domain
        x, y = np.meshgrid(np.linspace(-1, 1, img.shape[1]), np.linspace(-1, 1, img.shape[0]))
        r = np.sqrt(x ** 2 + y ** 2)
        hp_filter = 1 - 1 / (1 + (cutoff / r) ** (2 * order))

        # Apply the highpass filter to the Fourier transform of the image
        fft_img_hp = fft_img * hp_filter

        # Shift the zero-frequency component back to the corners of the spectrum
        fft_img_hp = np.fft.ifftshift(fft_img_hp)

        # Calculate the inverse Fourier transform of the highpass-filtered image
        img_hp = np.fft.ifft2(fft_img_hp)

        # Take the absolute value of the real part of the inverse Fourier transform
        img_hp = np.abs(np.real(img_hp))

        # Normalize the highpass-filtered image to the range [0, 1]
        img_hp_norm = img_hp / np.max(img_hp)

        # Invert the highpass-filtered image to create a highpass mask
        mask = 1 - img_hp_norm

        return mask

    def cvtColor(hsv_img):
        # Assuming you have an HSV image loaded in the variable 'hsv_img'
        # Let's convert the image back to RGB color space
        h, s, v = np.split(hsv_img, 3, axis=-1)
        h = h.reshape(h.shape[:2])
        s = s.reshape(s.shape[:2])
        v = v.reshape(v.shape[:2])
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        z = np.zeros_like(h)

        # Set up the RGB channels according to the hue value
        rgb_img = np.dstack((
            np.where((0 <= h) & (h < 60), c,
                     np.where((120 <= h) & (h < 180), z, np.where((240 <= h) & (h < 300), x, m))),
            np.where((300 <= h), c, np.where((60 <= h) & (h < 120), x, np.where((180 <= h) & (h < 240), c, m))),
            np.where((0 <= h) & (h < 360), v, m)
        ))
        return rgb_img

    def saturate(img, amount):
        hsv_img = np.copy(img)
        hsv_img = np.asarray(hsv_img, dtype=np.float32) / 255.0  # Normalize pixel values
        hsv_img = np.clip(hsv_img, 0, 1)  # Clip values to the range [0, 1]
        hsv_img = np.squeeze(cv2.cvtColor(hsv_img, cv2.COLOR_RGB2HSV_FULL))  # Convert to HSV

        # Let's adjust the saturation of the image
        saturation_factor = 1.5  # Adjust this value to your preference
        hsv_img[..., 1] *= saturation_factor

        # Now let's convert the image back to RGB color space
        hsv_img = np.clip(hsv_img, 0, 255)  # Clip values to the range [0, 255]
        hsv_img = np.asarray(hsv_img, dtype=np.uint8)  # Convert back to uint8
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB_FULL)  # Convert back to RGB

        # The resulting image with increased saturation is stored in the 'rgb_img' variable

    def rgb2ycbcr(rgb_img):
        # Create the conversion matrix for RGB to YCbCr
        conv_mat = np.array([[0.299, 0.587, 0.114],
                             [-0.168736, -0.331264, 0.5],
                             [0.5, -0.418688, -0.081312]])

        # Reshape the input image to a 2D array of pixels
        pixels = rgb_img.reshape(-1, 3)

        # Apply the conversion matrix to the pixels
        ycbcr_pixels = np.dot(pixels, conv_mat.T)

        # Reshape the converted pixels back into the original image shape
        ycbcr_img = ycbcr_pixels.reshape(rgb_img.shape)

        # Convert the image data type to uint8 and return it
        return ycbcr_img.astype(np.uint8)

    def ycbcr2rgb(ycbcr_img):
        # Create the conversion matrix for YCbCr to RGB
        conv_mat = np.array([[1.0, 0.0, 1.402],
                             [1.0, -0.344136, -0.714136],
                             [1.0, 1.772, 0.0]])

        # Reshape the input image to a 2D array of pixels
        pixels = ycbcr_img.reshape(-1, 3)

        # Apply the conversion matrix to the pixels
        rgb_pixels = np.dot(pixels, conv_mat.T)

        # Reshape the converted pixels back into the original image shape
        rgb_img = rgb_pixels.reshape(ycbcr_img.shape)

        # Convert the image data type to uint8 and return it
        return rgb_img.astype(np.uint8)

    def adjust_luminance(img, mask, adjustment):
        # Convert the input image to the YCbCr color space
        img = (255.0 * img).astype(np.uint8)
        mask = (255.0 * mask).astype(np.uint8)

        ycbcr_img = rgb2ycbcr(img)
        ycbcr_mask = rgb2ycbcr(mask)
        print(ycbcr_mask)
        print(ycbcr_img)
        plot_histogram(ycbcr_img, title="ycbcr")
        plot_histogram(ycbcr_mask, title="ycmask")
        # Separate the luminance channel (Y)
        y_channel = ycbcr_img[..., 0]
        y_channel_mask = ycbcr_mask[..., 0]

        # Apply the adjustment to the luminance channel using the mask
        y_adjusted = np.clip(y_channel + (adjustment * y_channel_mask), 0, 255).astype(np.uint8)

        # Replace the original luminance channel with the adjusted one
        ycbcr_img[..., 0] = y_adjusted

        # Convert the image back to the RGB color space and return it
        return ycbcr2rgb(ycbcr_img)

    def tonemap_reinhard(image, gamma=2.2, intensity=0.18, light_adapt=0.8):
        """
        Tonemaps the input HDR image using the Reinhard algorithm.

        Args:
            image: The input HDR image as a NumPy array.
            gamma: The gamma correction value to apply to the output image.
            intensity: The target scene brightness.
            light_adapt: The adaptation rate for the brightness.

        Returns:
            The tonemapped output image as a NumPy array.
        """
        # Convert the image to floating point RGB.
        #image = image.astype(np.float32)

        # Compute the log average luminance.
        lum = np.exp(np.mean(np.log(0.0001 + image)))

        # Normalize the image by the log average luminance.
        image /= lum

        # Apply the Reinhard tonemapping algorithm.
        mapped = np.zeros_like(image)
        mapped = intensity * (mapped / np.max(mapped))
        mapped = light_adapt * (mapped * (1 + mapped / np.max(mapped) ** 2)) / (1 + mapped)
        mapped *= lum

        # Apply gamma correction to the tonemapped image.
        mapped = np.power(np.clip(mapped, 0, 1), 1 / gamma)

        # Convert the tonemapped image to 8-bit RGB.
        mapped = (255 * mapped).astype(np.uint8)

        return mapped

    def f(t):
        # Helper function to compute the nonlinear transformation function for the LAB color space
        delta = 6.0 / 29.0
        t_thresh = delta ** 3
        return np.where(t > t_thresh, t ** (1 / 3), (1 / 3) * (29 / 6) ** 2 * t + 4 / 29)

    def numpy_lab2rgb(lab_img):
        XYZ = np.zeros_like(lab_img, dtype=np.float32)
        XYZ[..., 0] = (lab_img[..., 0] + 16.0) / 116.0
        XYZ[..., 1] = (lab_img[..., 1] / 500.0) + XYZ[..., 0]
        XYZ[..., 2] = XYZ[..., 0] - (lab_img[..., 2] / 200.0)
        mask = XYZ > 0.2068966
        XYZ[mask] = XYZ[mask] ** 3
        XYZ[~mask] = (XYZ[~mask] - 16.0 / 116.0) / 7.787
        D50 = np.array([0.9642, 1.0, 0.8249], dtype=np.float32)
        RGB_linear = np.dot(XYZ,
                            np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]],
                                     dtype=np.float32).T)
        RGB_linear = np.clip(RGB_linear, 0.0, 1.0)
        RGB_linear_D50 = RGB_linear / D50
        sRGB = np.where(RGB_linear_D50 <= 0.0031308, 12.92 * RGB_linear_D50,
                        1.055 * (RGB_linear_D50 ** (1.0 / 2.4)) - 0.055)
        sRGB = np.clip(sRGB, 0.0, 1.0)
        return (RGB_linear_D50 * 255).astype(np.uint8)

    def numpy_split_lab(lab_img):
        L, a, b = np.rollaxis(lab_img, axis=-1)
        return L, a, b

    def numpy_merge_lab(l_channel_adjusted, a_channel, b_channel):
        merged_lab = np.dstack((l_channel_adjusted, a_channel, b_channel))
        return merged_lab

    def rgb2lab(img):
        # Convert the RGB image to a float array with values between 0 and 1
        img = img.astype(np.float32) / 255.0

        # Convert the RGB image to the XYZ color space
        # using the transformation matrix specified by the CIE
        # (https://en.wikipedia.org/wiki/CIE_1931_color_space)
        r, g, b = np.split(img, 3, axis=2)
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # Convert the XYZ image to the LAB color space using the D50 white point
        # (https://en.wikipedia.org/wiki/Lab_color_space#Conversion_from_XYZ_to_Lab)
        x /= 0.950456
        z /= 1.088754
        l = 116.0 * f(y) - 16.0
        a = 500.0 * (f(x) - f(y))
        b = 200.0 * (f(y) - f(z))

        # Stack the LAB channels back into a single image and return it
        return np.concatenate((l, a, b), axis=2)

    def apply_luminance_mask_lab(img, mask):
        # Convert the image to the LAB color space
        lab_img = rgb2lab(img)
        print(lab_img)

        lab_mask = rgb2lab(mask)
        print(lab_mask)
        # Split the LAB image into its channels
        l_channel, a_channel, b_channel = numpy_split_lab(lab_img)
        l_channel_mask, a_channel, b_channel = numpy_split_lab(lab_mask)

        # Apply the mask to the luminance channel
        l_channel_adjusted = l_channel + l_channel_mask  # np.where(l_channel_mask > 0, l_channel - 50, l_channel)

        # Merge the adjusted channels back into a LAB image
        lab_img_adjusted = numpy_merge_lab(l_channel_adjusted, a_channel, b_channel)

        # Convert the adjusted image back to the RGB color space and return it
        return numpy_lab2rgb(lab_img_adjusted)

    def rgb2lab(img):
        # Convert the RGB image to a float array with values between 0 and 1
        img = img.astype(np.float32) / 255.0

        # Convert the RGB image to the XYZ color space
        # using the transformation matrix specified by the CIE
        # (https://en.wikipedia.org/wiki/CIE_1931_color_space)
        r, g, b = np.split(img, 3, axis=2)
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # Convert the XYZ image to the LAB color space using the D50 white point
        # (https://en.wikipedia.org/wiki/Lab_color_space#Conversion_from_XYZ_to_Lab)
        x /= 0.950456
        z /= 1.088754
        l = 116.0 * f(y) - 16.0
        a = 500.0 * (f(x) - f(y))
        b = 200.0 * (f(y) - f(z))

        # Stack the LAB channels back into a single image and return it
        return np.concatenate((l, a, b), axis=2)


