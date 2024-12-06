# necessary imports
import cv2 as cv
import numpy as np
from skimage import color
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def generate_gaussian_kernel(chunk_size, sigma):
    """
    Create a 2D Gaussian kernel.
    """
    x = np.arange(-chunk_size // 2 + 1, chunk_size // 2 + 1)
    y = np.arange(-chunk_size // 2 + 1, chunk_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def compute_structure_tensor(image, chunk_size=5, sigma=2):
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    
    height, width = image.shape[:2]
    tensor_list = []
    gaussian_kernel = generate_gaussian_kernel(chunk_size, sigma)
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk_grad_x = grad_x[y:y + chunk_size, x:x + chunk_size]
            chunk_grad_y = grad_y[y:y + chunk_size, x:x + chunk_size]
            
            J_xx = np.sum(chunk_grad_x ** 2 * gaussian_kernel)
            J_yy = np.sum(chunk_grad_y ** 2 * gaussian_kernel)
            J_xy = np.sum(chunk_grad_x * chunk_grad_y * gaussian_kernel)
            
            tensor_list.append(((x, y), np.array([[J_xx, J_xy], [J_xy, J_yy]])))
    return tensor_list

def calculate_orientation(tensor):
    # lowkey just like self-explanitory lock in
    _, eigvecs = np.linalg.eigh(tensor)  # Eigenvectors
    dominant_orientation = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])  # Angle of the largest eigenvector
    return np.degrees(dominant_orientation)

def orient_hsv(image, tensor_list, chunk_size=5, mode="all"):
    assert len(image.shape) == 2, "Image should be 2d grayscale"
    # Initialize the HSV image
    height, width = image.shape
    hsv_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Normalize image for intensity (0 to 1)

    normalized_image = image
    
    # normalized_image = image/np.max(image)
    
    for (x, y), tensor in tensor_list:
        # Calculate orientation
        orientation = calculate_orientation(tensor)

        hue = (orientation % 180) / 180.0  # Normalize angle to [0, 1] for hue
        
        # indices for slicing for chunk
        chunk_x = slice(x, x + chunk_size)
        chunk_y = slice(y, y + chunk_size)

        # How aligned the vectors are
        local_coherence = calculate_coherence(tensor)

        assert local_coherence < 1, "Coherence greater than 1"
        # print(local_coherence)

        if mode == 'all':
            hsv_image[chunk_y, chunk_x, 0] = hue  # Hue: Orientation
            hsv_image[chunk_y, chunk_x, 1] = local_coherence # Saturation: Coherence
            hsv_image[chunk_y, chunk_x, 2] = normalized_image[chunk_y, chunk_x]  # Value: Intensity
            
        elif mode == 'coherence':
            hsv_image[chunk_y, chunk_x, 0] = 0
            hsv_image[chunk_y, chunk_x, 1] = 0
            hsv_image[chunk_y, chunk_x, 2] = local_coherence # * np.mean(normalized_image[chunk_y, chunk_x])
        elif mode == 'angle':
            hsv_image[chunk_y, chunk_x, 0] = hue  # Hue: Orientation
            hsv_image[chunk_y, chunk_x, 1] = 1
            hsv_image[chunk_y, chunk_x, 2] = 1
        else:
            assert False, "Invalid mode"
    
    # Gaussian blur so it looks nice
    hsv_image[:, :, 0] = gaussian(hsv_image[:, :, 0], sigma=3)
    hsv_image[:, :, 1] = gaussian(hsv_image[:, :, 1], sigma=3)
    if mode == 'coherence':
        hsv_image[:, :, 2] = gaussian(hsv_image[:, :, 2], sigma=3)

    
    rgb_image = cv.cvtColor((hsv_image * 255).astype(np.uint8), cv.COLOR_HSV2RGB)
    return rgb_image

def calculate_coherence(tensor):
    # Eigenvalue-based coherence
    eigvals, _ = np.linalg.eigh(tensor)
    lambda1, lambda2 = eigvals  # Sorted eigenvalues (smallest first)
    return np.abs((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-5))
    # np.abs((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-5))
    # return coherence # shouldnt coherence always be positive

# TODO: fundamentally change coherence and orientation calculations: instead of breaking into
# chunks, and then getting chunk-wide values, use this process:
# 1. calculate x and y gradient with gradient filter - add menu to choose type of computation and sigma (if applicable)
# 2. convolve 2-channel (I_x, I_y) to 3-channel (gaussian - weighted Jxx, Jxy, Jyy) image-wide, and add choice for sigma
# 3. black magic from wikipedia
# S_w(p) = [[mu_20, mu_11], [mu_11, mu_02]]
# k20 = mu20 - mu02 + 2i*mu11 = (lambda1 - lambda2)exp(2i*phi)
# k11 = mu20 + mu02 = lambda1 + lambda2 (trace of matrix = sum of eigenvectors)
# from this, |k20|/k11 = coherence, and atan2(im(k20), re(k20)) = orientation
#
# 2 important hyperparameters: inner scale and outer scale
# The inner scale determines the frequency range over which the orientation is estimated, and is the 