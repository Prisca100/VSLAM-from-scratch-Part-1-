import cv2
import numpy as np
from data import filters
from scipy.ndimage import convolve

def gaussian_blur(size, sigma):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    g = g / g.sum()
    return g

def img_preprocessing(image, smooth=True, smooth_type="gaussian", size=5, sigma=1):
    img = cv2.imread(image)
    if img is None:
        raise Exception("Error: Could not find the image")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if smooth:
        # Apply gaussian blur on image
        if smooth_type == "gaussian":
            # g = filters["gaussian_5x5"]
            g = gaussian_blur(3, 1)
            img = convolve(img, g)
        else:
            print("we don't have the filter yet. Proceeding without blur.")
    return img

def edge_detection_gradient(image, filter_type ="sobel", smooth=True):
    # gradients
    img = img_preprocessing(image, smooth)
    kernel = filters[filter_type]
    grad_x_kernel = np.array(kernel["g_x"])
    grad_y_kernel = np.array(kernel["g_y"])

    # convolution
    grad_x = convolve(img, grad_x_kernel)
    grad_y = convolve(img, grad_y_kernel)

    # Finding magnitude element-wise
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-255 for display purposes
    grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))
    return grad_magnitude, img

def edge_detection_laplace(image, filter_type="laplace", smooth=True):
    img = img_preprocessing(image, smooth)
    kernel = filters[filter_type]
    laplacian = convolve(img, kernel)
    laplacian = np.uint8(255 * laplacian / np.max(laplacian))
    return laplacian, img
    
# TODO
def corner_detection_openCV(image, filter):
    pass