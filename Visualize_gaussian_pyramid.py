import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic image for demonstration (simple gradient with features)
synthetic_image = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(synthetic_image, (64, 64), 40, 255, -1)  # Add a circle
cv2.rectangle(synthetic_image, (160, 160), (210, 210), 255, -1)  # Add a rectangle
synthetic_image += np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # Add gradient

# Display the synthetic image
plt.figure(figsize=(6, 6))
plt.title("Synthetic Input Image")
plt.imshow(synthetic_image, cmap='gray')
plt.axis('off')
plt.show()

# Function to create Gaussian blurred images for an octave
def generate_gaussian_pyramid(image, num_scales, base_sigma, k):
    gaussian_images = [image]  # Start with the original image
    for i in range(1, num_scales):
        sigma = base_sigma * (k ** i)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gaussian_images.append(blurred)
    return gaussian_images

# Function to compute DoG from a Gaussian pyramid
def compute_dog(gaussian_pyramid):
    dog_images = []
    for i in range(1, len(gaussian_pyramid)):
        dog = cv2.subtract(gaussian_pyramid[i], gaussian_pyramid[i - 1])
        dog_images.append(dog)
    return dog_images

# Visualization of Gaussian pyramid and DoG images for one octave
def visualize_octave(gaussian_pyramid, dog_images, octave_index):
    num_gaussian = len(gaussian_pyramid)
    num_dog = len(dog_images)
    
    fig, axes = plt.subplots(2, max(num_gaussian, num_dog), figsize=(20, 8))
    fig.suptitle(f"Octave {octave_index}: Gaussian Pyramid and DoG", fontsize=16)
    
    # Display Gaussian pyramid
    for i, img in enumerate(gaussian_pyramid):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Gaussian Scale {i}")
        axes[0, i].axis('off')

    # Display DoG images
    for i, img in enumerate(dog_images):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"DoG {i}")
        axes[1, i].axis('off')

    # Hide unused subplots
    for ax in axes.flatten()[len(gaussian_pyramid):]:
        ax.axis('off')
    plt.show()

# Parameters
base_sigma = 1.6  # Initial sigma value
num_scales = 5    # Number of Gaussian levels per octave
k = 2 ** (1 / (num_scales - 1))  # Multiplicative factor for sigma between levels
num_octaves = 3   # Number of octaves to visualize

# Generate and visualize octaves
current_image = synthetic_image.copy()
for octave_index in range(num_octaves):
    # Generate Gaussian pyramid for the current octave
    gaussian_pyramid = generate_gaussian_pyramid(current_image, num_scales, base_sigma, k)
    # Compute DoG images for the octave
    dog_images = compute_dog(gaussian_pyramid)
    # Visualize Gaussian and DoG for this octave
    visualize_octave(gaussian_pyramid, dog_images, octave_index)
    # Downsample image for the next octave
    current_image = cv2.pyrDown(current_image)  # Reduce size by half
