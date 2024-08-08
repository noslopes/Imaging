import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage, datasets
from scipy.ndimage import median_filter

img = Image.open('IMG_3362.jpg')
img1 = Image.open('cat.jpg')
img2 = Image.open('bridge.jpg')

def Gaussian(img):
    img = np.array(img)

    sigma = 1 
    img_filtered = gaussian_filter(img, sigma)

    img_filtered = Image.fromarray(img_filtered)
    img_filtered.save('filtered_image.png')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

def SobelEdgeDetect(img):
    ia_32 = np.array(img, dtype=np.int32)
    print(ia_32.flatten()[:5])

    sobel_h = ndimage.sobel(ia_32, 0)
    sobel_v = ndimage.sobel(ia_32, 1)
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    magnitude *= 255.0 / np.max(magnitude)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.gray()
    axs[0, 0].imshow(ia_32)
    axs[0, 1].imshow(sobel_h)
    axs[1, 0].imshow(sobel_v)
    axs[1, 1].imshow(magnitude)
    titles = ["original", "horizontal", "vertical", "magnitude"]
    for i, ax in enumerate(axs.ravel()):
        ax.set_title(titles[i])
        ax.axis("off")
    plt.show()

def SPDenoiser(img):
    img = np.array(img)

    # Apply median filter
    filtered_image2 = median_filter(img, size=3)
    filtered_image = median_filter(img, size=1)

    # Display the original and filtered images
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Filtered Image')
    ax[2].imshow(filtered_image2, cmap='gray')
    ax[2].set_title('Filtered Image 2')
    plt.show()

def is_blue(pixel):
    # Define the range for medium blue to light blue in BGR format
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 200, 200])
    return np.all(lower_blue <= pixel) and np.all(pixel <= upper_blue)

def bluedenoiser(img):
    img = np.array(img)

    test = []
    y = 1
    for x in range(1, img.shape[1] - 1):
        pxl = [
                img[y-1, x-1], img[y-1, x], img[y-1, x+1],
                img[y, x-1], img[y, x+1],
                img[y+1, x-1], img[y+1, x], img[y+1, x+1]
            ]
        if is_blue(pxl):
            test.append(1)
        else:
            test.append(0)

    print(test)


Images = []
# Images.append(img)
# Images.append(img1)
Images.append(img2)

for i in Images:
    bluedenoiser(i)