import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import os
from skimage.io import imsave


def image_read(path, mode="RGB"):
    img_BGR = cv2.imread(path).astype("float32")
    assert mode == "RGB" or mode == "GRAY" or mode == "YCrCb", "mode error"
    if mode == "RGB":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == "GRAY":
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == "YCrCb":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def norm(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def plot_images(*images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]

    for i, img in enumerate(images):
        if len(img.shape) == 2 or img.shape[2] == 1:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)

        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def check(img):
    min_value = np.min(img)
    max_value = np.max(img)
    mean_value = np.mean(img)
    std_value = np.std(img)
    print(
        f"shape: {img.shape}, scale: [{min_value:.2f}, {max_value:.2f}], mean: {mean_value:.2f}, std: {std_value:.2f}"
    )


def crop(img, scale):
    h = img.shape[0]
    w = img.shape[1]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w]


def compute_gradient(image):
    gradient_x = sobel(image, axis=0)

    gradient_y = sobel(image, axis=1)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_x, gradient_y, gradient_magnitude


def ensure_even_dimensions(img):
    h, w = img.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return img[:new_h, :new_w]

def normalize_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    return image.astype(np.uint8)
def display_images(data_IR, data_VIS, fi, title_IR="Infrared (IR)", title_VIS="Visible (VI)", title_Fuse="Fused Image"):
    """
    显示红外图像、可见光图像和融合图像。
    
    参数:
    data_IR (Tensor): 红外图像数据
    data_VIS (Tensor): 可见光图像数据
    fi (ndarray): 融合图像
    title_IR (str): 红外图像标题
    title_VIS (str): 可见光图像标题
    title_Fuse (str): 融合图像标题
    """
    # 创建图形和子图
    plt.figure(figsize=(15, 5))

    # 显示红外图像
    plt.subplot(1, 3, 1)
    plt.title(title_IR)
    plt.imshow(data_IR.cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")

    # 显示可见光图像
    plt.subplot(1, 3, 2)
    plt.title(title_VIS)
    plt.imshow(data_VIS.cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")

    # 显示融合图像
    plt.subplot(1, 3, 3)
    plt.title(title_Fuse)
    plt.imshow(fi, cmap="gray")
    plt.axis("off")

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
def display_fused_image(img_name, data_IR, data_VIS_color, data_VIS, fi, fi_img_color):
    """
    显示红外图像、可见光图像和彩色融合图像。
    
    参数:
    img_name (str): 图像文件名
    data_IR (Tensor): 红外图像数据
    data_VIS_color (PIL Image): 可见光彩色图像
    data_VIS (Tensor): 可见光灰度图像数据
    fi (ndarray): 灰度融合图像
    fi_img_color (ndarray): 彩色融合图像
    """
    
    # 检查融合图像是否正常
    if np.max(fi) == 0:
        print(f"Warning: Fused image {img_name} is completely black!")
        return

    # 显示图像
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示红外图像
    axes[0].imshow(np.squeeze(data_IR.cpu().numpy()), cmap='gray')
    axes[0].set_title("Infrared (IR)")
    axes[0].axis('off')

    # 显示可见光图像
    axes[1].imshow(data_VIS_color)
    axes[1].set_title("Visible (VI)")
    axes[1].axis('off')

    # 显示彩色融合图像
    axes[2].imshow(fi_img_color)
    axes[2].set_title("Fused Image (Color)")
    axes[2].axis('off')

    plt.show()
def ycbcr_to_rgb(ycbcr_img):
    """
    Convert YCbCr image to RGB image.

    Parameters:
        ycbcr_img (np.ndarray): YCbCr image with shape (height, width, 3).

    Returns:
        np.ndarray: RGB image with shape (height, width, 3).
    """
    # Define the transformation matrix from YCbCr to RGB
    transform_matrix = np.array(
        [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]]
    )

    # Define the offset for YCbCr
    offset = np.array([-16, -128, -128])

    # Normalize input YCbCr image to [0, 255]
    ycbcr_img = ycbcr_img.astype(np.float32)
    ycbcr_img = ycbcr_img - offset

    # Apply the transformation matrix
    rgb_img = np.dot(ycbcr_img, transform_matrix.T)

    # Clip values to the [0, 255] range and convert to uint8
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    return rgb_img


def display_ycbcr_image(ycbcr_img):
    """
    Display YCbCr image by converting it to RGB.

    Parameters:
        ycbcr_img (np.ndarray): YCbCr image with shape (height, width, 3).
    """
    rgb_img = ycbcr_to_rgb(ycbcr_img)

    plt.imshow(rgb_img)
    plt.title("Converted RGB Image from YCbCr")
    plt.axis("off")
    plt.show()


def image_read_cv2(path, mode="RGB"):
    img_BGR = cv2.imread(path).astype("float32")
    assert mode == "RGB" or mode == "GRAY" or mode == "YCrCb", "mode error"
    if mode == "RGB":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == "GRAY":
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == "YCrCb":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def img_save(image, imagename, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Convert image to 8-bit if it's a floating point grayscale image

    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize the image to 0-255
        image = (
            255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        ).astype(np.uint8)

    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)


__all__ = ["image_read", "norm", "plot_images", "check", "crop", "compute_gradient"]
