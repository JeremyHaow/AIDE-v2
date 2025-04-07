import numpy as np
from PIL import Image
import random
from scipy import stats
from skimage import feature, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, ToTensor

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    cropped_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray)
        if metric == 'sd':
            m = np.std(crop_gray / 255.0)
        elif metric == 'ghe':
            m = histogram_entropy_response(crop_gray / 255.0)
        elif metric == 'le':
            m = local_entropy_response(crop_gray)
        elif metric == 'ac':
            m = autocorrelation_response(crop_gray / 255.0)
        elif metric == 'td':
            m = texture_diversity_response(crop_gray / 255.0)
        images.append((crop, m))

    images.sort(key=lambda x: x[1], reverse=True)
    
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]

    repeat_images = texture_images.copy()
    while len(texture_images) < n:
        texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])

    return texture_images


def autocorrelation_response(image_array):
    """
    Calculates the average autocorrelation of the input image.
    """
    f = np.fft.fft2(image_array, norm='ortho')
    power_spectrum = np.abs(f) ** 2
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()
    acf = np.mean(acf)

    return acf

def histogram_entropy_response(image):
    """
    Calculates the entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    prob_dist = histogram / histogram.sum()
    entr = stats.entropy(prob_dist + 1e-7, base=2)    # Adding a small value (1e-7) to avoid log(0)

    return entr

def local_entropy_response(image):
    """
    Calculates the spatial entropy of the image using a local entropy filter.
    """
    entropy_image = entropy(image, disk(10))  
    mean_entropy = np.mean(entropy_image)

    return mean_entropy

def texture_diversity_response(image):
    M = image.shape[0]  
    l_div = 0

    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])

    # Vertical differences
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])

    # Diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])

    # Counter-diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])

    return l_div


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    cropped_images = []
    texture_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))

    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray) / 255.0
        
        histogram, _ = np.histogram(crop_gray.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()
        m = stats.entropy(prob_dist + 1e-7, base=2)
        if m > threshold: 
            texture_images.append(crop)

    if len(texture_images) == 0:
        texture_images = [CenterCrop(image)]

    return texture_images

############################################################## GET TEXTURE IMAGES ##############################################################

def get_texture_images(x, patch_size=32, grid_size=4):
    """使用texture_crop函数获取简单和复杂的图像，确保没有重叠
    
    该方法从输入图像中提取复杂和简单的纹理区域，并将它们分别拼接成正方形图像。
    复杂图像由熵值最高的区域组成，简单图像由熵值最低的区域组成。
    
    Args:
        x: 输入张量 [B, C, H, W]
        patch_size: 图像块大小
        grid_size: 网格大小
        
    Returns:
        simple_image: 简单图像张量 [B, C, patch_size*grid_size, patch_size*grid_size]
            由熵值最低的区域拼接而成的正方形图像
        complex_image: 复杂图像张量 [B, C, patch_size*grid_size, patch_size*grid_size]
            由熵值最高的区域拼接而成的正方形图像
    """
    # 记录输入设备以确保返回相同设备的张量
    device = x.device
    
    B, C, H, W = x.shape
    batch_simple_images = []
    batch_complex_images = []
    transform = ToTensor()
    
    for i in range(B):
        # 将张量转换为PIL图像
        img = transforms.ToPILImage()(x[i].cpu())
        
        # 使用texture_crop获取复杂图像（选择熵值高的区域）
        complex_crops = texture_crop(
            img, 
            stride=patch_size, 
            window_size=patch_size, 
            metric='ghe',  # 使用全局熵作为度量
            position='top',  # 选择顶部（熵值高的区域）
            n=grid_size * grid_size
        )
        
        # 使用texture_crop获取简单图像（选择熵值低的区域）
        simple_crops = texture_crop(
            img, 
            stride=patch_size, 
            window_size=patch_size, 
            metric='ghe',  # 使用全局熵作为度量
            position='bottom',  # 选择底部（熵值低的区域）
            n=grid_size * grid_size
        )
        
        # 将裁剪的图像拼接为正方形图像
        # 创建一个patch_size*grid_size × patch_size*grid_size的正方形图像
        complex_img = Image.new('RGB', (patch_size * grid_size, patch_size * grid_size))
        simple_img = Image.new('RGB', (patch_size * grid_size, patch_size * grid_size))
        
        # 将复杂图像块按网格排列，形成正方形图像
        for idx, crop in enumerate(complex_crops):
            row = idx // grid_size
            col = idx % grid_size
            complex_img.paste(crop, (col * patch_size, row * patch_size))
            
        # 将简单图像块按网格排列，形成正方形图像
        for idx, crop in enumerate(simple_crops):
            row = idx // grid_size
            col = idx % grid_size
            simple_img.paste(crop, (col * patch_size, row * patch_size))
        
        # 转换为张量
        complex_tensor = transform(complex_img)
        simple_tensor = transform(simple_img)
        
        batch_complex_images.append(complex_tensor)
        batch_simple_images.append(simple_tensor)
    
    # 将列表合并为批次张量
    complex_image = torch.stack(batch_complex_images)
    simple_image = torch.stack(batch_simple_images)
    
    # 确保返回的张量与输入张量在同一设备上
    return simple_image.to(device), complex_image.to(device)