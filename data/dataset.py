import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class TrainDataset(Dataset):
    """训练数据集类"""
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir (str): 数据集根目录
            transform: 数据增强转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 收集训练集图像路径和标签
        train_dir = os.path.join(root_dir, 'train')
        for label, class_name in enumerate(['0_real', '1_fake']):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        print(f'训练集加载完成: {len(self.image_paths)}张图像')
        print(f'真实图像数量: {sum(1 for l in self.labels if l == 0)}')
        print(f'伪造图像数量: {sum(1 for l in self.labels if l == 1)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TestDataset(Dataset):
    """测试数据集类"""
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir (str): 数据集根目录
            transform: 数据转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 收集测试集图像路径和标签
        test_dir = os.path.join(root_dir, 'test')
        for label, class_name in enumerate(['0_real', '1_fake']):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        print(f'测试集加载完成: {len(self.image_paths)}张图像')
        print(f'真实图像数量: {sum(1 for l in self.labels if l == 0)}')
        print(f'伪造图像数量: {sum(1 for l in self.labels if l == 1)}')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 