import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import dct
from scipy import stats
import open_clip
from torchvision import transforms
import random
import math
from .srm_filter_kernel import all_normalized_hpf_list
from data.crops import texture_crop, get_texture_images
from PIL import Image
import os

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class PatchReorganizer(nn.Module):
    """图像块重组模块 - 将patches重组为NxN的图像"""
    def __init__(self, grid_size=3):
        super(PatchReorganizer, self).__init__()
        self.grid_size = grid_size

    def forward(self, patches):
        """
        输入: 
            patches: [B, N, C, patch_size, patch_size] - patches序列
        输出: [B, C, H, W] - 重组后的图像
        """
        B, N, C, patch_size, _ = patches.shape
        num_patches = self.grid_size * self.grid_size
        
        # 随机选择patches
        indices = torch.randperm(N)[:num_patches]
        selected_patches = torch.index_select(patches, 1, indices)
        
        # 重组图像
        new_h = self.grid_size * patch_size
        new_w = self.grid_size * patch_size
        output = torch.zeros(B, C, new_h, new_w, device=patches.device)
        
        for b in range(B):
            for idx in range(num_patches):
                i = idx // self.grid_size
                j = idx % self.grid_size
                output[b, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = selected_patches[b, idx]
        
        return output

class HPF(nn.Module):
    """高频滤波器模块
    用于提取图像中的高频特征，使用30个预定义的SRM滤波器
    """
    def __init__(self):
        super(HPF, self).__init__()

        # 加载30个SRM滤波器
        all_hpf_list_5x5 = []

        # 将3x3的滤波器填充为5x5
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        # 优化：先将列表转换为numpy数组，再转换为tensor
        hpf_weight = np.array(all_hpf_list_5x5).reshape(30, 1, 5, 5)
        hpf_weight = torch.from_numpy(hpf_weight).float().contiguous()
        hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   
        # 创建卷积层，输入通道为3(RGB)，输出通道为30(滤波器数量)
        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        """前向传播
        Args:
            input: 输入图像 [B, 3, H, W]
        Returns:
            output: 高频特征 [B, 30, H, W]
        """
        output = self.hpf(input)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AIDE_V2(nn.Module):
    """AIDE改进版本 - 基于原始AIDE架构的优化版本"""
    def __init__(self, resnet_path, convnext_path, patch_size=32, grid_size=4, sequence_length=16):
        super(AIDE_V2, self).__init__()
        
        # 高频特征提取器
        self.hpf = HPF()
        
        # 图像处理相关参数
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        
        # 图像转换模块
        self.transform = transforms.ToTensor()
        
        # ResNet分支
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])
       
        # 加载预训练的ResNet权重
        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()
    
            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")
        
        # 特征融合MLP
        self.fc = Mlp(258, 1024, 2)

        # 加载ConvNeXt模型
        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        # 移除分类头
        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        # 特征投影层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        )
        # 冻结ConvNeXt参数
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: 分类结果 [B, 2]
        """
        # 1. 使用texture_crop获取简单和复杂图像
        simple_image, complex_image = get_texture_images(x, self.patch_size, self.grid_size)
        
        # 2. 高频特征提取
        simple_image = self.hpf(simple_image)
        complex_image = self.hpf(complex_image)

        # 3. ConvNeXt特征提取
        with torch.no_grad():
            # 图像归一化
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(x, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(x, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(x, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(x, non_blocking=True).view(3, 1, 1)

            # 提取ConvNeXt特征
            local_convnext_image_feats = self.openclip_convnext_xxl(
                x * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(x.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        # 4. ResNet特征提取
        x_min = self.model_min(simple_image)
        x_max = self.model_max(complex_image)

        # 5. 特征融合
        x_1 = (x_min + x_max) / 2
        x = torch.cat([x_0, x_1], dim=1)
        x = self.fc(x)

        return x