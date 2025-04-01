import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCT_base_Rec_Module_v2(nn.Module):
    """改进的DCT评分模块，返回简单和复杂序列
    
    Args:
        window_size (int): 切片大小
        stride (int): 滑动步长
        output (int): 输出图像大小
        grade_N (int): 评分等级数
        level_fliter (list): 滤波器级别列表
    """
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
    def forward(self, x):
        """
        输入: [C, H, W]
        输出: 
            simple_sequence: [N, C, window_size, window_size] - 简单patches序列
            complex_sequence: [N, C, window_size, window_size] - 复杂patches序列
        """
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        
        # 1. 图像切片
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)  
        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size) 
        
        # 2. DCT变换
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        # 3. 应用滤波器
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)
        
        # 4. 计算复杂度分数
        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1,2,3])
            grade += w * _x            
            w *= k
        
        # 5. 根据分数排序，获取简单和复杂序列
        _, idx = torch.sort(grade)
        simple_indices = idx[:N]  # 简单patches的索引
        complex_indices = torch.flip(idx, dims=[0])[:N]  # 复杂patches的索引
        
        # 6. 提取简单和复杂序列
        simple_sequence = torch.index_select(level_x_unfold, 0, simple_indices)
        complex_sequence = torch.index_select(level_x_unfold, 0, complex_indices)
        
        return simple_sequence, complex_sequence 