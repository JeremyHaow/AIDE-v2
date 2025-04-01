# AIDE_V2

AIDE_V2是一个改进的深度伪造检测模型，基于原始AIDE模型进行优化。主要改进包括：

1. 改进的DCT评分模块，返回简单和复杂patches序列
2. 随机重组patches为NxN网格图像
3. 使用ResNet和ConvNeXt进行特征提取
4. 多视角特征融合

## 目录结构

```
AIDE_V2/
├── data/
│   ├── dctv2.py          # 改进的DCT评分模块
│   └── dataset.py        # 数据集加载类
├── models/
│   └── AIDE_V2.py       # 主模型实现
├── scripts/
│   ├── train_v2.py      # 训练脚本
│   └── train_v2.sh      # 训练启动脚本
├── checkpoints/         # 模型检查点保存目录
├── datasets/           # 数据集目录
├── pretrained/         # 预训练模型目录
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/AIDE_V2.git
cd AIDE_V2
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
- 将ResNet50预训练模型放在`pretrained/resnet50.pth`
- 将ConvNeXt预训练模型放在`pretrained/convnext_base.pth`

## 数据集准备

数据集应按以下结构组织：
```
datasets/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## 训练

1. 修改`scripts/train_v2.sh`中的参数配置
2. 运行训练脚本：
```bash
cd scripts
bash train_v2.sh
```

## 主要参数说明

- `--batch_size`: 批次大小，默认32
- `--num_epochs`: 训练轮数，默认100
- `--learning_rate`: 学习率，默认0.001
- `--patch_size`: patch大小，默认32
- `--grid_size`: 重组图像网格大小，默认4
- `--sequence_length`: 简单和复杂序列长度，默认16

## 引用

如果您在研究中使用了AIDE_V2，请引用：

```bibtex
@article{aide_v2,
  title={AIDE_V2: An Improved Deepfake Detection Model},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
``` 