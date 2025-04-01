import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.AIDE_V2 import AIDE_V2
from data.dataset import TrainDataset, TestDataset

def parse_args():
    parser = argparse.ArgumentParser(description='训练AIDE_V2模型')
    parser.add_argument('--data_root', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--patch_size', type=int, default=32, help='patch大小')
    parser.add_argument('--grid_size', type=int, default=4, help='重组图像网格大小')
    parser.add_argument('--sequence_length', type=int, default=16, help='简单和复杂序列长度')
    parser.add_argument('--resnet_path', type=str, default='../pretrained/resnet50.pth', help='ResNet预训练模型路径')
    parser.add_argument('--convnext_path', type=str, default='../pretrained/convnext_base.pth', help='ConvNeXt预训练模型路径')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='模型检查点保存目录')
    return parser.parse_args()

def train(args):
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    train_dataset = TrainDataset(args.data_root)
    test_dataset = TestDataset(args.data_root)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    model = AIDE_V2(
        num_classes=2,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
        sequence_length=args.sequence_length
    ).to(device)
    
    # 加载预训练模型
    if os.path.exists(args.resnet_path):
        model.resnet.load_state_dict(torch.load(args.resnet_path))
    if os.path.exists(args.convnext_path):
        model.convnext.load_state_dict(torch.load(args.convnext_path))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}',
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        # 验证
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'\nTest Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'latest_model.pth'))

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main() 