import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import cv2

from models.AIDE_V2 import create_model, save_checkpoint, load_checkpoint
from data.dataset import TrainDataset, TestDataset

def parse_args():
    parser = argparse.ArgumentParser(description='AIDE_V2模型训练、评估和推理')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'inference'],
                      help='运行模式：训练、评估或推理')
    parser.add_argument('--data_root', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 模型参数
    parser.add_argument('--patch_size', type=int, default=32, help='patch大小')
    parser.add_argument('--grid_size', type=int, default=4, help='重组图像网格大小')
    parser.add_argument('--sequence_length', type=int, default=16, help='简单和复杂序列长度')
    parser.add_argument('--resnet_path', type=str, default='../pretrained/resnet50.pth', help='ResNet预训练模型路径')
    parser.add_argument('--convnext_path', type=str, default='../pretrained/convnext_base.pth', help='ConvNeXt预训练模型路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'linear'],
                      help='学习率调度器类型')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='模型检查点保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    # 数据增强参数
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--random_rotation', type=float, default=15.0, help='随机旋转角度范围')
    parser.add_argument('--random_horizontal_flip', type=float, default=0.5, help='随机水平翻转概率')
    parser.add_argument('--color_jitter', type=float, default=0.4, help='颜色抖动强度')
    parser.add_argument('--random_erasing', type=float, default=0.2, help='随机擦除概率')
    
    # 评估参数
    parser.add_argument('--eval_batch_size', type=int, default=64, help='评估批次大小')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='评估使用的检查点路径')
    
    # 推理参数
    parser.add_argument('--image_path', type=str, default=None, help='单张图像推理路径')
    parser.add_argument('--inference_checkpoint', type=str, default=None, help='推理使用的检查点路径')
    
    # Tensorboard参数
    parser.add_argument('--log_dir', type=str, default='../logs', help='Tensorboard日志目录')
    
    return parser.parse_args()

def get_data_transforms(args):
    """获取数据转换"""
    train_transforms = []
    test_transforms = []
    
    # 基本转换
    train_transforms.extend([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms.extend([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据增强
    if args.use_augmentation:
        train_transforms.insert(0, transforms.RandomRotation(args.random_rotation))
        train_transforms.insert(1, transforms.RandomHorizontalFlip(args.random_horizontal_flip))
        train_transforms.insert(2, transforms.ColorJitter(
            brightness=args.color_jitter,
            contrast=args.color_jitter,
            saturation=args.color_jitter,
            hue=args.color_jitter
        ))
        if args.random_erasing > 0:
            train_transforms.append(transforms.RandomErasing(p=args.random_erasing))
    
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

def train(args, model, train_loader, test_loader, criterion, optimizer, scheduler, writer):
    """训练函数"""
    best_acc = 0.0
    start_epoch = 0
    
    # 恢复训练
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume)
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for inputs, labels in pbar:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
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
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录训练指标
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', 100.*train_correct/train_total, epoch)
        
        # 验证
        test_loss, test_acc = evaluate(args, model, test_loader, criterion)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        print(f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 保存检查点
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
        }, args.checkpoint_dir, test_acc > best_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc

def evaluate(args, model, test_loader, criterion):
    """评估函数"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    return test_loss, test_acc

def inference(args, model, image_path):
    """推理函数"""
    model.eval()
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(args.device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 输出结果
    class_names = ['真实图像', 'AI生成图像']
    print(f'预测类别: {class_names[predicted_class]}')
    print(f'置信度: {confidence:.2%}')
    print(f'真实图像概率: {probabilities[0][0]:.2%}')
    print(f'AI生成图像概率: {probabilities[0][1]:.2%}')

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_model(
        resnet_path=args.resnet_path,
        convnext_path=args.convnext_path,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
        sequence_length=args.sequence_length
    ).to(args.device)
    
    # 创建Tensorboard写入器
    writer = SummaryWriter(args.log_dir)
    
    if args.mode == 'train':
        # 获取数据转换
        train_transform, test_transform = get_data_transforms(args)
        
        # 创建数据集和数据加载器
        train_dataset = TrainDataset(args.data_root, transform=train_transform)
        test_dataset = TestDataset(args.data_root, transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, 
                               shuffle=False, num_workers=4)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
        
        # 创建学习率调度器
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:  # linear
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
                                                  end_factor=0.1, total_iters=args.num_epochs)
        
        # 训练模型
        train(args, model, train_loader, test_loader, criterion, optimizer, scheduler, writer)
        
    elif args.mode == 'eval':
        # 加载检查点
        if args.eval_checkpoint:
            checkpoint = torch.load(args.eval_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建测试数据集和数据加载器
        _, test_transform = get_data_transforms(args)
        test_dataset = TestDataset(args.data_root, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, 
                               shuffle=False, num_workers=4)
        
        # 评估模型
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(args, model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
    elif args.mode == 'inference':
        # 加载检查点
        if args.inference_checkpoint:
            checkpoint = torch.load(args.inference_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 单张图像推理
        if args.image_path:
            inference(args, model, args.image_path)
        else:
            print('请提供图像路径')

if __name__ == '__main__':
    main() 