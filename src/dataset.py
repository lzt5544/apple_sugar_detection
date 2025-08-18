import os
from pickle import TUPLE
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class AppleSugarDataset(Dataset):
    def __init__(self, img_dir, excel_file, spec_pp=None, transform=None, train=True):
        self.img_dir = img_dir
        self.excel_file = pd.read_csv(excel_file)
        self.transform = transform
        self.train = train
        
        # 可选：数据增强参数
        self.brightness_factor = 0.2
        self.contrast_factor = 0.2
        self.saturation_factor = 0.2
        
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.data_dir, self.csv_file.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        sugar_content = self.csv_file.iloc[idx, 1].astype(np.float32)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 训练集数据增强
        if self.train:
            # 随机水平翻转
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
            
            # 随机调整亮度
            if random.random() > 0.5:
                image = transforms.functional.adjust_brightness(
                    image, brightness_factor=1 + random.uniform(-self.brightness_factor, self.brightness_factor)
                )
            
            # 随机调整对比度
            if random.random() > 0.5:
                image = transforms.functional.adjust_contrast(
                    image, contrast_factor=1 + random.uniform(-self.contrast_factor, self.contrast_factor)
                )
            
            # 随机调整饱和度
            if random.random() > 0.5:
                image = transforms.functional.adjust_saturation(
                    image, saturation_factor=1 + random.uniform(-self.saturation_factor, self.saturation_factor)
                )
        
        return image, torch.tensor(sugar_content)

def get_data_loaders(data_dir, train_csv, test_csv, batch_size=32, img_size=224, num_workers=4):
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = AppleSugarDataset(
        data_dir=data_dir,
        csv_file=train_csv,
        transform=train_transform,
        train=True
    )
    
    test_dataset = AppleSugarDataset(
        data_dir=data_dir,
        csv_file=test_csv,
        transform=test_transform,
        train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader