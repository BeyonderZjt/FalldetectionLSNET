# fall_detection_dataset_lsnet.py (修改后)
import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

class FallDetectionDatasetV2(Dataset):
    def __init__(self, root_dir, range_time_dir, transform=None):
        """
        跌倒检测数据集（多域谱图版本）
        
        Args:
            root_dir: 多普勒-时间谱图根目录
            range_time_dir: 距离-时间谱图根目录
            transform: 数据增强
        """
        self.root_dir = root_dir
        self.range_time_dir = range_time_dir
        self.transform = transform
        self.class_names = ['walking', 'sitting_down', 'standing_up', 
                           'pick_up_object', 'drink_water', 'fall']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        # 收集所有图像路径和对应的标签
        self.doppler_paths = []  # 多普勒谱图路径
        self.range_time_paths = []  # 距离-时间谱图路径
        self.labels = []
        
        # 遍历文件夹
        for folder_num in range(1,7):
            folder_name = str(folder_num)
            doppler_folder_path = os.path.join(root_dir, folder_name)
            range_folder_path = os.path.join(range_time_dir, folder_name)  # 距离谱图文件夹
            
            if os.path.isdir(doppler_folder_path) and os.path.isdir(range_folder_path):
                cls_idx = folder_num - 1
                # 获取该文件夹下所有图像文件
                for img_name in os.listdir(doppler_folder_path):
                    if self._is_image_file(img_name):
                        doppler_img_path = os.path.join(doppler_folder_path, img_name)
                        range_img_path = os.path.join(range_folder_path, img_name)  # 对应距离谱图路径
                        
                        # 确保两个域的谱图都存在
                        if os.path.exists(range_img_path):
                            self.doppler_paths.append(doppler_img_path)
                            self.range_time_paths.append(range_img_path)
                            self.labels.append(cls_idx)
    
    def _is_image_file(self, filename):
        """检查文件是否为图像文件"""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        return any(filename.lower().endswith(ext) for ext in img_extensions)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.doppler_paths)
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本，返回两个域的谱图和标签"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 加载多普勒-时间谱图
        doppler_img_path = self.doppler_paths[idx]
        doppler_image = Image.open(doppler_img_path).convert('RGB')
        
        # 加载距离-时间谱图
        range_img_path = self.range_time_paths[idx]
        range_image = Image.open(range_img_path).convert('RGB')
        
        label = self.labels[idx]
        
        if self.transform:
            doppler_image = self.transform(doppler_image)
            range_image = self.transform(range_image)  # 对两个谱图应用相同变换
            
        # 返回两个域的谱图和标签（保持不变，供输入前融合使用）
        return doppler_image, range_image, label

def get_data_transforms():
    """获取训练和测试数据的变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2), #新增垂直翻转
        transforms.RandomRotation(15), #原为10
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)), #新增高斯模糊
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def split_dataset(dataset,train_ratio=0.8,random_state=42):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)    
    val_size = dataset_size - train_size

    torch.manual_seed(random_state)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def check_data_leakage(train_dataset,val_dataset,full_dataset):
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    
    overlap_indices = train_indices.intersection(val_indices)
    
    if overlap_indices:
        print(f"⚠️ 数据泄露警告: 发现 {len(overlap_indices)} 个重叠样本!")
        return False
    else:
        print("✓ 训练集和验证集无重叠")
        return True
