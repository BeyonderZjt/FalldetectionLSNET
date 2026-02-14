
# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from torch.utils.data import DataLoader
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm

# # 导入输入前融合模型
# from model_lsnet_multi_ef import LSNetEarlyFusion
# from fall_detection_dataset_lsnet_multibranch_ef import FallDetectionDatasetV2, get_data_transforms,split_dataset,check_data_leakage

# def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, device='cuda'):
#     since = time.time()

#     train_loss_history = []
#     train_acc_history = []
#     val_loss_history = []
#     val_acc_history = []
#     learning_rates = []
#     best_model_wts = model.state_dict()
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch+1}/{num_epochs}')
#         print('-' * 10)

#         current_lr = optimizer.param_groups[0]['lr']
#         learning_rates.append(current_lr)
#         print(f'Learning Rate: {current_lr:.6f}')
    
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0
            
#             dataloader = dataloaders[phase]
#             pbar = tqdm(
#                 total=len(dataloader),
#                 desc=f'{phase:5}',
#                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
#                 ncols=80
#             )

#             # 接收两个输入（多普勒和距离-时间谱图）
#             for doppler_inputs, range_inputs, labels in dataloader:
#                 doppler_inputs = doppler_inputs.to(device)
#                 range_inputs = range_inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     # 将两个输入传入模型
#                     outputs = model(doppler_inputs, range_inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
                
#                 running_loss += loss.item() * doppler_inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
                
#                 batch_acc = torch.sum(preds == labels.data).double() / doppler_inputs.size(0)
#                 pbar.set_postfix({
#                     'Loss': f'{loss.item():.4f}',
#                     'Acc': f'{batch_acc.item():.4f}'
#                 })
#                 pbar.update(1)
            
#             pbar.close()
            
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = model.state_dict().copy()
            
#             if phase == 'train':
#                 train_loss_history.append(epoch_loss)
#                 train_acc_history.append(epoch_acc.cpu().numpy())
#             else:
#                 val_loss_history.append(epoch_loss)
#                 val_acc_history.append(epoch_acc.cpu().numpy())
        
#         #scheduler.step()
#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:.4f}')

#     model.load_state_dict(best_model_wts)

#     history = {
#         'train_loss': train_loss_history,
#         'train_acc': train_acc_history,
#         'val_loss': val_loss_history,
#         'val_acc': val_acc_history,
#         'learning_rates': learning_rates
#     }

#     return model, history


# def evaluate_model(model, test_loader, device, class_names):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         # 接收两个输入
#         for doppler_inputs, range_inputs, labels in test_loader:
#             doppler_inputs = doppler_inputs.to(device)
#             range_inputs = range_inputs.to(device)
#             labels = labels.to(device)

#             # 将两个输入传入模型
#             outputs = model(doppler_inputs, range_inputs)
#             _, preds = torch.max(outputs, 1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     print('Classification Report:')
#     print(classification_report(all_labels, all_preds, target_names=class_names))

#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     plt.title('Confusion Matrix (Early Fusion)')  
#     plt.tight_layout()
#     plt.savefig('confusion_matrix_lsnet_early_fusion.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     return all_labels, all_preds

# def plot_training_history(history):
#     """绘制训练历史"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     ax1.plot(history['train_loss'], label='Training Loss')
#     ax1.plot(history['val_loss'], label='Validation Loss')
#     ax1.set_title('Training and Validation Loss (Early Fusion)')  
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss')
#     ax1.legend()

#     ax2.plot(history['train_acc'], label='Training Accuracy')
#     ax2.plot(history['val_acc'], label='Validation Accuracy')
#     ax2.set_title('Training and Validation Accuracy (Early Fusion)')  
#     ax2.set_xlabel('Epochs')
#     ax2.set_ylabel('Accuracy')
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig('training_history_lsnet_early_fusion.png', dpi=300, bbox_inches='tight')
#     plt.show()

# def main():
#     # 参数设置
#     doppler_dir = './image'  # 多普勒-时间谱图目录
#     range_time_dir = './Range_Time'  # 距离-时间谱图目录
#     batch_size = 16
#     num_epochs = 150
#     learning_rate = 0.001
#     num_classes = 6
#     random_state = 42

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f'Using device: {device}')
    
#     # 数据变换
#     train_transform, test_transform = get_data_transforms()

#     # 创建数据集（多域输入）
#     full_dataset = FallDetectionDatasetV2(
#         root_dir=doppler_dir,
#         range_time_dir=range_time_dir,
#         transform=None
#     )
    
#     # 分割数据集为训练集和验证集
#     train_dataset, val_dataset = split_dataset(
#         full_dataset, 
#         train_ratio=0.8, 
#         random_state=random_state
#     )
    
#     # 应用不同的变换
#     train_dataset.dataset.transform = train_transform
#     val_dataset.dataset.transform = test_transform

#     # 检查数据泄露
#     check_data_leakage(train_dataset, val_dataset, full_dataset)

#     # 创建数据加载器
#     dataloaders = {
#         'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
#         'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     }

#     datasizes = {
#         'train': len(train_dataset),
#         'val': len(val_dataset)
#     }

#     print(f'Number of training samples: {datasizes["train"]}')
#     print(f'Number of validation samples: {datasizes["val"]}')

#     # 类别名称
#     class_names = full_dataset.class_names
    
#     # 使用输入前融合模型
#     model = LSNetEarlyFusion(num_classes=num_classes, pretrained=False)
#     model = model.to(device)

#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-5)

#     # 学习率调度器
#    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

#     # 训练模型
#     print('Training started (Early Fusion Model)...')
#     model, history = train_model(
#         model, dataloaders, criterion, optimizer,
#         num_epochs=num_epochs, device=device
#     )

#     # 绘制训练历史
#     plot_training_history(history)

#     # 评估模型
#     print('Evaluating model on validation set...')
#     evaluate_model(model, dataloaders['val'], device, class_names)

#     # 保存模型
#     torch.save(model.state_dict(), 'fall_detection_lsnet_early_fusion.pth')
#     print("Model saved to fall_detection_lsnet_early_fusion.pth")

# if __name__ == '__main__':
#     main()
#     print("Finished training")

# train_fall_detection_lsnet.py (修改后，适配输入前融合模型)
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 导入输入前融合模型
from model_lsnet_multi_ef import LSNetEarlyFusion
from fall_detection_dataset_lsnet_multibranch_ef import FallDetectionDatasetV2, get_data_transforms,split_dataset,check_data_leakage

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device='cuda', patience=50):
    since = time.time()

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    learning_rates = []
    best_model_wts = model.state_dict()
    best_acc = 0.0
    early_stop_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f'Learning Rate: {current_lr:.6f}')
    
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            dataloader = dataloaders[phase]
            pbar = tqdm(
                total=len(dataloader),
                desc=f'{phase:5}',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                ncols=80
            )

            # 接收两个输入（多普勒和距离-时间谱图）
            for doppler_inputs, range_inputs, labels in dataloader:
                doppler_inputs = doppler_inputs.to(device)
                range_inputs = range_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 将两个输入传入模型
                    outputs = model(doppler_inputs, range_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * doppler_inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                batch_acc = torch.sum(preds == labels.data).double() / doppler_inputs.size(0)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{batch_acc.item():.4f}'
                })
                pbar.update(1)
            
            pbar.close()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 早停逻辑和最佳模型保存
            if phase == 'val':
                # 学习率调度器.step()放在验证集计算之后
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    early_stop_counter = 0  # 重置计数器
                else:
                    early_stop_counter += 1
                    print(f'早停计数器: {early_stop_counter}/{patience}')
                    if early_stop_counter >= patience:
                        print(f'⚠️  验证集准确率连续{patience}个epoch无提升，触发早停！')
                        model.load_state_dict(best_model_wts)
                        history = {
                            'train_loss': train_loss_history,
                            'train_acc': train_acc_history,
                            'val_loss': val_loss_history,
                            'val_acc': val_acc_history,
                            'learning_rates': learning_rates
                        }
                        time_elapsed = time.time() - since
                        print(f'Training completed (early stopped) in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best val Acc: {best_acc:.4f}')
                        return model, history
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().numpy())
        
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    history = {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'learning_rates': learning_rates
    }

    return model, history


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 接收两个输入
        for doppler_inputs, range_inputs, labels in test_loader:
            doppler_inputs = doppler_inputs.to(device)
            range_inputs = range_inputs.to(device)
            labels = labels.to(device)

            # 将两个输入传入模型
            outputs = model(doppler_inputs, range_inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix (Early Fusion)')  
    plt.tight_layout()
    plt.savefig('confusion_matrix_lsnet_early_fusion.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_labels, all_preds

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss (Early Fusion)')  
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy (Early Fusion)')  
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 新增：添加学习率曲线子图
    fig, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(history['learning_rates'], label='Learning Rate')
    ax3.set_title('Learning Rate Schedule (Early Fusion)')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    plt.tight_layout()
    plt.savefig('learning_rate_history_lsnet_early_fusion.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.tight_layout()
    plt.savefig('training_history_lsnet_early_fusion.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 参数设置
    doppler_dir = './image'  # 多普勒-时间谱图目录
    range_time_dir = './Range_Time'  # 距离-时间谱图目录
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.0001
    num_classes = 6
    random_state = 42
    patience = 50  # 早停耐心值

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # 数据变换
    train_transform, test_transform = get_data_transforms()

    # 创建数据集（多域输入）
    full_dataset = FallDetectionDatasetV2(
        root_dir=doppler_dir,
        range_time_dir=range_time_dir,
        transform=None
    )
    
    # 分割数据集为训练集和验证集
    train_dataset, val_dataset = split_dataset(
        full_dataset, 
        train_ratio=0.8, 
        random_state=random_state
    )
    
    # 应用不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    # 检查数据泄露
    check_data_leakage(train_dataset, val_dataset, full_dataset)

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    datasizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    print(f'Number of training samples: {datasizes["train"]}')
    print(f'Number of validation samples: {datasizes["val"]}')

    # 类别名称
    class_names = full_dataset.class_names
    
    # 使用输入前融合模型
    model = LSNetEarlyFusion(num_classes=num_classes, pretrained=False)
    model = model.to(device)

     
    # 新增：计算并打印模型参数量和FLOPs
    print("\n模型参数统计:")
    model.calculate_parameters_flops()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-5)

    # 学习率调度器：当验证损失停止改善时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # 每次衰减为当前学习率的1/2
        patience=100,  # 多少个epoch无改善后衰减
        verbose=True,
        min_lr=1e-6  # 最小学习率
    )

    # 训练模型
    print('Training started (Early Fusion Model)...')
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device, patience=patience
    )

    # 绘制训练历史
    plot_training_history(history)

    # 评估模型
    print('Evaluating model on validation set...')
    evaluate_model(model, dataloaders['val'], device, class_names)

    # 保存模型
    torch.save(model.state_dict(), 'fall_detection_lsnet_early_fusion.pth')
    print("Model saved to fall_detection_lsnet_early_fusion.pth")

if __name__ == '__main__':
    main()
    print("Finished training")