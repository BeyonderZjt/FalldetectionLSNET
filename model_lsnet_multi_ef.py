
import torch
import torch.nn as nn
from thop import profile
from lsnet import LSNet  # 导入LSNet模型

# CBAM注意力模块
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class LSNetEarlyFusion(nn.Module):  # 前融合模型类
    def __init__(self, num_classes=6, pretrained=True):
        super(LSNetEarlyFusion, self).__init__()
        
        # 定义轻量化3x3深度可分离卷积，对两个输入分别进行初步特征提取
        self.doppler_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),  # 深度卷积
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),  # 逐点卷积
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        self.range_time_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),  # 深度卷积
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),  # 逐点卷积
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # 创建LSNet主网络，输入通道数为6
        self.main_net = LSNet(
            img_size=224,
            patch_size=8,
            in_chans=6,  # 输入通道数为6（3+3）
            num_classes=num_classes,
            embed_dim=[32, 64, 96, 128],
            key_dim=[8, 8, 8, 8],
            depth=[1, 1, 2, 2],
            num_heads=[2, 2, 2, 2],
            distillation=False
        )
        
        # CBAM注意力模块（输入通道数与LSNet输出特征维度一致）
        self.cbam = CBAMLayer(channel=128)  # CBAM模块
        
        if pretrained:
            try:
                checkpoint = torch.load('lsnet_t_pretrained.pth')
                # 加载与预训练模型匹配的权重
                model_dict = self.main_net.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.main_net.load_state_dict(model_dict)
                print("成功加载预训练权重")
            except:
                print("未找到预训练权重，随机初始化权重")
                
        # 调整分类头（添加CBAM后再进入分类层）
        if num_classes != 1000: 
            self.main_net.head = nn.Sequential(
                nn.Dropout(0.5), #原为0.4
                nn.Linear(128, num_classes)
            )
    
    def forward(self, doppler_x, range_x):
        # 对两个输入分别进行初步特征提取
        doppler_feat = self.doppler_conv(doppler_x)
        range_feat = self.range_time_conv(range_x)
        
        # 在通道维度拼接（3+3=6通道）
        fused_input = torch.cat([doppler_feat, range_feat], dim=1)
        
        # 输入到主网络获取特征
        x = self.main_net.patch_embed(fused_input) 
        for block in self.main_net.blocks1:
            x = block(x)
        for block in self.main_net.blocks2:
            x = block(x)
        for block in self.main_net.blocks3:
            x = block(x)
        for block in self.main_net.blocks4:
            x = block(x)
        
        # 应用CBAM注意力机制
        x = self.cbam(x) 
        
        # 全局平均池化并进入分类头
        x = x.mean(dim=(2, 3))  # 对空间维度求平均
        output = self.main_net.head(x)
        return output
 # 修复：计算模型参数量和FLOPs的方法（添加设备匹配）
    def calculate_parameters_flops(self, input_size=(3, 224, 224)):
        """
        计算模型的参数量和FLOPs
        input_size: 输入图像的尺寸 (通道数, 高度, 宽度)
        """
        # 获取模型当前所在的设备
        device = next(self.parameters()).device
        
        # 创建随机输入张量并移到模型所在设备
        doppler_input = torch.randn(1, *input_size).to(device)
        range_input = torch.randn(1, *input_size).to(device)
        
        # 计算FLOPs和参数量
        flops, params = profile(self, inputs=(doppler_input, range_input))
        
        # 格式化输出
        flops = flops / 1e6  # 转换为MFLOPs
        params = params / 1e6  # 转换为M参数
        
        print(f"模型参数量: {params:.2f} M")
        print(f"模型FLOPs: {flops:.2f} MFLOPs")
        
        return params, flops
# 原单分支模型保持不变
def lsnet_t(num_classes=6,pretrained=True):
    model = LSNet(
        img_size=224,
        patch_size=8,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=[32, 64, 96, 128],
        key_dim=[8, 8, 8, 8],
        depth=[1, 1, 2, 2],
        num_heads=[2, 2, 2, 2],
        distillation=False
    )

    if pretrained:
        try:
            checkpoint = torch.load('lsnet_t_pretrained.pth')
            model.load_state_dict(checkpoint, strict=False)
            print("成功加载预训练权重")
        except:
            print("未找到预训练权重，随机初始化权重")
            
    if num_classes != 1000: 
       model.head = nn.Sequential(
           nn.Dropout(0.3),
           nn.Linear(128, num_classes)
       )
    return model