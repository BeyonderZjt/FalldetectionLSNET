# predict_lsnet_early_fusion.py (修改后，适配输入前融合模型)
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 导入输入前融合模型
from model_lsnet import LSNetEarlyFusion

class FallDetectionPredictor:
    def __init__(self, model_path, num_classes=6):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载输入前融合模型
        self.model = LSNetEarlyFusion(pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = [
            'walking', 
            'sitting_down', 
            'standing_up', 
            'pick_up_object', 
            'drink_water', 
            'fall'
        ]
        
        # 数据变换保持和训练一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
  
    def predict(self, doppler_img_path, range_img_path):
        """对一对谱图（多普勒-时间和距离-时间）进行预测"""
        if not os.path.exists(doppler_img_path):
            raise FileNotFoundError(f"多普勒-时间谱图路径不存在：{doppler_img_path}")
        if not os.path.exists(range_img_path):
            raise FileNotFoundError(f"距离-时间谱图路径不存在：{range_img_path}")
        
        # 加载两个谱图
        doppler_image = Image.open(doppler_img_path).convert('RGB')
        range_image = Image.open(range_img_path).convert('RGB')
        
        # 预处理
        doppler_tensor = self.transform(doppler_image).unsqueeze(0).to(self.device)
        range_tensor = self.transform(range_image).unsqueeze(0).to(self.device)
        
        # 预测（输入前融合模型需要两个输入）
        with torch.no_grad():
            outputs = self.model(doppler_tensor, range_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        return predicted_class, confidence.item()
    
   
    def predict_batch(self, doppler_img_dir, range_img_dir):
        """对一批谱图对进行预测"""
        results = []
        
        for img_name in os.listdir(doppler_img_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                doppler_path = os.path.join(doppler_img_dir, img_name)
                range_path = os.path.join(range_img_dir, img_name)
                
                if os.path.exists(range_path):
                    try:
                        predicted_class, confidence = self.predict(doppler_path, range_path)
                        results.append({
                            'img_name': img_name,
                            'predicted_class': predicted_class,
                            'confidence': confidence
                        })
                    except Exception as e:
                        print(f"处理 {img_name} 时出错: {str(e)}")
                else:
                    print(f"未找到对应的距离-时间谱图：{range_path}")
        return results

def main():
    """示例用法（输入前融合模型预测）"""
    # 加载输入前融合模型权重
    predictor = FallDetectionPredictor(
        model_path='./fall_detection_lsnet_early_fusion.pth', 
        num_classes=6
    )

    # 单张图像对预测
    doppler_img_path = './Doppler_Time/6/1.jpg'  # 多普勒-时间谱图路径
    range_img_path = './Range_Time/6/1.jpg'      # 距离-时间谱图路径
    try:
        predicted_class, confidence = predictor.predict(doppler_img_path, range_img_path)
        print(f"预测结果：{predicted_class}, 置信度：{confidence:.4f}")
        print(f"使用模型：输入前融合LSNet模型")
        
        if predicted_class == 'fall':
            print("⚠️ 检测到跌倒事件！！")
        else:
            print("✅ 正常活动")
    except Exception as e:
        print(f"预测失败: {str(e)}")

    # 批量预测示例
    # doppler_img_dir = './Doppler_Time/6'
    # range_img_dir = './Range_Time/6'
    # results = predictor.predict_batch(doppler_img_dir, range_img_dir)
    # for res in results:
    #     print(f"{res['img_name']}: {res['predicted_class']} ({res['confidence']:.4f})")

if __name__ == '__main__':
    main()