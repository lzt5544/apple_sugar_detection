import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.config import *
from src.model import get_model, load_model_weights

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class AppleSugarInference:
    def __init__(self, model_path, model_name=MODEL_NAME, img_size=IMG_SIZE):
        # 检查设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        
        # 初始化模型
        self.model = get_model(model_name=model_name, pretrained=False)
        
        # 加载模型权重
        print(f'从 {model_path} 加载模型权重')
        self.model = load_model_weights(self.model, model_path)
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 图像大小
        self.img_size = img_size
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        # 应用变换
        image = self.transform(image)
        image = image.unsqueeze(0)  # 添加批次维度
        
        # 移动图像到设备
        image = image.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(image)
            sugar_content = output.item()
        
        return original_image, sugar_content
    
    def visualize_prediction(self, image, sugar_content, save_path=None):
        # 显示图像和预测结果
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.title(f'预测糖度值: {sugar_content:.2f}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f'预测结果已保存到 {save_path}')
        
        plt.show()

def main(args):
    # 确保目录存在
    ensure_dirs()

    # 创建推理实例
    inference = AppleSugarInference(
        model_path=args.model_path,
        model_name=args.model_name,
        img_size=args.img_size
    )
    
    # 进行预测
    image, sugar_content = inference.predict(args.image_path)
    
    # 显示预测结果
    print(f'预测的苹果糖度值: {sugar_content:.2f}')
    
    # 可视化预测结果
    save_path = None
    if args.save_result:
        save_path = os.path.join(RESULTS_DIR, f'prediction_{os.path.basename(args.image_path)}')
    
    inference.visualize_prediction(image, sugar_content, save_path)

# 解析命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='苹果糖度检测推理')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, choices=['resnet50', 'custom'], help='模型架构')
    
    # 输入参数
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='图像 resize 尺寸')
    
    # 输出参数
    parser.add_argument('--save_result', action='store_true', help='保存预测结果')
    
    args = parser.parse_args()
    
    main(args)