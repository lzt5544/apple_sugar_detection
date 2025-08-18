# 苹果糖度检测项目

这是一个基于PyTorch的苹果糖度检测项目，使用深度学习模型通过苹果图像预测其糖度值。

## 项目结构

```
apple_sugar_detection/
├── .gitignore
├── README.md
├── requirements.txt       # 项目依赖
├── data/                  # 数据集目录
│   ├── train.csv          # 训练集标签
│   └── test.csv           # 测试集标签
├── models/                # 模型保存目录
├── results/               # 结果保存目录
│   └── logs/              # TensorBoard日志
└── src/                   # 源代码目录
    ├── __init__.py
    ├── config.py          # 配置文件
    ├── data_loader.py     # 数据加载器
    ├── model.py           # 模型定义
    ├── train.py           # 训练脚本
    └── inference.py       # 推理脚本
```

## 环境配置

1. 克隆项目到本地
```bash
git clone <repository_url>
cd apple_sugar_detection
```

2. 创建虚拟环境（可选但推荐）
```bash
# 使用conda
conda create -n apple_sugar python=3.8
conda activate apple_sugar

# 或使用venv
python -m venv venv
# Windows激活
venv\Scripts\activate
# Linux/Mac激活
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备

1. 在`data/`目录下准备苹果图像数据集
2. 创建训练集和测试集的CSV文件，格式如下：
```csv
image_name,sugar_content
apple_001.jpg,12.5
apple_002.jpg,14.2
...
```

## 模型训练

运行训练脚本：
```bash
cd src
python train.py --model_name resnet50 --batch_size 32 --epochs 50 --lr 1e-4
```

可选参数：
- `--data_dir`: 数据目录路径，默认为`../data`
- `--train_csv`: 训练集CSV文件路径，默认为`../data/train.csv`
- `--test_csv`: 测试集CSV文件路径，默认为`../data/test.csv`
- `--model_name`: 模型名称，可选`resnet50`或`custom`，默认为`resnet50`
- `--pretrained`: 是否使用预训练权重，默认为`True`
- `--freeze_layers`: 是否冻结骨干网络层，默认为`True`
- `--batch_size`: 批次大小，默认为32
- `--epochs`: 训练轮数，默认为50
- `--lr`: 学习率，默认为1e-4
- `--img_size`: 图像大小，默认为224

## 推理预测

使用训练好的模型进行预测：
```bash
cd src
python inference.py --model_path ../results/resnet50_best.pth --image_path ../data/test_image.jpg
```

必选参数：
- `--model_path`: 模型权重文件路径
- `--image_path`: 输入图像文件路径

可选参数：
- `--model_name`: 模型名称，需与训练时保持一致，默认为`resnet50`
- `--img_size`: 图像大小，需与训练时保持一致，默认为224
- `--results_dir`: 结果保存目录，默认为`../results`
- `--save_result`: 是否保存预测结果图像

## 结果分析

训练过程中的损失和指标会记录到TensorBoard日志中，可以通过以下命令查看：
```bash
tensorboard --logdir ../results/logs
```

## 注意事项

1. 确保数据集格式正确，图像路径有效
2. 训练前可以先检查配置文件`config.py`中的参数设置
3. 根据实际硬件条件调整`batch_size`和`num_workers`参数
4. 如果使用GPU训练，请确保已安装正确版本的CUDA和cuDNN

## 联系方式

如有问题，请联系: [your_email@example.com]