import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.config import *
from src.data_loader import get_data_loaders
from src.model import get_model
from src.trainer import Trainer, TrainingConfig, ModelCheckpointCallback, EarlyStoppingCallback, TensorBoardCallback


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='苹果糖度检测模型训练')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='模型名称')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批量大小')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='权重衰减')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='图像尺寸')
    parser.add_argument('--pretrained', action='store_true', default=PRETRAINED, help='是否使用预训练模型')
    parser.add_argument('--freeze_layers', action='store_true', default=FREEZE_LAYERS, help='是否冻结预训练模型层')
    parser.add_argument('--save_interval', type=int, default=SAVE_INTERVAL, help='模型保存间隔')
    parser.add_argument('--save_best_only', action='store_true', default=SAVE_BEST_ONLY, help='是否只保存最佳模型')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help='早停耐心值')
    parser.add_argument('--tensorboard_logging', action='store_true', default=TENSORBOARD_LOGGING, help='是否使用TensorBoard日志')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确保目录存在
    ensure_dirs()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据加载
    train_loader, test_loader = get_data_loaders(
        data_dir=DATA_DIR,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f'训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}')

    # 模型加载
    model = get_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_layers=args.freeze_layers
    ).to(device)
    print(f'模型: {args.model_name} 加载完成')

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 创建训练配置
    config = TrainingConfig(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=test_loader,
        save_dir=MODELS_DIR,
        save_interval=args.save_interval,
        save_best_only=args.save_best_only,
        early_stopping_patience=args.early_stopping_patience,
        tensorboard_logging=args.tensorboard_logging,
        log_dir=LOGS_DIR
    )

    # 创建回调
    callbacks = []
    if args.save_interval > 0 or args.save_best_only:
        callbacks.append(ModelCheckpointCallback(config))
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(config))
    if args.tensorboard_logging:
        callbacks.append(TensorBoardCallback(config))

    # 创建训练器
    trainer = Trainer(config, callbacks)

    # 开始训练
    print('开始训练...')
    trainer.train()
    print('训练完成!')


if __name__ == '__main__':
    main()