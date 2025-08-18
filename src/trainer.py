import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_data_loaders
from src.model import get_model, load_model_weights

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 配置类
class TrainingConfig:
    def __init__(
        self,
        data_dir: str = '../data',
        train_csv: str = '../data/train.csv',
        test_csv: str = '../data/test.csv',
        img_size: int = 224,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_layers: bool = True,
        weights_path: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        seed: int = 42,
        results_dir: str = '../results',
        save_interval: int = 10,
        device: Optional[str] = None,
        num_workers: int = 4,
        patience: int = 10,
        early_stopping: bool = False,
        mixed_precision: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.img_size = img_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.results_dir = results_dir
        self.save_interval = save_interval
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        self.patience = patience
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision

    def to_dict(self) -> Dict:
        return vars(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        return cls(**config_dict)

# 回调基类
class Callback:
    def on_train_start(self, trainer: 'Trainer') -> None:
        pass

    def on_train_end(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    def on_batch_start(self, trainer: 'Trainer') -> None:
        pass

    def on_batch_end(self, trainer: 'Trainer') -> None:
        pass

# 模型保存回调
class ModelCheckpointCallback(Callback):
    def __init__(self, save_dir: str, save_interval: int = 10, save_best_only: bool = True) -> None:
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_best_only = save_best_only
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        epoch = trainer.current_epoch
        metrics = trainer.metrics

        # 保存最佳模型
        if self.save_best_only and 'val_rmse' in metrics:
            if metrics['val_rmse'] < trainer.best_rmse:
                trainer.best_rmse = metrics['val_rmse']
                checkpoint_path = os.path.join(self.save_dir, f'{trainer.config.model_name}_best.pth')
                trainer.save_checkpoint(checkpoint_path, epoch, is_best=True)
                logger.info(f'Saved best model to {checkpoint_path} with RMSE: {metrics["val_rmse"]:.4f}')

        # 按间隔保存模型
        if not self.save_best_only and (epoch + 1) % self.save_interval == 0:
            checkpoint_path = os.path.join(self.save_dir, f'{trainer.config.model_name}_epoch_{epoch+1}.pth')
            trainer.save_checkpoint(checkpoint_path, epoch)
            logger.info(f'Saved model at epoch {epoch+1} to {checkpoint_path}')

# 早停回调
class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 10, monitor: str = 'val_rmse', mode: str = 'min') -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        metrics = trainer.metrics

        if self.monitor not in metrics:
            logger.warning(f'Monitor metric {self.monitor} not found in metrics.')
            return

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                logger.info(f'Early stopping triggered after {trainer.current_epoch+1} epochs')
                trainer.stop_training = True

# TensorBoard回调
class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, trainer: 'Trainer') -> None:
        epoch = trainer.current_epoch
        metrics = trainer.metrics

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def on_batch_end(self, trainer: 'Trainer') -> None:
        global_step = trainer.current_epoch * len(trainer.train_loader) + trainer.current_batch
        self.writer.add_scalar('train/batch_loss', trainer.current_loss, global_step)

    def on_train_end(self, trainer: 'Trainer') -> None:
        self.writer.close()

# 训练器类
class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = scheduler or optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        self.callbacks = callbacks or []

        # 初始化设备
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 训练状态
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        self.metrics = {}
        self.best_rmse = float('inf')
        self.stop_training = False

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None

    def save_checkpoint(self, filepath: str, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }

        if is_best:
            checkpoint['best_rmse'] = self.best_rmse
        else:
            checkpoint['val_rmse'] = self.metrics.get('val_rmse', float('inf'))

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_rmse = checkpoint.get('best_rmse', float('inf'))
        logger.info(f'Loaded checkpoint from {filepath}, starting from epoch {self.current_epoch+1}')

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        metrics = {}

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')

        for i, batch in enumerate(progress_bar):
            self.current_batch = i
            # 假设最后一个元素是目标值
            *inputs, targets = batch
            # 将所有输入移到设备上
            inputs = [x.to(self.device) for x in inputs]
            targets = targets.to(self.device).unsqueeze(1)

            # 调用批次开始回调
            for callback in self.callbacks:
                callback.on_batch_start(self)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(*inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(*inputs)
                loss = self.criterion(outputs, targets)

            # 反向传播和优化
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 统计损失
            running_loss += loss.item()
            self.current_loss = loss.item()

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())

            # 调用批次结束回调
            for callback in self.callbacks:
                callback.on_batch_end(self)

        # 计算 epoch 损失
        epoch_loss = running_loss / len(self.train_loader)
        metrics['train_loss'] = epoch_loss

        return metrics

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        metrics = {}

        with torch.no_grad():
            for batch in self.test_loader:
                # 假设最后一个元素是目标值
                *inputs, targets = batch
                # 将所有输入移到设备上
                inputs = [x.to(self.device) for x in inputs]
                targets = targets.to(self.device).unsqueeze(1)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(*inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(*inputs)
                    loss = self.criterion(outputs, targets)

                val_loss += loss.item()

                # 收集预测和目标值
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算指标
        epoch_loss = val_loss / len(self.test_loader)
        mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
        rmse = np.sqrt(mse)

        metrics['val_loss'] = epoch_loss
        metrics['val_mse'] = mse
        metrics['val_rmse'] = rmse

        return metrics

    def train(self) -> None:
        # 调用训练开始回调
        for callback in self.callbacks:
            callback.on_train_start(self)

        logger.info(f'Starting training for {self.config.epochs} epochs')
        logger.info(f'Using device: {self.device}')
        logger.info(f'Train data: {len(self.train_loader.dataset)} samples')
        logger.info(f'Test data: {len(self.test_loader.dataset)} samples')

        for epoch in range(self.current_epoch, self.config.epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch

            # 调用 epoch 开始回调
            for callback in self.callbacks:
                callback.on_epoch_start(self)

            start_time = time.time()

            # 训练一个 epoch
            train_metrics = self.train_one_epoch()

            # 评估
            val_metrics = self.evaluate()

            # 合并指标
            self.metrics = {**train_metrics, **val_metrics}

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(self.metrics.get('val_loss', float('inf')))

            # 打印 epoch 信息
            epoch_time = time.time() - start_time
            logger.info(f'Epoch {epoch+1}/{self.config.epochs}, Time: {epoch_time:.2f}s')
            logger.info(f'Train Loss: {self.metrics["train_loss"]:.4f}, Val Loss: {self.metrics["val_loss"]:.4f}')
            logger.info(f'Val MSE: {self.metrics["val_mse"]:.4f}, Val RMSE: {self.metrics["val_rmse"]:.4f}')

            # 调用 epoch 结束回调
            for callback in self.callbacks:
                callback.on_epoch_end(self)

            logger.info('-' * 50)

        # 调用训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info(f'Training completed. Best RMSE: {self.best_rmse:.4f}')

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Apple Sugar Content Detection Training')
    parser.add_argument('--config', type=str, help='Path to config file (json)')
    
    # 添加命令行参数（与配置类对应）
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory with images')
    parser.add_argument('--train_csv', type=str, default='../data/train.csv', help='Path to train CSV file')
    parser.add_argument('--test_csv', type=str, default='../data/test.csv', help='Path to test CSV file')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'custom'], help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--freeze_layers', action='store_true', default=True, help='Freeze backbone layers')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained weights')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 保存和日志参数
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory for results')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    
    # 额外参数
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='Enable mixed precision training')
    
    args = parser.parse_args()
    
    # 从配置文件或命令行参数创建配置
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig.from_dict(config_dict)
    else:
        config = TrainingConfig(
            data_dir=args.data_dir,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            img_size=args.img_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
            freeze_layers=args.freeze_layers,
            weights_path=args.weights_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            results_dir=args.results_dir,
            save_interval=args.save_interval,
            num_workers=args.num_workers,
            patience=args.patience,
            early_stopping=args.early_stopping,
            mixed_precision=args.mixed_precision,
        )
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建结果目录
    os.makedirs(config.results_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.results_dir, 'config.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f'Saved configuration to {config_path}')
    
    # 加载数据
    logger.info('Loading data...')
    train_loader, test_loader = get_data_loaders(
        data_dir=config.data_dir,
        train_csv=config.train_csv,
        test_csv=config.test_csv,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers
    )
    
    # 初始化模型
    logger.info('Initializing model...')
    model = get_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        freeze_layers=config.freeze_layers
    )
    
    # 加载预训练权重（如果提供）
    if config.weights_path:
        logger.info(f'Loading weights from {config.weights_path}')
        model = load_model_weights(model, config.weights_path)
    
    # 创建回调
    callbacks = []
    
    # 模型保存回调
    checkpoint_callback = ModelCheckpointCallback(
        save_dir=config.results_dir,
        save_interval=config.save_interval,
        save_best_only=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    if config.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            patience=config.patience,
            monitor='val_rmse',
            mode='min'
        )
        callbacks.append(early_stopping_callback)
    
    # TensorBoard回调
    log_dir = os.path.join(config.results_dir, 'logs')
    tb_callback = TensorBoardCallback(log_dir=log_dir)
    callbacks.append(tb_callback)
    
    # 创建训练器
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        callbacks=callbacks
    )
    
    # 如果提供了权重路径，加载检查点
    if config.weights_path and os.path.exists(config.weights_path):
        try:
            trainer.load_checkpoint(config.weights_path)
        except Exception as e:
            logger.warning(f'Failed to load checkpoint: {e}')
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()