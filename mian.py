from torch import nn, optim

from src.model import get_model
from src.dataset import get_data_loaders
from src.trainer import TrainingConfig, Trainer, ModelCheckpointCallback, EarlyStoppingCallback, TensorBoardCallback
from src.model import ResNet1dEncoder, BasicBlock1d, AppleSugarModel, VitEncoder, MultiViewEncoder


# 模型相关
spectral_encoder_type = 'transformer' # 'resnet1d', 'transformer'
image_encoder_type = 'multiview' # 'resnet', 'vit', 'multiview'
output_dim = 1 # 模型输出维度
fusion_method = 'cross_attention' # 'concat', 'bilinear', 'cross_attention'
hidden_dim = 512
dropout = 0.3
output_activation = None
spectral_params = {
     # resnet1d 参数
    'resnet1d' : {
        'layers' : (2, 2),
        # in_channels 共用参数
        # output_dim 共用参数
        'pool_type' : 'avg', # 共用参数
        'zero_init_residual' : False
    },
    
    # Transformers 参数
    'transformer': {
        'input_channels' : 1,
        'seq_len' : 256,
        'embed_dim' : 64,
        'output_dim' : None,
        'n_heads' : 4,             
        'n_layers' : 3, 
        'expansion_ratio': 4, 
        'dropout': 0.1,
        'use_cnn_preproc': True,
        'pool_type' : 'mean'
    }
}
image_params = {
    'resnet' : {
        'model_name' : 'resnet18',
        'pretrained' : True,
        'freeze_layers' : False,
        'output_dim' : None,
        'pool_type' : 'avg'
    },
    
    'vit' : {
        'model_name' : 'vit_tiny_patch16_224',
        'pretrained' : True,
        'freeze_layers' : False,
        'output_dim' : None,
    }
}
multiview_params = {
    'num_views' : 2,
    'fusion_method' : 'lstm',
    'output_dim' : None,
    'dropout' : 0.1
}

# spectral_params = {
#     'pool_type' : 'avg',
#     'seq_len' : 120
# }

# multiview_params = {
#     'output_dim' : 512
# }

# image_params = {
    
# }

# 数据集相关
img_dir = r'data/fig'
excel_path = r'data/spec.xlsx'
batch_size = 64
img_size = 224
seed = 42
views = 2
spec_preprocess = ['snv']

# 训练相关


spectral_encoder = ResNet1dEncoder(BasicBlock1d, (2, 2), output_dim=512)
image_encoder = MultiViewEncoder(VitEncoder(), 2, output_dim=512)

model = AppleSugarModel(
    spectral_encoder = None,
    image_encoder = image_encoder,
    fusion_method = 'concat'
)

train_loader, val_loader, test_loader = get_data_loaders(
    img_dir = img_dir,
    excel_path = excel_path,
    batch_size = batch_size,
    img_size = img_size,
    random_seed = seed,
    views = views,
    spec_preprocess_steps = spec_preprocess
)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
# optimizer = optim.SGD(lr=3e-3)
criterion = nn.SmoothL1Loss()

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 创建训练配置
config = TrainingConfig(
    device = 'cuda',
    epochs = 200
)

# 创建回调
callbacks = []
if True:
    callbacks.append(ModelCheckpointCallback('models', 10, True))
if True:
    callbacks.append(EarlyStoppingCallback(100))
if True:
    callbacks.append(TensorBoardCallback('logs'))

# 创建训练器
trainer = Trainer(        
                    config,
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    callbacks,
                )

# 开始训练
print('开始训练...')
trainer.train()
print('训练完成!')