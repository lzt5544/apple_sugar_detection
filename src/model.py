import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicBlock1d(nn.Module):
    """1D卷积基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1dEncoder(nn.Module):
    def __init__(
        self,
        block,  
        layers,
        in_channels=1,
        output_dim=None, 
        pool_type="avg",
        zero_init_residual=False,
    ):
        super().__init__()
        
        # 初始层
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layers = nn.ModuleList()
        channels = [64 * (2**i) for i in range(len(layers))]
        strides = [1] + [2] * (len(layers) - 1)
        
        for i, (channel, num_blocks, stride) in enumerate(zip(channels, layers, strides)):
            self.layers.append(self._make_layer(block, channel, num_blocks, stride))

        # 池化方式
        self.pool_type = pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:  # None
            self.pool = nn.Identity()
        
        # 投影层
        default_output_dim = channels[-1] * block.expansion
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        layers.extend([block(self.in_channels, out_channels) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        if isinstance(self.pool, nn.Identity):
            features = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
        else:
            x = self.pool(x)  # [B, C, 1]
            features = x.view(x.size(0), -1)  # [B, C]
        
        return self.proj(features)  # 投影到目标维度

class VitEncoder(nn.Module):
    """ViT图像编码器"""
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, freeze_layers=False, output_dim=None):
        super(VitEncoder, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes=0)
        
        # 冻结参数
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 投影层
        default_output_dim = self.backbone.num_features
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)
    
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 seq_len=256, 
                 embed_dim=64,
                 n_heads=4,             
                 n_layers=3, 
                 expansion_ratio=4, 
                 dropout=0.1,
                 use_cnn_preproc=True,
                 pool_type='mean'):
        super().__init__()
        
        # CNN预处理
        if use_cnn_preproc:
            self.pre_net = nn.Sequential(
                nn.Conv1d(input_channels, embed_dim//2, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim//2),
                nn.GELU(),
                nn.Conv1d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
            seq_len = seq_len // 4 
        else:
            self.pre_net = nn.Linear(input_channels, embed_dim)
        
        # 光谱位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim*expansion_ratio,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出处理
        self.pool_type = pool_type
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        if hasattr(self, 'pre_net') and isinstance(self.pre_net[0], nn.Conv1d):
            x = self.pre_net(x)          # [B,C,L] -> [B,D,L//4]
            x = x.permute(0, 2, 1)      # [B,D,L//4] -> [B,L//4,D]
        else:
            x = x.permute(0, 2, 1)      # [B,C,L] -> [B,L,C]
            x = self.pre_net(x)          # [B,L,C] -> [B,L,D]
        
        x = x + self.pos_embed
        
        if self.pool_type == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.encoder(x)  # [B,L,D]
        
        if self.pool_type == 'cls':
            return x[:, 0] 
        elif self.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.pool_type == 'max':
            return x.max(dim=1).values
        else:
            return x

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        model_name="resnet18",
        pretrained=True,
        freeze_layers=False,
        output_dim=None, 
        pool_type="avg",
    ):
        super(ResNetEncoder, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=pool_type,
        )

        # 冻结参数
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

        default_output_dim = self.backbone.num_features
        
        # 投影层
        if output_dim and output_dim != default_output_dim:
            self.proj = nn.Linear(default_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = default_output_dim

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, num_features]
        return self.proj(features)
    
class MultiViewEncoder(nn.Module):
    def __init__(self, base_encoder, num_views=4, fusion_method='attention'):
        super().__init__()
        self.views_encoder = nn.ModuleList([
            deepcopy(base_encoder) for _ in range(num_views)
        ])
        
        # 多视角融合策略
        self.fusion_method = fusion_method
        if fusion_method == 'attention':
            self.view_attention = nn.Sequential(
                nn.Linear(base_encoder.output_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Softmax(dim=1)
            )
        elif fusion_method == 'lstm':
            self.lstm = nn.LSTM(
                input_size=base_encoder.output_dim,
                hidden_size=base_encoder.output_dim,
                num_layers=1,
                batch_first=True
            )
    
    def forward(self, multi_view_images):
        """ 输入: [batch_size, num_views, C, H, W] """
        batch_size = multi_view_images.shape[0]
        num_views = multi_view_images.shape[1]
        
        # 各视角独立编码
        view_features = []
        for i in range(num_views):
            view_img = multi_view_images[:, i]  # [B,C,H,W]
            feat = self.views_encoder[i](view_img)  # [B,D]
            view_features.append(feat)
        
        # 多视角融合
        if self.fusion_method == 'mean':
            fused = torch.stack(view_features, dim=1).mean(dim=1)
        elif self.fusion_method == 'max':
            fused = torch.stack(view_features, dim=1).max(dim=1).values
        elif self.fusion_method == 'attention':
            all_feats = torch.stack(view_features, dim=1)  # [B,N,D]
            weights = self.view_attention(all_feats)  # [B,N,1]
            fused = (all_feats * weights).sum(dim=1)  # [B,D]
        elif self.fusion_method == 'lstm':
            all_feats = torch.stack(view_features, dim=1)  # [B,N,D]
            _, (hidden, _) = self.lstm(all_feats)
            fused = hidden.squeeze(0)  # [B,D]
            
        return fused
    
class AppleSugarModel(nn.Module):
    def __init__(
        self,
        spectral_encoder: Optional[nn.Module] = None,
        image_encoder: Optional[nn.Module] = None,
        fusion_method: str = 'concat',
        hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.3,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        # 编码器检查
        assert spectral_encoder or image_encoder, "至少需要提供一个编码器"
        self.spectral_encoder = spectral_encoder
        self.image_encoder = image_encoder
        self.is_multimodal = bool(spectral_encoder and image_encoder)
        
        # 单模态输出
        if not self.is_multimodal:
            encoder = spectral_encoder if spectral_encoder else image_encoder
            self.proj = nn.Linear(encoder.output_dim, output_dim)
        
        # 多模态融合
        else:
            self.fusion = CrossModalFusion(
                in_dim_spectral=spectral_encoder.output_dim,
                in_dim_image=image_encoder.output_dim,
                method=fusion_method,
                hidden_dim=hidden_dim
            )
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, output_dim)
            )
        
        # 输出激活
        self.output_act = self._get_activation(output_activation)

    def forward(self, 
               spectral: Optional[torch.Tensor] = None, 
               images: Optional[torch.Tensor] = None):

        assert spectral is not None or images is not None, "需要至少一个输入"
        
        if not self.is_multimodal:
            features = self.spectral_encoder(spectral) if spectral is not None \
                      else self.image_encoder(images)
            output = self.proj(features)
        
        else:
            assert spectral is not None and images is not None, "双模态需要两个输入"
            spectral_feat = self.spectral_encoder(spectral)
            image_feat = self.image_encoder(images)
            fused = self.fusion(spectral_feat, image_feat)
            output = self.output(fused)
        
        return self.output_act(output.squeeze(-1)) if self.output_act else output.squeeze(-1)

    def _get_activation(self, name):
        return {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            None: None
        }.get(name)

from typing import Optional, Literal, Dict, Any
import torch.nn as nn

def get_model(
    # 编码器选择
    spectral_encoder_type: Optional[Literal['resnet1d', 'transformer']] = None,
    image_encoder_type: Optional[Literal['resnet', 'vit', 'multiview']] = None,
    
    # 通用配置
    output_dim: int = 1,
    fusion_method: Literal['concat', 'bilinear', 'cross_attention'] = 'concat',
    hidden_dim: int = 512,
    dropout: float = 0.3,
    output_activation: Optional[Literal['sigmoid', 'relu', 'tanh']] = None,
    
    # 光谱编码器配置
    spectral_params: Optional[Dict[str, Any]] = None,
    
    # 图像编码器配置
    image_params: Optional[Dict[str, Any]] = None,
    
    # 多视角配置
    multiview_params: Optional[Dict[str, Any]] = None
) -> nn.Module:
    
    # 默认参数
    default_spectral_params = {
        'resnet1d': {'layers': [2,2,2,2], 'pool_type': 'mean'},
        'transformer': {'embed_dim': 64, 'pool_type': 'mean', 'use_cnn_preproc': True}
    }
    default_image_params = {
        'resnet': {'model_name': 'resnet18', 'pool_type': 'avg'},
        'vit': {'model_name': 'vit_tiny_patch16_224', 'freeze_layers': False}
    }
    default_multiview_params = {'num_views': 2, 'fusion_method': 'attention'}
    
    # 合并参数
    spectral_params = {**default_spectral_params.get(spectral_encoder_type, {}), 
                      **(spectral_params or {})}
    image_params = {**default_image_params.get(image_encoder_type.replace('multiview','') 
                  if image_encoder_type else '', {}), 
                  **(image_params or {})}
    multiview_params = {**default_multiview_params, **(multiview_params or {})}
    
    # 构建光谱编码器
    spectral_encoder = None
    if spectral_encoder_type == 'resnet1d':
        spectral_encoder = ResNet1dEncoder(
            block=BasicBlock1d,
            in_channels=1,
            output_dim=hidden_dim if fusion_method != 'concat' else None,
            **{k:v for k,v in spectral_params.items() 
               if k in ['layers', 'pool_type', 'zero_init_residual']}
        )
    elif spectral_encoder_type == 'transformer':
        spectral_encoder = TransformerEncoder(
            input_channels=1,
            seq_len=101,
            output_dim=hidden_dim if fusion_method != 'concat' else None,
            **{k:v for k,v in spectral_params.items() 
               if k in ['embed_dim', 'n_heads', 'n_layers', 'pool_type', 'use_cnn_preproc']}
        )

    # 构建图像编码器
    image_encoder = None
    if image_encoder_type:
        # 基础编码器
        if 'resnet' in image_encoder_type:
            base_encoder = ResNetEncoder(
                output_dim=hidden_dim if fusion_method != 'concat' else None,
                **{k:v for k,v in image_params.items() 
                   if k in ['model_name', 'pool_type', 'freeze_layers']}
            )
        else:  # vit
            base_encoder = VitEncoder(
                output_dim=hidden_dim if fusion_method != 'concat' else None,
                **{k:v for k,v in image_params.items() 
                   if k in ['model_name', 'freeze_layers']}
            )
        
        # 多视角处理
        if 'multiview' in image_encoder_type:
            image_encoder = MultiViewEncoder(
                base_encoder=base_encoder,
                **multiview_params
            )
        else:
            image_encoder = base_encoder

    return AppleSugarModel(
        spectral_encoder=spectral_encoder,
        image_encoder=image_encoder,
        fusion_method=fusion_method,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        output_activation=output_activation
    )
