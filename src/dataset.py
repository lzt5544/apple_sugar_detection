import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Union
from scipy.linalg import lstsq
from scipy.signal import savgol_filter
import random

# 光谱预处理类 (已提供)
class SpectralPreprocessor:
    @staticmethod
    def snv(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=-1, keepdims=True)
        sd = np.std(x, axis=-1, ddof=1, keepdims=True) + 1e-8
        return ((x - mu) / sd).astype(np.float32)

    @staticmethod
    def msc(x: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        if reference is None:
            reference = np.mean(x, axis=0)
        coeffs = lstsq(reference[:, None], x.T, lapack_driver='gelsy')[0].T
        return (x - coeffs[:, 0:1]) / coeffs[:, 1:2]

    @staticmethod
    def detrend(x: np.ndarray, type: str = "linear") -> np.ndarray:
        from scipy.signal import detrend as sp_detrend
        if x.ndim == 1:
            return sp_detrend(x, type=type).astype(np.float32)
        return np.apply_along_axis(sp_detrend, -1, x, type=type).astype(np.float32)

    @staticmethod
    def mean_centering(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        return (x - mean).astype(np.float32)

    @staticmethod
    def standardization(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, ddof=1, keepdims=True) + 1e-8
        return ((x - mean) / std).astype(np.float32)

    @staticmethod
    def savitzky_golay(x: np.ndarray, window_length: int = 11, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
        if x.ndim == 1:
            return savgol_filter(x, window_length, polyorder, deriv=deriv).astype(np.float32)
        return np.apply_along_axis(lambda y: savgol_filter(y, window_length, polyorder, deriv=deriv), -1, x).astype(np.float32)

    @staticmethod
    def derivative(x: np.ndarray, order: int = 1, delta: float = 1.0) -> np.ndarray:
        if order == 1:
            deriv = np.diff(x, n=1, axis=-1)
            return np.concatenate([deriv, np.zeros_like(deriv[..., :1])], axis=-1).astype(np.float32) / delta
        elif order == 2:
            deriv = np.diff(x, n=2, axis=-1)
            return np.concatenate([np.zeros_like(deriv[..., :1]), deriv, np.zeros_like(deriv[..., :1])], axis=-1).astype(np.float32) / (delta ** 2)
        else:
            raise ValueError("仅支持一阶和二阶导数")

    @staticmethod
    def minmax_range(x: np.ndarray, range: tuple = (0, 1)) -> np.ndarray:
        min_val = np.min(x, axis=-1, keepdims=True)
        max_val = np.max(x, axis=-1, keepdims=True)
        return (x - min_val) / (max_val - min_val + 1e-8) * (range[1] - range[0]) + range[0]

    @staticmethod
    def log_transform(x: np.ndarray, offset: float = 1.0) -> np.ndarray:
        return np.log10(x + offset).astype(np.float32)

    @staticmethod
    def pipeline(x: Union[np.ndarray, list], steps: list = ["snv"]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        for step in steps:
            if step == "snv":
                x = SpectralPreprocessor.snv(x)
            elif step == "msc":
                x = SpectralPreprocessor.msc(x)
            elif step == "detrend":
                x = SpectralPreprocessor.detrend(x)
            elif step == "normalize":
                x = SpectralPreprocessor.minmax_range(x)
            elif step == "log":
                x = SpectralPreprocessor.log_transform(x)
            elif step == "mean_center":
                x = SpectralPreprocessor.mean_centering(x)
            elif step == "standardize":
                x = SpectralPreprocessor.standardization(x)
            elif step == "savgol":
                x = SpectralPreprocessor.savitzky_golay(x)
            elif step == "derivative1":
                x = SpectralPreprocessor.derivative(x, order=1)
            elif step == "derivative2":
                x = SpectralPreprocessor.derivative(x, order=2)
            elif step != "raw":
                raise ValueError(f"非支持的预处理步骤: {step}")
        
        return x.squeeze()

def load_spectrum_table(xls_path: str):
    df = pd.read_excel(xls_path)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = df.columns[0]
    sugar_col = df.columns[1]
    spec_cols = df.columns[2:]

    def parse_id(s):
        s = str(s).strip()
        m = re.match(r'^\s*(\d+)[_\-](\d+)[_\-](\d+)\s*$', s)
        if not m:
            raise ValueError(f"Unrecognized ID format: {s}")
        return tuple(map(int, m.groups()))

    sid, cid, _ = zip(*df[id_col].map(parse_id))
    df["sid"] = sid
    df["cid"] = cid

    for c in spec_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, id_col, sugar_col, list(spec_cols)

class AppleSugarDataset(Dataset):
    def __init__(
        self,
        img_dir=None,
        df=None,
        sugar_col=None,
        spec_cols=None,
        views=1,
        spec_pp=None,
        transform=None,
        train=True,
        default_img_size=(224, 224),
    ):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform
        self.train = train
        self.views = views
        self.default_img_size = default_img_size

        self.sugar = self.df[sugar_col].values
        self.specs = self.df[spec_cols].values
        self.sid = self.df["sid"].values
        self.cid = self.df["cid"].values

        if spec_pp:
            self.specs = spec_pp(self.specs)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spectrum = torch.FloatTensor(self.specs[idx])
        sugar = torch.FloatTensor([self.sugar[idx]])
        sid = str(self.sid[idx])
        cid = str(self.cid[idx])

        image_tensor = None
        if self.img_dir is not None:
            if self.views == 1:
                img_path = os.path.join(self.img_dir, f"{sid}_{cid}_1.jpg")
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image_tensor = self.transform(image)
                    else:
                        image_tensor = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
                except FileNotFoundError:
                    image_tensor = torch.zeros(3, *self.default_img_size, dtype=torch.float32)
            else:
                image_tensor = torch.zeros((self.views, 3, *self.default_img_size), dtype=torch.float32)
                for view in range(1, self.views + 1):
                    img_path = os.path.join(self.img_dir, f"{sid}_{cid}_{view}.jpg")
                    try:
                        image = Image.open(img_path).convert("RGB")
                        if self.transform:
                            img_tensor = self.transform(image)
                        else:
                            img_tensor = torch.FloatTensor(np.array(image) / 255.0).permute(2, 0, 1)
                        image_tensor[view - 1] = img_tensor
                    except FileNotFoundError:
                        pass

        sample = {
            "spectrum": spectrum,
            "sugar": sugar,
            "sid": torch.LongTensor([int(sid)]),
            "cid": torch.LongTensor([int(cid)]),
        }

        if image_tensor is not None:
            sample["image"] = image_tensor

        return sample

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    # 确保可重复性
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def get_data_loaders(
    img_dir,
    excel_path,
    batch_size=32,
    img_size=224,
    num_workers=4,
    random_seed=42,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    views=1,
    spec_preprocess_steps=["snv"],
    train_augment=True
):
    # 加载并分割数据
    df, _, sugar_col, spec_cols = load_spectrum_table(excel_path)
    train_df, val_df, test_df = split_data(df, train_ratio, val_ratio, test_ratio, random_seed)
    
    # 光谱预处理
    spec_pp = lambda x: SpectralPreprocessor.pipeline(x, steps=spec_preprocess_steps)
    
    # 图像变换
    if train_augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=train_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=train_transform,
        train=True,
        default_img_size=(img_size, img_size)
    )
    
    val_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=val_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=test_transform,
        train=False,
        default_img_size=(img_size, img_size))
    
    test_dataset = AppleSugarDataset(
        img_dir=img_dir,
        df=test_df,
        sugar_col=sugar_col,
        spec_cols=spec_cols,
        views=views,
        spec_pp=spec_pp,
        transform=test_transform,
        train=False,
        default_img_size=(img_size, img_size))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 参数设置
    img_dir = "path/to/images"
    excel_path = "path/to/data.xlsx"
    batch_size = 32
    img_size = 224
    random_seed = 42
    views = 3  # 使用3个视角
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        img_dir=img_dir,
        excel_path=excel_path,
        batch_size=batch_size,
        img_size=img_size,
        random_seed=random_seed,
        views=views,
        spec_preprocess_steps=["snv", "detrend"]  # 光谱预处理步骤
    )
    
    # 测试数据加载
    sample = next(iter(train_loader))
    print("Batch shapes:")
    print(f"Spectrum: {sample['spectrum'].shape}")
    print(f"Sugar: {sample['sugar'].shape}")
    if 'image' in sample:
        print(f"Images: {sample['image'].shape}")  # (batch, views, C, H, W) 或 (batch, C, H, W)