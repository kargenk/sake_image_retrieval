import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from albumentations import (Compose, HueSaturationValue, Normalize,
                            RandomBrightnessContrast, RandomResizedCrop,
                            RandomRotate90, Resize, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR.joinpath('data')
OUTPUT_DIR = BASE_DIR.joinpath('outputs')
MODEL_DIR = BASE_DIR.joinpath('models')

class MeigaraDataset(Dataset):
    """ 銘柄ラベル分類用のデータセットクラス """
    def __init__(self, df, target_cols, transform):
        self.df = df
        self.file_path = df['path'].to_numpy()
        self.transform = transform
        
        self.labels = df[target_cols].to_numpy()
        if len(target_cols) == 1:
            self.labels = np.ravel(self.labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_path[idx]
        image = self._read_image(file_path)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx])
        return {'images': image, 'labels': label}
    
    def _read_image(self, path: str) -> None:
        with open(path, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)
        
        return image

class SakeDataset(Dataset):
    """
    類似画像検索用のデータセットクラス.
    """
    def __init__(self, image_paths: list, labels: list = None,
                 transform: dict[str, typing.Any] = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        item = {}
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
            item['image'] = image
        
        if self.labels:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            item['label'] = label
        
        return item
    
    def _read_image(self, path: str) -> None:
        with open(path, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)
        
        return image

def get_transforms(img_size: int = 224, phase: str = 'train') -> Compose:
    """
    データ拡張を返す関数.

    Args:
        img_size (int, optional): 画像サイズ. Defaults to 224.
        phase (str, optional): 訓練か検証か. Defaults to 'train'.

    Returns:
        Compose: AlbumentationのCompose
    """
    match phase:
        case 'train':
            return Compose(
                [
                    RandomResizedCrop(img_size, img_size),
                    ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.15,
                        rotate_limit=60,
                        p=0.5
                    ),
                    # RandomRotate90(p=0.3),
                    HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.5
                    ),
                    RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.5
                    ),
                    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2()
                ]
            )
        case 'valid':
            return Compose(
                [
                    Resize(img_size, img_size),
                    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2()
                ]
    )

def read_data() -> list[pd.DataFrame]:
    # データの読み込み
    df_cite = pd.read_csv(DATA_DIR.joinpath('cite.csv'))
    df_train = pd.read_csv(DATA_DIR.joinpath('train.csv'))
    df_test = pd.read_csv(DATA_DIR.joinpath('test.csv'))
    df_sub = pd.read_csv(DATA_DIR.joinpath('sample_submission.csv'))
    
    # 画像ファイルパスを追加
    cite_filenames = df_cite['cite_filename'].to_list()
    df_cite['path'] = [str(DATA_DIR.joinpath('cite_images', filename))
                           for filename in cite_filenames]
    train_filenames = df_train['filename'].to_list()
    df_train['path'] = [str(DATA_DIR.joinpath('query_images', filename))
                           for filename in train_filenames]
    test_filenames = df_test['filename'].to_list()
    df_test['path'] = [str(DATA_DIR.joinpath('query_images', filename))
                           for filename in test_filenames]
    
    return df_cite, df_train, df_test, df_sub

if __name__ == '__main__':
    df_cite, df_train, df_test, df_sub = read_data()
    cite_dataset = SakeDataset(df_cite['path'].to_list(),
                               transform=get_transforms())
    sample = cite_dataset.__getitem__(0)
    print(sample['image'].shape)
