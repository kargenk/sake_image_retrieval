import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR.joinpath('data')
OUTPUT_DIR = BASE_DIR.joinpath('outputs')
MODEL_DIR = BASE_DIR.joinpath('models')

class SakeDataset(Dataset):
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

def get_transforms(img_size: int = 224) -> torch.Tensor:
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
