import copy
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import MeigaraDataset, get_transforms, read_data
from model import SakeNet
from retrieval_utils import compute_mrr, compute_rank_list, fix_seed


def make_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    ブランド・銘柄を数値ラベルに変換する関数.

    Args:
        df (pd.Dataframe): データフレーム

    Returns:
        pd.DataFrame: 銘柄を数値ラベルに変換して列を追加したデータフレーム
    """
    le = LabelEncoder()
    df['brand_label'] = le.fit_transform(df['brand_id'])
    df['meigara_label'] = le.fit_transform(df['meigara'])
    return df

def plot_train_sample(data: dict[str, torch.Tensor], num_samples: int = 5) -> None:
    import japanize_matplotlib
    import matplotlib.pyplot as plt
    japanize_matplotlib.japanize()
    for i in range(num_samples):
        image = data['images'][i]
        label = data['labels'][i]
        plt.figure(figsize=(4, 4))
        plt.imshow(image[0])
        plt.title(f'label: {int(label)} {label2meigara[int(label)]}')
        plt.savefig(f'{label2meigara[int(label)]}.png')

def train_epoch(
    model,
    optimizer,
    scheduler,
    dataloader,
    criterion,
    device,
    epoch,
) -> float:
    model.train()
    
    dataset_size = 0
    running_loss = 0

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in tbar:
        optimizer.zero_grad()
        
        images = data['images'].to(device, dtype=torch.float)
        labels = data['labels'].to(device, torch.long)
        batch_size = images.size(0)

        with torch.cuda.amp.autocast():
            preds = model(images)
            loss = criterion(preds, labels)

        # 各種更新
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        tbar.set_postfix(Epoch=epoch,
                         Train_Loss=epoch_loss,
                         LR=optimizer.param_groups[0]['lr'])
    
    return epoch_loss

@torch.inference_mode()
def valid_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch
) -> float:
    model.eval()
    
    dataset_size = 0
    running_loss = 0

    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in tbar:
        images = data['images'].to(device, dtype=torch.float)
        labels = data['labels'].to(device, dtype=torch.long)
        batch_size = images.size(0)

        preds = model(images)
        loss = criterion(preds, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        tbar.set_postfix(Epoch=epoch,
                         Train_Loss=epoch_loss,
                         LR=optimizer.param_groups[0]['lr'])
    
    return epoch_loss

def run_training(
    model,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs
) -> list[nn.Module, dict[str, float]]:
    start = time.time()
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_epoch(model,
                                       optimizer,
                                       scheduler,
                                       train_loader,
                                       criterion,
                                       device,
                                       epoch)
        val_epoch_loss = valid_epoch(model,
                                     val_loader,
                                     criterion,
                                     device,
                                     epoch)
        
        # ログ
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # validロスが小さくなった時にモデルを保存
        if val_epoch_loss <= best_epoch_loss:
            print(f'Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})')
            best_epoch_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)
    
    # 訓練時間を表示
    end = time.time()
    time_elapsed = end -start
    print(f'Training complete in {time_elapsed // 3600 :.0f}h'\
            + f'{(time_elapsed % 3600) // 60 :.0f}m'\
            + f'{(time_elapsed % 3600) % 60 :.0f}s')
    print(f'Best Loss: {best_epoch_loss :.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, history

def prepare_loaders(df: pd.DataFrame, fold: int, cfg: Config) -> list[DataLoader]:
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = MeigaraDataset(df_train, 'meigara_label',
                                   get_transforms(phase='train'))
    valid_dataset = MeigaraDataset(df_valid, 'meigara_label',
                                   get_transforms(phase='valid'))
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              pin_memory=True)
    
    return train_loader, valid_loader

if __name__ == '__main__':
    EXP_NAME = 'convnext_base_meigara'
    BASE_DIR = Path(__file__).parents[1]
    MODEL_DIR = BASE_DIR.joinpath('models')
    OUTPUT_DIR = BASE_DIR.joinpath('outputs')
    model_path = MODEL_DIR.joinpath(f'{EXP_NAME}.pth')
    history_path = OUTPUT_DIR.joinpath(f'{EXP_NAME}.log')
    
    fix_seed(Config.seed)  # シード値の固定
    cfg = Config()
    
    # フォルダの作成
    if not model_path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # データの読み込み
    _, df_train, _, _ = read_data()
    
    # ラベルとマッピングを作成
    df_train = make_target_label(df_train)
    label2brand = df_train[['brand_id', 'brand_label']]\
        .set_index('brand_label').to_dict()['brand_id']
    label2meigara = df_train[['meigara', 'meigara_label']]\
        .set_index('meigara_label').to_dict()['meigara']
    cfg.num_target_class = len(label2meigara)
    
    # 交差検証時にテストデータとなるfold数を列に追加
    skf = StratifiedKFold(n_splits=cfg.n_fold)
    for fold, (_, val_ids) in enumerate(skf.split(X=df_train, y=df_train['meigara_label'])):
        df_train.loc[val_ids, 'kfold'] = fold
    df_train.kfold = df_train.kfold.astype(int)
    
    # データローダの準備
    train_loader, val_loader = prepare_loaders(df_train, fold=0, cfg=cfg)
    if cfg.debug:
        sample = next(iter(train_loader))
        plot_train_sample(sample)
    
    # モデル、損失関数、最適化手法、スケジューラの定義
    model = SakeNet(cfg).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer_class = getattr(optim, cfg.optimizer_name)
    lr_scheduler_class = getattr(optim.lr_scheduler, cfg.scheduler_name)
    optimizer = optimizer_class(model.parameters(),
                                **cfg.optimizer_params)
    scheduler = lr_scheduler_class(optimizer,
                                   **cfg.scheduler_params)
    
    model, history = run_training(model, optimizer, scheduler,
                                  criterion=criterion,
                                  device=cfg.device,
                                  num_epochs=cfg.num_epochs)
    
    # ログファイルの保存
    with open(history_path, 'w', encoding='utf-8') as f:
        keys = list(history.keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
