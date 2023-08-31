from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import SakeDataset, get_transforms, read_data
from model import SakeNet
from retrieval_utils import FaissKNeighbors, fix_seed, infer

if __name__ == '__main__':
    EXP_NAME='convnext_base'
    BASE_DIR = Path(__file__).parents[1]
    MODEL_DIR = BASE_DIR.joinpath('models')
    FEATURE_DIR = BASE_DIR.joinpath('features')
    OUTPUT_DIR = BASE_DIR.joinpath('outputs')
    
    model_path = MODEL_DIR.joinpath(f'{EXP_NAME}.pth')
    index_path = FEATURE_DIR.joinpath(f'train_embeddings_{EXP_NAME}.npy')
    query_path = FEATURE_DIR.joinpath(f'query_embeddings_{EXP_NAME}.npy')
    
    fix_seed(Config.seed)  # シード値の固定
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    cfg = Config()
    
    # データの読み込み
    df_cite, df_train, df_test, df_sub = read_data()
    ## 訓練用
    train_dataset = SakeDataset(df_cite['path'].to_list(),
                               transform=get_transforms())
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.n_workers,
                              pin_memory=True)
    ## テスト用
    test_dataset = SakeDataset(df_test['path'].to_list(),
                               transform=get_transforms())
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.n_workers,
                             pin_memory=True)
    
    # モデルの読み込み
    model = SakeNet(Config).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    else:
        MODEL_DIR.mkdir(parents=True)
        torch.save(model.state_dict(), model_path)

    # 埋め込みベクトルの作成
    if index_path.exists():
        train_embeddings = np.load(index_path)
    else:
        FEATURE_DIR.mkdir(parents=True)
        train_embeddings = infer(train_loader, model)
        np.save(index_path, train_embeddings)
    print(f'index: {train_embeddings.shape}')
    
    # インデックスの作成
    idx2cite_gid = dict(zip(df_cite.index, df_cite['cite_gid']))
    knn = FaissKNeighbors(model_name=cfg.model_name, index_name=EXP_NAME, k=20)
    knn.fit(train_embeddings)
    knn.save_index()
    
    # クエリ画像の検索
    if query_path.exists():
        query_embeddings = np.load(query_path)
    else:
        query_embeddings = infer(test_loader, model)
        np.save(query_path, query_embeddings)
    print(f'query: {query_embeddings.shape}')
