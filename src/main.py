from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import SakeDataset, get_transforms, read_data
from model import SakeNet
from retrieval_utils import (FaissKNeighbors, compute_mrr, compute_rank_list,
                             fix_seed, infer)

if __name__ == '__main__':
    EXP_NAME='convnext_base_meigara_ep100'
    BASE_DIR = Path(__file__).parents[1]
    MODEL_DIR = BASE_DIR.joinpath('models', 'fold0')
    FEATURE_DIR = BASE_DIR.joinpath('features')
    OUTPUT_DIR = BASE_DIR.joinpath('outputs')
    
    cfg = Config()
    device = cfg.device
    fix_seed(cfg.seed)  # シード値の固定
    
    # データの読み込み
    df_cite, df_train, df_test, df_sub = read_data()
    
    ## インデックス用
    index_dataset = SakeDataset(df_cite['path'].to_list(),
                                transform=get_transforms(phase='valid'))
    index_loader = DataLoader(index_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              pin_memory=True)
    
    ## クエリ用
    query_dataset = SakeDataset(df_test['path'].to_list(),
                                transform=get_transforms(phase='valid'))
    query_loader = DataLoader(query_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              pin_memory=True)
    
    # モデルの読み込み
    model_path = MODEL_DIR.joinpath(f'{EXP_NAME}.pth')
    model = SakeNet(Config).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print(f'Loaded {model_path}')
    else:
        print('model not found.')

    # 参照データの埋め込みベクトルを作成
    index_path = FEATURE_DIR.joinpath(f'cite_embeddings_{EXP_NAME}.npy')
    if index_path.exists():
        index_embeddings = np.load(index_path)
    else:
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        index_embeddings = infer(index_loader, model)
        np.save(index_path, index_embeddings)
    print(f'index: {index_embeddings.shape}')
    
    # インデックスの作成
    idx2cite_gid = dict(zip(df_cite.index, df_cite['cite_gid']))
    knn = FaissKNeighbors(model_name=cfg.model_name, index_name=EXP_NAME, k=20)
    knn.fit(index_embeddings)
    knn.save_index()
    
    # クエリ画像の特徴量を取得
    query_path = FEATURE_DIR.joinpath(f'query_embeddings_{EXP_NAME}.npy')
    if query_path.exists():
        query_embeddings = np.load(query_path)
    else:
        query_embeddings = infer(query_loader, model)
        np.save(query_path, query_embeddings)
    print(f'query: {query_embeddings.shape}')
    
    # クエリ画像の検索
    cite_gids = []
    for _q_emb in tqdm(query_embeddings):
        distance, pred = knn.predict(_q_emb)
        _cite_gids = [str(idx2cite_gid[idx]) for idx in pred]
        cite_gids.append(' '.join(_cite_gids))
    df_test['cite_gid'] = cite_gids
    df_test[['gid', 'cite_gid']].to_csv(
        OUTPUT_DIR.joinpath(f'submission_{EXP_NAME}.csv'), index=False)
    
    # TODO: 性能評価
    rank_list = compute_rank_list(df_test['cite_gid'].to_list(),
                                  df_test['gid'].to_list())
    score = compute_mrr(rank_list)
    print(f'{EXP_NAME} MRR: {score}')
