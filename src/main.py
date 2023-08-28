from pathlib import Path

import torch
from torch.data.utils import DataLoader

from config import Config
from dataset import SakeDataset, get_transforms, read_data
from retrieval_utils import fix_seed

if __name__ == '__main__':
    fix_seed(Config.seed)  # シード値の固定
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    cfg = Config()
    
    # データの読み込み
    df_cite, df_train, df_test, df_sub = read_data()
    cite_dataset = SakeDataset(df_cite['path'].to_list(),
                               transform=get_transforms())
    cite_loader = DataLoader(cite_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
