import os
import random
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
INDEX_DIR = Path(__file__).parents[1].joinpath('index')

class FaissKNeighbors:
    def __init__(self, model_name: str, index_name: str, k: int = 10) -> None:
        self.model_name = model_name
        self.index_path = str(INDEX_DIR.joinpath(f'{index_name}.index'))
        self.index = None
        self.d = None
        self.k = k
    
    def fit(self, x: np.array) -> None:
        """
        インデックスを作成する.

        Args:
            x (np.array): ベクトル
        """
        x = x.copy()
        self.d = x.shape[1]
        # 内積を指標にインデックスを作成、正規化しておけばコサイン類似度として測れる
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(x.astype(np.float32))
    
    def save_index(self) -> None:
        """ インデックスを保存する """
        if not INDEX_DIR.exists():
            INDEX_DIR.mkdir(parents=True)
        faiss.write_index(self.index, self.index_path)
        print(f'{self.index_path} saved.')
    
    def read_index(self) -> None:
        """ インデックスを読み込む """
        self.index = faiss.read_index(self.index_path)
        self.d = self.index.d
        print(f'{self.index_path} read.')
    
    def predict(self, x: np.array) -> tuple[np.array]:
        """
        検索結果としてインデックスのうち近い距離をもつものを返す

        Args:
            x (np.array): クエリベクトル

        Returns:
            tuple[np.array]: 距離のリストとインデックスのリスト
        """
        x = x.copy()
        x = x.reshape(-1, self.d)
        
        distances, indices = self.index.search(x.astype(np.float32), k=self.k)
        if x.shape[0] == 1:
            return distances[0], indices[0]
        else:
            return distances, indices

def fix_seed(seed: int = 3407) -> None:
    """
    再現性の担保のために乱数のシード値を固定する関数.

    Args:
        seed (int): シード値
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)     # random
    np.random.seed(seed)  # NumPy
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def infer(data_loader: DataLoader, model: nn.Module) -> np.array:
    """
    モデルに推論させて埋め込みベクトルを作成し、インデックスを作成する.

    Args:
        data_loader (DataLoader): データローダ
        model (nn.Module): モデル

    Returns:
        np.array: 埋め込みベクトル
    """
    model.eval()
    embeddings = []
    for batch in tqdm(data_loader):
        images = batch['image'].to(DEVICE, non_blocking=True).float()
        with torch.inference_mode():
            embedding = model.get_embedding(images)
            embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    
    return embeddings
