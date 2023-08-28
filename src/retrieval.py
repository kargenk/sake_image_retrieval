from pathlib import Path

import pandas as pd

from retrieval_utils import fix_seed

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR.joinpath('data')
OUTPUT_DIR = BASE_DIR.joinpath('outputs')
MODEL_DIR = BASE_DIR.joinpath('models')

class Config:
    img_size = 224
    model_name = 'convnext_base'
    in_channels = 3
    embedding_dim = 128
    pretrained = True
    batch_size = 128
    n_workers  = 0
    seed = 3407

def read_data() -> list[pd.DataFrame]:
    # データの読み込み
    df_cite = pd.read_csv(DATA_DIR.joinpath('cite.csv'))
    df_train = pd.read_csv(DATA_DIR.joinpath('train.csv'))
    df_test = pd.read_csv(DATA_DIR.joinpath('test.csv'))
    df_sub = pd.read_csv(DATA_DIR.joinpath('sample_submission.csv'))
    
    # 画像ファイルパスを追加
    cite_filenames = df_cite['cite_filename'].to_list()
    df_cite['path'] = [str(DATA_DIR.joinpath('cite_images', filename)
                           for filename in cite_filenames)]
    train_filenames = df_train['filename'].to_list()
    df_train['path'] = [str(DATA_DIR.joinpath('query_images', filename)
                           for filename in train_filenames)]
    test_filenames = df_train['filename'].to_list()
    df_test['path'] = [str(DATA_DIR.joinpath('query_images', filename)
                           for filename in test_filenames)]
    
    return df_cite, df_train, df_test, df_sub

if __name__ == '__main__':
    # シード値の固定とデータの読み込み
    fix_seed(Config.seed)
    df_cite, df_train, df_test, df_sub = read_data()