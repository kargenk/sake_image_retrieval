from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR.joinpath('data')
OUTPUT_DIR = BASE_DIR.joinpath('outputs')
MODEL_DIR = BASE_DIR.joinpath('models')

def check_df(dfs: list[pd.DataFrame], name_list: list[str]) -> None:
    """
    データフレームの中身と欠損値の確認を行う関数.

    Args:
        dfs (list[pd.DataFrame]): データフレームのリスト
        name_list (list[str]): データフレーム名のリスト
    """
    for name, df in zip(name_list, dfs):
        print('-' * 15, f'\n{name}: {df.shape}')
        print(df.head())
        print('NaNs:')
        print(df.isna().sum(), '\n')

def show_same_meigara(meigara: str) -> None:
    """
    同じ銘柄の酒の画像を描画する関数.

    Args:
        meigara (str): 描画対象の銘柄名
    """
    image_paths = df_train.loc[df_train['meigara'] == meigara, 'path'].to_numpy()
    brand_ids = df_train.loc[df_train['meigara'] == meigara, 'brand_id'].to_numpy()
    col = 2
    row = int(len(image_paths) / col)
    
    # サンプル数が奇数の場合、２列で表示できる分だけ表示する
    n_samples = row * col
    fig, axs = plt.subplots(row, col, figsize = (col * 5, row * 5))
    for i, path in enumerate(image_paths[:n_samples]):
        j = int(i / 2)
        k = i % 2
        image = Image.open(path)
        axs[j, k].imshow(image)
        title = f'brand_id: {str(brand_ids[i])}'
        axs[j, k].set_title(title)
        axs[j, k].axis('off')  # 軸を非表示にする
    plt.show()

if __name__ == '__main__':
    # データの読み込み
    df_cite = pd.read_csv(DATA_DIR.joinpath('cite.csv'))
    df_train = pd.read_csv(DATA_DIR.joinpath('train.csv'))
    df_test = pd.read_csv(DATA_DIR.joinpath('test.csv'))
    df_sub = pd.read_csv(DATA_DIR.joinpath('sample_submission.csv'))
    
    # データの形式と内容を確認
    check_df(dfs=[df_cite, df_train, df_test],
             name_list=['Cite', 'Train', 'Test'])
    # 提出ファイルの確認
    print('-' * 15, f'\nSub:')
    print(df_sub.dtypes)
    print(df_sub.head())
    pred_sample = df_sub['cite_gid'].values[0]
    print(pred_sample, '\n')
    
    # 酒のブランド(brand_id)と銘柄(meigara)の種類をプロット
    top_k = 20
    for col in ['brand_id', 'meigara']:
        print(f'訓練データ内の{col}の種類数: {df_train[col].nunique()}')
        ax = sns.countplot(x=df_train[col],
                        order=pd.value_counts(df_train[col]).iloc[:top_k].index)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR.joinpath(f'{col}.png'))
    
    # 銘柄が同じ画像サンプルを確認
    df_brand = df_train.groupby('meigara')['brand_id'].nunique().to_frame()
    print(df_brand[df_brand['brand_id'] > 1])
