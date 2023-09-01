# Sake Image Retrieval
This repository is a solution of [Competition held on Nishika](https://competition.nishika.com/competitions/sake/summary).

## 🛠Requirements
- python = "^3.10"
- pandas = "^2.0.3"
- matplotlib = "^3.7.2"
- seaborn = "^0.12.2"
- japanize-matplotlib = "^1.1.3"
- torch = "2.0.0"
- pillow = "^10.0.0"
- torchinfo = "^1.8.0"
- timm = "^0.9.5"
- albumentations = "^1.3.1"
- faiss-gpu = "^1.7.2"

## 🌲Directory
<pre>
sake_image_retrieval
├───data
│   ├───cite_images   : 引用画像
│   └───query_images  : クエリ画像
│
├───environments      : Dockerfileなどの実行環境
│
├───features          : 中間特徴量
│
├───index             : 引用画像の特徴量から作成したインデックス
│
├───outputs           : 提出ファイル
│
└───src               : ソースコード
</pre>

## ⚙️Installation
Clone this repository.
```bash
git clone https://github.com/kargenk/sake_image_retrieval.git
```

### Using Poetry
Install Poetry:
```bash
# Install the Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -
```

Create environment with Poetry:
```bash
cd sake_image_retrieval/src

# Install the project dependencies and Activate
poetry install
poetry shell
```

## 💻Usage
<!-- Make mock data and solve the problem:
```bash
# preparation data for Linear Programing
python generate_mock_data.py

# Solve the problem
python integer_programming.py
```
When you execute `mock_data.py`, you can see mock data files in `data/toy`.
`zeroth_continuous`, `first`, and `second` mean continuous lecture, first semester, and second semester respectively.
And executing `integer_programming.py`, also see optimized timetable file in `outputs/toy/`. -->

<!-- ## 📝Note

### Execution Environments
- OS: Ubuntu 20.04
- CPU: 12th Gen Intel(R) Core(TM) i9-12900(16 cores, 24 threads)
- Memory: 64GB

### Calculation Time⌛
In the above environment, takes
- N \[sec\] (without p_lowers, p_uppers)
- N \[sec\] (with p_lowers, p_uppers)
![calculation Time](api/img/calculation_time.png) -->

## 🚀Updates
**2023.09.01**
- add ConvNeXt(ImageNet pretrain)

## 📧Authors
kargenk a.k.a **gengen**(https://twitter.com/gengen_ml)

## ©License
This repository is free, but I would appreciate it if you could inform the author when you use it.
<!-- under [MIT licence](https://en.wikipedia.org/wiki/MIT_License) -->