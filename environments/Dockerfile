# ベースとなるイメージを読み込む
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# パッケージ一覧を更新
RUN apt-get update
# RUN apt-get -y upgrade

# 必要なパッケージのインストール
RUN apt-get install -y --no-install-recommends \
    sudo \
    screen \
    vim \
    wget \
    curl \
    git \
    zsh \
    unzip \
    x11-apps

# ユーザ作成
ARG USERNAME=test
ARG GROUPNAME=test
ARG UID=1000
ARG GID=1000
ARG PASSWORD=test
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME -G sudo && \
    echo ${USERNAME}:${PASSWORD} | chpasswd
RUN echo 'Defaults visiblepw'                 >> /etc/sudoers
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /home/$USERNAME

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# shellをbashに変える
SHELL ["/bin/bash", "-c"]

# zshのセッティング($HOME/.zshrc内のZSH_THEMEを"agnoster"に変更すると見た目良し)
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN apt-get install -y fonts-powerline language-pack-en
RUN update-locale

# pyenvのセッティング
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl

RUN curl https://pyenv.run | zsh && \
    echo '' >> $HOME/.zshrc && \
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $HOME/.zshrc && \
    echo 'eval "$(pyenv init -)"' >> $HOME/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.zshrc

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.10.9 && \
    pyenv global 3.10.9 && \
    pyenv rehash

# Poetryで環境を構築
RUN pip install poetry
COPY pyproject.toml* poetry.lock* ./
RUN poetry config virtualenvs.in-project true
RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

# # 設定ファイル群をユーザー権限に
# RUN chown -R $UID:$GID $PYENV_ROOT '.oh-my-zsh/' '.zshrc' '.wget-hsts'

# 指定ユーザーに変更
USER $USERNAME
WORKDIR /home/$USERNAME/
RUN export PATH="$PATH:$HOME/.local/bin"

CMD ["/bin/zsh"]