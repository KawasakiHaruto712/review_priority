FROM python:3.12.4

WORKDIR /work

RUN apt-get update &&\
    apt-get -y install --no-install-recommends locales &&\
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 &&\
    rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=JST-9
ENV TERM=xterm

# バインドマウントした data/ 内の git リポジトリ（releases_repo 等）はホストユーザ所有なので、
# コンテナ内 root から触ると Git の dubious ownership 保護で拒否される。全ディレクトリを安全扱いにする。
RUN git config --system --add safe.directory '*'

# ソースコードは実行時に docker-compose のバインドマウントで /work に展開されるため、
# ビルド時に必要なのは依存関係の定義ファイルだけ
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt
