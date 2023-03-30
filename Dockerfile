FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /workspaces/Scardina

RUN mv /etc/apt/sources.list.d /etc/apt/_sources.list.d \
 && apt update \
 && apt install -y --no-install-recommends curl \
 && apt-key del 7fa2af80 \
 && curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
 && dpkg -i cuda-keyring_1.0-1_all.deb \
 && mv /etc/apt/_sources.list.d /etc/apt/sources.list.d

RUN apt update \
 && apt install -y --no-install-recommends \
      tmux \
      htop \
      git \
      curl \
      moreutils \
      lsb-release \
      software-properties-common \
      ca-certificates \
      openssh-client \
      python3-pip \
      python3.8 \
      python3.8-distutils \
      libpython3.8-dev

RUN ln -f -s $(which python3.8) $(dirname $(which python3.8))/python3
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.0 python3 -
ENV PATH=$PATH:/root/.local/bin PYTHONPATH=/workspaces/Scardina
COPY pyproject.toml poetry.toml poetry.lock ./
# RUN poetry install

CMD poetry shell
