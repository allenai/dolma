FROM python:3.11-slim-bullseye as base

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG PYTHON_VERSION=3.9
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV TZ="America/Los_Angeles"
ARG DEBIAN_FRONTEND="noninteractive"

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg2 \
    wget \
    awscli \
    build-essential \
    fuse \
    cmake \
    gcc \
    python3-pip \
    unzip \
    git \
    zlib1g-dev \
    pkg-config \
    apt-utils \
    automake \
    tree \
    htop \ 
    vim \
    less

RUN if [ $(uname -m) = "aarch64" ] ; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    else \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    fi

RUN unzip awscliv2.zip && ./aws/install 

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust.sh && sh rust.sh -y
RUN apt-get clean

ENV PATH="/root/.cargo/bin:${PATH}"
ENV PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
ENV LIBRARY_PATH="/usr/local/opt/zlib/lib"
ENV C_INCLUDE_PATH="/usr/local/opt/zlib/include"

RUN git clone https://github.com/allenai/dolma.git
RUN cd dolma && git checkout main && \
    git reset --hard ac91597158be50ed0239cdcfb6b9ec76d0b6becb && \ 
    pip install maturin && \
    maturin build --release && \
    pip install "$(find target/wheels -name '*.whl' | head -n 1)"[resiliparse,trafilatura] && \
    cd .. && \
    rm -rf dolma

# COPY ./python /work/python
# COPY ./src /work/src
# COPY ./Cargo.toml /work/Cargo.toml
# COPY ./Cargo.lock /work/Cargo.lock
# COPY ./pyproject.toml /work/pyproject.toml
# RUN pip install cmake "maturin[patchelf]>=1.1,<2.0"
# RUN maturin build --release 