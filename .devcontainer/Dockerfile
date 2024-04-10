FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    vim \
    build-essential \
    libtool \
    autoconf \
    unzip \
    wget \
    pkg-config \
    libssl-dev

RUN if [ $(uname -m) = "aarch64" ] ; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    else \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    fi

RUN unzip awscliv2.zip && ./aws/install

USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /workspaces

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
ENV DEBIAN_FRONTEND noninteractive
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b

ENV PATH="/home/vscode/miniforge3/bin:${PATH}"

RUN mkdir -p /workspaces/dolma
COPY ../python /workspaces/dolma/python
COPY ../src /workspaces/dolma/src
COPY ../Cargo.toml /workspaces/dolma/Cargo.toml
COPY ../Cargo.lock /workspaces/dolma/Cargo.lock
COPY ../pyproject.toml /workspaces/dolma/pyproject.toml

COPY .devcontainer/postInstall.sh .
RUN ./postInstall.sh

RUN echo "source activate base" >> ~/.bashrc
RUN echo "echo 'Happy hacking!\n'" >> ~/.bashrc
