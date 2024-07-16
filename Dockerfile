FROM --platform=linux/amd64 python:3.11-slim-bullseye as base

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG PYTHON_VERSION=3.9
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV TZ="America/Los_Angeles"
ARG DEBIAN_FRONTEND="noninteractive"

WORKDIR /work

RUN apt-get update && apt-get install -y \
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
    python3-pip

RUN pip3 install -U pip
RUN pip3 install -U wheel setuptools

# Make the base image friendlier for interactive workloads. This makes things like the man command
# work.
RUN yes | unminimize

