FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        ffmpeg \
        wget \
        libpulse-dev \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt