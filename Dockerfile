# cuda11.1.0, pytorch1.8 the base image is on nvcr, please login nvcr.io before using it 
# or you can build up your own pytorch env
FROM nvcr.io/nvidia/pytorch:20.11-py3

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python
RUN pip install future tensorboard 

