FROM continuumio/miniconda3:latest
RUN apt-get update && apt-get install -y libgl1

# 컨테이너 내 작업 디렉토리
WORKDIR /work

# environment.yml 복사
COPY ./environ/environment.yml /work/environment.yml

# Conda 환경 생성
RUN conda env create -f environment.yml

# 기본 쉘 설정 (bash)
SHELL ["/bin/bash", "--login", "-c"]

ARG ENV_NAME=deepfakehunters

# 컨테이너 진입 시 자동 활성화되도록
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc

EXPOSE 8000
