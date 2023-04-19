FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /home



RUN echo 'deb http://mirrors.cloud.tencent.com/ubuntu/ focal main restricted universe multiverse\n\
deb http://mirrors.cloud.tencent.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb http://mirrors.cloud.tencent.com/ubuntu/ focal-updates main restricted universe multiverse\n\
#deb http://mirrors.cloud.tencent.com/ubuntu/ focal-proposed main restricted universe multiverse\n\
#deb http://mirrors.cloud.tencent.com/ubuntu/ focal-backports main restricted universe multiverse\n\
deb-src http://mirrors.cloud.tencent.com/ubuntu/ focal main restricted universe multiverse\n\
deb-src http://mirrors.cloud.tencent.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb-src http://mirrors.cloud.tencent.com/ubuntu/ focal-updates main restricted universe multiverse\n'\
        > /etc/apt/sources.list && \
        apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai && apt-get install -y --no-install-recommends \
        git                            \
        npm                            \
        wget                           \
        python3                        \
        python3-dev                    \
        python3-pip                    \  
        python3-opencv              && \
        apt-get autoremove          && \          
        apt-get clean               && \                
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --no-cache-dir    \
        pip                            \
        setuptools                  && \
        python3 -m pip install --upgrade pip


RUN git clone http://deepmaterial.work/git/luziqing/segment_anything_streamlit.git && cd segment_anything_streamlit && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    cd streamlit_dc/streamlit_drawable_canvas/frontend && npm install && npm run build && cd ../../ && pip install -e . && \
    cd /home/segment_anything_streamlit && mkdir checkpoint && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoint\sam_vit_b_01ec64.pth && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoint\sam_vit_l_0b3195.pth && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoint\sam_vit_h_4b8939.pth


WORKDIR /home/segment_anything_streamlit
