FROM python:3.9.6-slim-buster
ENV PYTHONUNBUFFERED=1

RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    tesseract-ocr-chi-sim\
    build-essential \
    pkg-config \
    libpoppler-cpp-dev \
    poppler-utils \
    # build-essential \
    # libpq-dev \
    # git \
    # curl \
    && rm -rf /var/lib/apt/lists/*

# RUN which git

# install talib banary deps
# ENV PYTHON_TA_LIB_VERSION "0.4.24"
# RUN cd /tmp \
#     && curl -L -O http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
#     && tar -zxf ta-lib-0.4.0-src.tar.gz \
#     && cd ta-lib/ \
#     && ./configure --prefix=/usr \
#     && make \
#     && make install \
#     && pip3 install setuptools numpy \
#     && pip3 install ta-lib==${PYTHON_TA_LIB_VERSION} \
#     && rm -rf /tmp/* \

# RUN pip3 install TA-Lib-binary==0.4.21
# RUN pip3 install talib-binary
# RUN pip3 install TA-Lib-binary
# RUN pip3 install TA-Lib==0.4.21

# COPY requirements.txt .
# RUN python3.9 -m pip install --no-cache-dir --upgrade \
#     pip \
#     setuptools \
#     wheel
# RUN python3.9 -m pip install --no-cache-dir \
#     -r requirements.txt

WORKDIR /app
COPY requirements.txt requirements.txt
# RUN pip3 install --upgrade pip
# RUN pip3 install -r requirements.txt
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


# COPY TA_Lib-0.4.24-cp39-cp39-win_amd64.whl TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
# RUN pip3 install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl

COPY . /app
EXPOSE 8000