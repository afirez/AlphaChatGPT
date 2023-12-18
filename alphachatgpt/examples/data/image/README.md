# 长图 OCR

## tesseract 

### Install Windows tesseract (option)

Windows 安装，下载 exe 安装

https://digi.bib.uni-mannheim.de/tesseract/

这里下载 tesseract-ocr-w64-setup-v5.3.0.20221214.exe

另外安装中文的模型 chi_sim.traineddata，放在 Tesseract-OCR\tessdata 下

安装目录，比如 D:\Program Files\Tesseract-OCR 

### Python 依赖安装 

```
pip install Pillow
pip install pytesseract
```

### Run

python 运行

```
python py_ocr_img.py
```

### Docker Run

构建并启动镜像

```
docker-compose up --build
```

启动镜像

```
docker-compose up
```

