# flac 转 MP3

python moviepy flac2mp3 

## 可打包成可执行文件

```bash
conda create -n v2 python=3.10
conda activate v2
conda install moviepy
conda install pyinstaller
```


```bash
pyinstaller --onefile flac2mp3.py
```