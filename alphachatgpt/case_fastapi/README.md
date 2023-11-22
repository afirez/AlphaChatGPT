# FastAPI 

## 安装

```bash
pip install fastapi
pip install uvicorn

```

## 运行

```bash
uvicorn main:app --reload
```


$ uvicorn main:app --reload

InFo: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
InFo: Started reloader process using 
InFo: Started server process 
InFo: Waiting for application startup.
InFo: Application startup complete.

## subtitle

| Languages/Language 支持的取值         | LanguageIds 支持的取值         | 说明 | App所传系统语言 http header里面的 locale 字段  |
| --- | --- | --- | --- |
| cmn-Hant-CN         | 36 | 繁体中文 | zh-Hant |
| eng-US | 2 | 英语 | en |
| jpn-JP         | 3 | 日语 | ja |
| kor-KR         | 4 | 韩语 | ko |
| por-PT         | 8 | 葡萄牙语 | pt |
| vie-VN         | 10 | 越南语 | vi |
| tha-TH         | 30 | 泰语 | th |
| ara-SA         | 34 | 阿拉伯语 | ar |
| hin-IN         | 42 | 印地语 | hi |
