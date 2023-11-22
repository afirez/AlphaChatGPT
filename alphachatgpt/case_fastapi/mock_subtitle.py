import time

url_mock_1 = "https://raw.githubusercontent.com/afirez/assetsit/main/output1117.vtt"

"""
 create mode 100644 subtitles/ara-SA.vtt
 create mode 100644 subtitles/eng-US.vtt
 create mode 100644 subtitles/hin-IN.vtt
 create mode 100644 subtitles/jpn-JP.vtt
 create mode 100644 subtitles/kor-KR.vtt
 create mode 100644 subtitles/por-PT.vtt
 create mode 100644 subtitles/tha-TH.vtt
 create mode 100644 subtitles/vie-VN.vtt
"""

urls_mock = {
    "36": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/cmn-Hant-CN.vtt",
    "2": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/eng-US.vtt",
    "3": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/jpn-JP.vtt",
    "4": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/kor-KR.vtt",
    "8": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/por-PT.vtt",
    "10": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/vie-VN.vtt",
    "30": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/tha-TH.vtt",
    "34": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/ara-SA.vtt",
    "42": "https://raw.githubusercontent.com/afirez/assetsit/main/subtitles/hin-IN.vtt",
}

def expire_time():
    expire = time.time() * 1000 + (30 * 24 * 60 * 60 * 1000)
    return expire

def mock_subtitle_text():
    url = url_mock_1
    expire = expire_time()

    text = f"""
    [
        {
            "id": 0,
            "language": "cmn-Hans-CN",
            "language_id": 1,
            "url": "{url}",
            "expire": {expire},
            "format": "webvtt",
            "sub_id": 0
        },
        {
            "id": 2,
            "language": "rus-RU",
            "language_id": 5,
            "url": "{url}",
            "expire": {expire},
            "format": "webvtt",
            "sub_id": 429984091
        }
    ]          
    """
    return text

