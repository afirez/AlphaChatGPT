from fastapi import FastAPI

import db 

from models import ApiR, SubtitleEntity

import mock_subtitle

app = FastAPI()


@app.on_event("startup")
async def startup():
    await db.database.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.database.disconnect()




subtitle_list = [
    SubtitleEntity(
        id=0,
        drama_id = 1,
        episode_id = 1,
        language="en",
        language_id="0",
        url=mock_subtitle.url_mock_1,
        expire=int(mock_subtitle.expire_time()),
        format="webvtt",
        sub_id="0",
    ),
    SubtitleEntity(
        id=1,
        drama_id = 1,
        episode_id = 1,
        language="zh-hant",
        language_id="1",
        url=mock_subtitle.url_mock_1,
        expire=int(mock_subtitle.expire_time()),
        format="webvtt",
        sub_id="1",
    ),
]


@app.get("/")
def read_root():
    return {"msg": "Hello World!"}


@app.get("/item/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: SubtitleEntity):
    return {"item_name": item.language, "item_id": item_id}


@app.get("/subtitle/list")
def get_subtitle_list():
    return ApiR.ok_list(list=subtitle_list)
    # return subtitle_list

@app.get("/subtitle/{sub_id}")
def get_subtitle_by_sub_id(sub_id: int):
    items = [item for item in subtitle_list if item.sub_id == sub_id]
    if len(items) > 0:
        return ApiR.ok(data=items[0])
    return ApiR.ok()

@app.post("/subtitle/add")
def update_item(item: SubtitleEntity):
    subtitle_list.append(item)
    return ApiR.ok()