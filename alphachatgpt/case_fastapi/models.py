from pydantic import BaseModel

class ApiR(object):
    """
    公共响应实体
    """

    def __init__(self, code:int, msg:str, data:any) -> None:
        self.code = code
        self.msg = msg
        self.data = data

    @staticmethod
    def ok(data:any = None):
        return ApiR(code= 0, msg="ok", data=data)
    
    @staticmethod
    def ok_list(list:list = []):
        return ApiR(
            code= 0, 
            msg="ok", 
            data={
                "list": list
            }
        )
    
    @staticmethod
    def error(code:int = 500, msg:str ="error", data:any = None):
        return ApiR(code=code, msg=msg, data=data)

class SubtitleEntity(BaseModel):
    """
    字幕
    """
    id: int 
    sub_id: int
    drama_id: int
    episode_id: int
    language: str 
    language_id: int
    url: str
    expire: int
    format: str