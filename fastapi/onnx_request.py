from pydantic import BaseModel


class OnnxRequest(BaseModel):

    data: list
