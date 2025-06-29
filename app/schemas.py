from pydantic import BaseModel
from typing import List, Optional

class PhotoSchema(BaseModel):
    photo_id: str
    url: str

class PredictionSchema(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]

class PostRequestSchema(BaseModel):
    post_id: str
    text: str
    photos: List[PhotoSchema]

class PostResponseSchema(BaseModel):
    post_id: str
    predictions: List[PredictionSchema]