from pydantic import BaseModel, HttpUrl, Field
from typing import Optional


class UrlRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")


class UrlResponse(BaseModel):
    label: str
    probability: float
    important_features: dict
    processing_time: Optional[str] = None



