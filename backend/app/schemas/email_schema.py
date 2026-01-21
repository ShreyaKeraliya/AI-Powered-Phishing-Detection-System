from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional



class EmailModelType(str, Enum):
    tfidf_rf = "tfidf_rf"  # TF-IDF + RandomForest (adversarially robust)
    distilbert = "distilbert"


class EmailRequest(BaseModel):
    subject: str = Field("", description="Email subject")
    body: str = Field(..., description="Email body/content")
    model_type: EmailModelType = Field(
        EmailModelType.tfidf_rf,
        description="Model to use for prediction: tfidf_rf (adversarially robust RandomForest) or distilbert",
    )


class EmailResponse(BaseModel):
    label: str
    probability: float
    model_used: str
    explanations: list[str]
    processing_time: Optional[str] = None



