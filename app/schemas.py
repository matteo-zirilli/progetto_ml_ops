from pydantic import BaseModel


class SentimentRequest(BaseModel):

    """Schema di Input: la request deve essere una stringa"""

    text: str


class SentimentResponse(BaseModel):

    """Schema di Output: la response deve essere una stringa (per la label) e un float (per lo score)"""

    label: str
    score: float


