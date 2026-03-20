from app.schemas import SentimentRequest, SentimentResponse
from app.model import SentimentModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import config as cf

#istanzio l'app
app = FastAPI(title="Sentiment Analysis API")


#importo i metadati da config
model = cf.metadata["model"]["model_baseline"]



#istanzio il modello
analyzer = SentimentModel(model)

#definisco la root
@app.get("/")
def root():
    content = """<h1>WELCOME!!Here you can test any text or chat or reviews to classify as Positive, Neutral or Negative</h1>
                <a href="/docs">Click here to test the API</a>"""
    return HTMLResponse(content = content)

#poiché devo inviare dei dati al server (un testo) creo un endpoint di post
@app.post("/predict", response_model = SentimentResponse)


def analyze_sentiment(request: SentimentRequest):

    try:
        result = analyzer.predict(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
