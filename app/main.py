from app.schemas import SentimentRequest, SentimentResponse
from app.model import SentimentModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

#istanzio l'app
app = FastAPI(title="Sentiment Analysis API")

#istanzio il modello
analyzer = SentimentModel()

#definisco la root
@app.get("/")
def root():
    content = """<h1>BENVENUTO AL SISTEMA PER LA VALUTAZIONE DELLA REPUTAZIONE DELLA TUA AZIENDA</h1>
                <a href="/docs">Clicca qui per testare la API</a>"""
    return HTMLResponse(content = content)

#poiché devo inviare dei dati al server (un testo) creo un endpoint di post
@app.post("/predict", response_model = SentimentResponse)


def analyze_sentiment(request: SentimentRequest):

    try:
        result = analyzer.predict(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
