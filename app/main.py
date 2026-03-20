from app.schemas import SentimentRequest, SentimentResponse
from app.model import SentimentModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import config as cf
import os
#istanzio l'app
app = FastAPI(title="Sentiment Analysis API")


# Controllo se la CI/CD  ha creato il file winner.txt
if os.path.exists("winner.txt"):
    with open("winner.txt", "r") as f:
        vincitore = f.read().strip() # .strip() toglie eventuali spazi vuoti
else:
    vincitore = "baseline" # Fallback di sicurezza se si lancia in locale senza file

# Assegno il percorso corretto in base al vincitore
if vincitore == "finetuned":
    # Prendo il percorso del finetuned dal tuo config ("./models/model_fine_tuned")
    model_path = cf.metadata["model"]["model_finetuned"] 
else:
    # Prendo la stringa di Hugging Face dal config
    model_path = cf.metadata["model"]["model_baseline"]

# 3. Istanziamo il modello come facevi già
analyzer = SentimentModel(model_path)

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
