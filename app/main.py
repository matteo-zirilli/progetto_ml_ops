from app.schemas import SentimentRequest, SentimentResponse
from app.model import SentimentModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import config as cf
import 
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
#istanzio l'app
app = FastAPI(title="Sentiment Analysis API")



#############################################METRICHE PER IL MONITORAGGIO###########################
# Contatore: Quante volte il modello ha predetto "positive" o "negative"?
PREDICTION_COUNTER = Counter(
    "model_predictions_total", 
    "Numero di predizioni effettuate per ogni classe", 
    ["label"]
)

# Verifico quanto è sicuro il modello delle sue risposte
CONFIDENCE_SCORE = Histogram(
    "model_confidence_score", 
    "Distribuzione del score di confidenza del modello"
)





###############################################CONSIDERAZIONI SU CICD#######################################
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





######################################################MODELLO E ENDPOINTS########################################
# Istanzio il modello
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
        result = analyzer.predict(request.text) #il modello fa la predizione

        # Estraggo i valori per le metriche
        label_predetta = result["label"]
        score_predetto = result["score"]

        PREDICTION_COUNTER.labels(label=label_predetta).inc() # Aggiunge +1 alla classe predetta
        CONFIDENCE_SCORE.observe(score_predetto)             # Registra la sicurezza del modello
        return result
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))


###########################################ATTIVAZIONE APP METRICHE DI MONITORAGGIO#####################################
Instrumentator().instrument(app).expose(app)