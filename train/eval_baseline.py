from app.model import SentimentModel #importo il modello
import json
from train.load_database import import_dataset
from train.eval_utils import report_model_evaluation


#importo il dataset tweet_eval, categoria "sentiment", subset di "test"

X, y = import_dataset("tweet_eval", "test", "sentiment")


#istanzio il modello
model = SentimentModel("cardiffnlp/twitter-roberta-base-sentiment-latest")

#eseguo la valutazione del modello

roberta_eval = report_model_evaluation(model, y, X)


#salvo le metriche f1_score e recall dentro un file json
with open("metrics_baseline.json","w") as f:

    json.dump(roberta_eval, f, indent = 4)
