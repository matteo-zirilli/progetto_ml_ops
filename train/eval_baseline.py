from app.model import SentimentModel #importo il modello
import json
from train.load_database import import_dataset
from train.eval_utils import report_model_evaluation
import config as cf


#importo i metadati da config

dataset = cf.metadata["dataset_id"]
split = cf.metadata["splits"]["test_split"]
config_name = cf.metadata["config_name"]
model_baseline = cf.metadata["model_baseline"]


X, y = import_dataset(dataset, split, config_name)


#istanzio il modello
model = SentimentModel(model_baseline)

#eseguo la valutazione del modello

roberta_eval = report_model_evaluation(model, y, X)


#salvo le metriche f1_score e recall dentro un file json
with open("metrics_baseline.json","w") as f:

    json.dump(roberta_eval, f, indent = 4)
