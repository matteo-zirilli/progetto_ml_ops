from app.model import SentimentModel #importo il modello
import json
from train.load_database import import_dataset
from train.eval_utils import report_model_evaluation
import config as cf




#importo i metadati da config

dataset = cf.metadata["dataset_id"]
split = cf.metadata["splits"]["test_split"]
config_name = cf.metadata["config_name"]
model_finetuned = cf.metadata["model"]["model_finetuned"]

#valuto il modello re-trained per vedere se ancora risponde bene a frasi in input
model = SentimentModel(model_finetuned)



text_for_test = "I absolutely loved this project, it was fantastic!"

print(f"Valutazione frase: {text_for_test}\n")
print(model.predict(text_for_test))




#verifico ora le performance sul nuovo set riaddestrato sul set di test del db 
#importo il dataset

X,y = import_dataset(dataset, split, config_name)

#istanzio il modello

#eseguo la valutazione del modello

model_fine_tuned_eval = report_model_evaluation(model, y, X)


#salvo le metriche f1_score e recall dentro un file json
with open("metrics_finetuned.json","w") as f:

    json.dump(model_fine_tuned_eval, f, indent = 4)



#RISULTATI
#vedendo i risultati non si ottengono risultati migliori rispetto al modello baseline. Questo è dovuto probabilmente al fatto
#che sto riaddestrando il modello su 5000 tweet mentre l'originale RoBERTa è addestrato su milioni di righe, da qui lo scarso risultato
#in effetti non è impostata una logica di append dell'informazione post riaddestramento