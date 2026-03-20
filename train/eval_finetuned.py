from app.model import SentimentModel #importo il modello
import json
from train.load_database import import_dataset
from train.eval_utils import report_model_evaluation

#valuto il modello re-trained per vedere se ancora risponde bene a frasi in input
model = SentimentModel("./models/model_fine_tuned")

#pipeline("text-classification", model="./models/model_fine_tuned", tokenizer = "./models/model_fine_tuned")

text_for_test = "I absolutely loved this project, it was fantastic!"

print(f"Valutazione frase: {text_for_test}\n")
print(model.predict(text_for_test))




#verifico ora le performance sul nuovo set riaddestrato sul set di test del db "tweet_eval"



#importo il dataset tweet_eval, categoria "sentiment", subset di "test"

X,y = import_dataset("tweet_eval", "test", "sentiment")

#istanzio il modello

#eseguo la valutazione del modello

model_fine_tuned_eval = report_model_evaluation(model, y, X)


#salvo le metriche f1_score e recall dentro un file json
with open("metrics_finetuned.json","w") as f:

    json.dump(model_fine_tuned_eval, f, indent = 4)
