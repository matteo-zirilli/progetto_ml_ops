from datasets import load_dataset
from app.model import SentimentModel #importo il modello
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Scarico il dataset (la variante 'sentiment')
dataset = load_dataset("tweet_eval", "sentiment")

# Stampo il dict  per vedere quante righe ci sono in ogni sezione
print("--- STRUTTURA DEL DATASET ---")
print(dataset)

# Guardo quali sono le colonne e che tipo di dati contengono
print("\n--- COLONNE  DEL SET DI TEST ---")
print(dataset["test"].features)

# Estraggo la prima riga del set di test per vederla
print("\n--- PRIMO TWEET  ---")
print(dataset["test"][0])


#prendo un subset del database appena importato, per evitare di andare contro limiti di memoria di Codespace
subset = dataset["test"].select(range(50))

X = subset["text"]
y = np.array(subset["label"])


#istanzio il modello
model = SentimentModel()
raw_predictions = model.predict(X)
#poiché nel modello Roberta le label sono negative, neutral e positive li rimappo in 0,1,2 come da dataset appena importato
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}


y_pred = np.array([label_mapping[pred["label"].lower()] for pred in raw_predictions])

# La magia di Scikit-Learn! Ora hai due array perfetti.
print("\n--- REPORT DI CLASSIFICAZIONE ---")
print(classification_report(y, y_pred))

print("\n--- MATRICE DI CONFUSIONE ---")
print(confusion_matrix(y, y_pred))

