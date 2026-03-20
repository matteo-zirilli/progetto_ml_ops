#definisco un seme per la riproducibilità
RANDOM_SEED = 42

#definisco i metadati su cui lavorerà il modello
metadata = {

    "dataset_id": "tweet_eval", #nome del dataset
    "config_name": "sentiment", #categoria o configurazione del dataset
    "splits": {

        "train_split": "train", #per dataset con split (come RoBERTa ad esempio) scelta per gli split di train e test
        "test_split": "test"
    },
    "column_mapping": {

        "text_feature": "text", #colonna delle feature
        "label_feature": "label" #colonna con la label
    },

    "label_encoding": {"negative": 0, "neutral": 1, "positive": 2},

    "model": {

        "model_baseline": "cardiffnlp/twitter-roberta-base-sentiment-latest", #modello baseline
        "model_finetuned": "./models/model_fine_tuned" # cartella dove si trova il modello fine-tuned
    }
}