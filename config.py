import operator as o #mi serve per le soglie che definisco sotto


#definisco un seme per la riproducibilità
RANDOM_SEED = 42

#definisco i metadati su cui lavorerà il modello
metadata = {

    ###########################################CONFIGURAZIONE DATASET PUBBLICO#############################################
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

    ###########################################MODELLO BASELINE E FINE-TUNED#############################################
    "model": {

        "model_baseline": "cardiffnlp/twitter-roberta-base-sentiment-latest", #modello baseline
        "model_finetuned": "./models/model_fine_tuned", # cartella dove si trova il modello fine-tuned
        "model_production" = "./models/production_model"
    },

    "classification_metrics":{              #metriche che voglio misurare nel modello

        "class_1": "negative", #qui indico le classi e le metriche che voglio monitorare
        "metric_1": "recall",
        "class_2": "macro avg",
        "metric_2": "f1-score"

    }

}

###########################################SOGLIE PER CICD#############################################
#baso le soglie in base alle metriche scelte sopra
thresholds = {
    f"{metadata['classification_metrics']['class_1']}_{metadata['classification_metrics']['metric_1']}":{"value":0.8, "operator":o.ge},
    f"{metadata['classification_metrics']['class_2']}_{metadata['classification_metrics']['metric_2']}":{"value":0.7, "operator":o.ge}
}