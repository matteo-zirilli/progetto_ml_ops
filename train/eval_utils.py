from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import config as cf




def report_model_evaluation(model, y, X):


    
    #salvo dentro la lista le mie predizioni tenendo a mente che il modello accetta solo singole stringhe
    raw_predictions =[model.predict(x) for x in X]
    #poiché nel modello Roberta e in quello Re-trained le label sono negative, neutral e positive li rimappo in 0,1,2 come da dataset appena importato
    label_mapping = cf.metadata["label_encoding"]
 


    y_pred = np.array([label_mapping[pred["label"].lower()] for pred in raw_predictions])

    print("\n--- MATRICE DI CONFUSIONE ---")
    print(confusion_matrix(y, y_pred))



    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y, y_pred))


    #salvo le metriche che mi interessano per il progetto
    #il progetto richiede per l'azienda il monitoraggio dei feedback degli utenti
    #quindi l'azienda sarà più sensibile ai feedback negativi piuttosto che quelli positivi
    #ovvero se un feedback è negativo l'azienda deve attenzionare quello che è il suo operato
    #di conseguenza ritengo che sia meglio ottimizzare nel modello la Recall sui Negativi e in generale anche F1-score
    #questo perché è meglio considerare qualche feedback negativo, investigarsi e rendersi conto che in realtà era un feedback positivo piuttosto che il contrario


    #salvo il report in un dizionario
    report_dict= classification_report(y, y_pred, output_dict=True)

    #recupero da config le metriche che voglio monitorare e salvare
    class_1 = cf.metadata["classification_metrics"]["class_1"]
    label_encoding = cf.metadata["label_encoding"]
    index_label = label_encoding[class_1]
    metric_1 = cf.metadata["classification_metrics"]["metric_1"]
    class_2 = cf.metadata["classification_metrics"]["class_2"]
    metric_2 = cf.metadata["classification_metrics"]["metric_2"]

    metric_val_1 = report_dict[f"{index_label}"][metric_1]
    metric_val_2 = report_dict[class_2][metric_2]


    metrics_to_save = {

        f"{class_1}_{metric_1}":metric_val_1,
        f"{class_2}_{metric_2}": metric_val_2

    }



    return metrics_to_save