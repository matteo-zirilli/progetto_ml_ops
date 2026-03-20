from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import config as cf




def report_model_evaluation(model, y, X):


    
    #salvo dentro la lista le mie predizioni tenendo a mente che il modello accetta solo singole stringhe
    raw_predictions =[model.predict(x) for x in X]
    #poiché nel modello Roberta e in quello Re-trained le label sono negative, neutral e positive li rimappo in 0,1,2 come da dataset appena importato
    label_mapping = cf.metadata["label_encoding"]
    label = cf.metadata["column_mapping"]["label_feature"]


    y_pred = np.array([label_mapping[pred[label].lower()] for pred in raw_predictions])

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

    recall_neg_val = report_dict["0"]["recall"]
    f1_macro_val = report_dict["macro avg"]["f1-score"]


    metrics_to_save = {

        "negative_recall":recall_neg_val,
        "macro_f1": f1_macro_val

    }



    return metrics_to_save