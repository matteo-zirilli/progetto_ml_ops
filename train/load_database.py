from datasets import load_dataset
import config as cf
import numpy as np

def import_dataset(dataset_name:str,data_subset: str, category: str):
# Scarico il dataset (la variante "category")
    dataset = load_dataset(dataset_name, category)

    # Stampo il dict  per vedere quante righe ci sono in ogni sezione
    print("--- STRUTTURA DEL DATASET ---")
    print(dataset)

    # Guardo quali sono le colonne e che tipo di dati contengono il data subset
    print(f"\n--- COLONNE  DEL SUBSET {data_subset} ---")
    print(dataset[data_subset].features)

    # Estraggo la prima riga del set di test per vederla
    print("\n--- PRIMO TWEET  ---")
    print(dataset[data_subset][0])


    #prendo un subset del database appena importato, per evitare di andare contro limiti di memoria di Codespace e mischio il set 
    subset = dataset[data_subset].shuffle(seed=cf.RANDOM_SEED).select(range(500))

    X = list(subset["text"])
    y = np.array(subset["label"])

    

    return X,y