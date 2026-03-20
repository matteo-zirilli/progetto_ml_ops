import config as cf
import json
import sys


#variabili di cofronto
finetuned_above_threshold = True
baseline_above_threshold = True
finetuned_beats_baseline = True

#APERTURA FILE METRICHE DEI MODELLI
with open("metrics_finetuned.json", "r") as f_in:
    data_finetuned = json.load(f_in)

with open("metrics_baseline.json", "r") as f_in:
    data_baseline = json.load(f_in)

# VALUTAZIONE SOGLIE E CONFRONTO DIRETTO
for key in cf.thresholds:
    operator = cf.thresholds[key]["operator"]
    threshold_value = cf.thresholds[key]["value"]

    # Valuto il Finetuned
    model_finetuned_value = data_finetuned[key]
    result_finetuned = operator(model_finetuned_value, threshold_value)
    
    if not result_finetuned:
        finetuned_above_threshold = False

    # Valuto la Baseline
    model_baseline_value = data_baseline[key]
    result_baseline = operator(model_baseline_value, threshold_value)
    
    if not result_baseline:
        baseline_above_threshold = False

    # Controllo se il finetuned è effettivamente migliore o uguale alla baseline
    # (Se anche solo in una metrica fa peggio, perde il confronto diretto)
    if model_finetuned_value < model_baseline_value:
        finetuned_beats_baseline = False

# CONFRONTO TRA MODELLI
print("\n--- RISULTATI CONFRONTO MODELLI ---")

if finetuned_above_threshold and finetuned_beats_baseline:
    print("VINCITORE: Modello Finetuned!")
    print("Motivazione: Supera le soglie minime di business e batte (o eguaglia) la Baseline.")
    
    # Scrivo il risultato per farlo leggere a GitHub Actions
    with open("winner.txt", "w") as f:
        f.write("finetuned")
    
    sys.exit(0) # Codice 0 = Successo, la pipeline continua

elif baseline_above_threshold:
    print("VINCITORE: Modello Baseline!")
    print("Motivazione: Il Finetuned non è migliorato o non ha superato le soglie. La Baseline rimane sicura.")
    
    with open("winner.txt", "w") as f:
        f.write("baseline")
        
    sys.exit(0) # Codice 0 = Successo, la pipeline continua pubblicando la baseline

else:
    print(" Nessun modello supera i requisiti minimi di business!")
    print("Azione: Deploy bloccato.")
    sys.exit(1) # Codice 1 = Errore fatale, la pipeline si ferma qui