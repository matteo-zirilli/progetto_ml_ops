---
title: Sentiment Analysis API
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---



# MLOps Sentiment Analysis API

Questo progetto implementa una pipeline MLOps completa (End-to-End) per l'addestramento, la valutazione automatica, il deploy e il monitoraggio di un modello di Sentiment Analysis (RoBERTa).

## Panoramica dell'Architettura
Il sistema non si limita a esporre un modello tramite API, ma gestisce l'intero ciclo di Continuous Integration e Continuous Deployment (CI/CD) in modo automatizzato. 

I pilastri del progetto sono:
0. **File di Configurazione:** File `config.py` dove sono gestite tutte le parametrizzazioni del progetto (seme random, metadati dei modelli baseline e fine-tuned, dataset scelto per il fine-tuned, caratteristiche del datase, soglie di confronto tra modelli).
1. **Model Training:** Script Python per il fine-tuning di un modello RoBERTa.
2. **Automated Gate Check:** Un sistema decisionale che confronta le performance del modello appena addestrato con una Baseline solida. Solo il modello migliore andrà in produzione.
3. **Containerizzazione:** L'app FastAPI è pacchettizzata tramite Docker.
4. **CI/CD Pipeline:** Implementata con GitHub Actions. Ogni *push* sul branch `main` innesca un addestramento, la valutazione e il deploy automatico su Hugging Face Spaces.
5. **Monitoraggio:** Endpoint `/metrics` esposto per la raccolta dati tramite Prometheus e la visualizzazione su Grafana per scovare anomalie operative o performance deludenti del modello.

## Struttura del Repository
- `/app` -> Contiene il codice dell'applicazione FastAPI (`main.py`) e gli schemi Pydantic.
- `/tests` -> Test di integrazione (`pytest`) che validano l'API prima del deploy.
- `/models` -> (Generata dinamicamente) Contiene i pesi del modello vincitore.
- `.github/workflows/sync.yml` -> Il cuore dell'automazione CI/CD.
- `gate_check.py` -> Lo script che valuta matematicamente chi vince tra Finetuned e Baseline.
- `docker-compose.yml` & `prometheus.yml` -> Infrastruttura di monitoraggio locale.

## Come funziona la Pipeline (Il Flusso)
1. Lo sviluppatore fa una modifica al codice e lancia `git push`.
2. **GitHub Actions** prende in carico il lavoro:
   - Installa le dipendenze.
   - Esegue il *Training* su un subset di dati (Test per validare l'infrastruttura). Se si vuole disabilitare la parte automatizzata di Training basta eliminare il relativo lancio in sync.yml. Il Training per la Safety del progetto che vive su risorse limitate è fatto su un dataset piccolo (100 righe), quindi sicuramente impostato così il baseline (RoBERTa) sarà migliore del modello fine-tuned.
   - Esegue il *Gate Check*: calcola l'accuratezza e genera il file `winner.txt` (vincitore tra modello baseline (RoBERTa) e modello fine-tuned).
   - Esegue i *Test di Integrazione* ricaricando dinamicamente il modello vincitore nell'API.
   - Pulisce i checkpoint inutili e carica i file pesanti tramite **Git LFS** (con il pacchetto base di GitHub e HF non viene supportato il pushing di file troppo pesanti).
   - Spinge il container Docker su **Hugging Face Spaces**.
3. L'API va online esponendo Swagger UI e l'endpoint Prometheus.

## Monitoraggio in Produzione (Grafana)
L'app utilizza `prometheus-fastapi-instrumentator` per tracciare le seguenti metriche MLOps:
- `model_predictions_total`: Contatore delle predizioni divise per label (positive/negative). Utile per monitorare le performance del modello.
- `model_confidence_score`: Istogramma che traccia la sicurezza del modello. Un calo della confidenza media indica necessità di riaddestramento.
- Metriche operative standard: latenza HTTP, tassi di errore 4xx/5xx.
- E'stato esportato un esempio di dashboard creata su Graphana nel file YML `export_dashboard_from_graphana`
- Dopo aver lanciato il compose con il comando seguente ed essere entrato su Graphana con `admin`, andare su Connections -> Data Sources -> Add data source -> Prometheus. Nel campo URL scrivere http://prometheus:9090 e cliccare su Save & Test.

**Per avviare il monitoraggio localmente:**
```bash
docker compose up -d