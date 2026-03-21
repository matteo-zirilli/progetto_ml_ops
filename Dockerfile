
FROM python:3.12-slim

# Creo una cartella chiamata /code dentro il container e ci entro
# Da qui in poi, tutti i comandi verranno eseguiti dentro /code.
WORKDIR /code

# Copio il file requirements in /code
COPY requirements.txt .

# Dico al container di eseguire l'installazione da terminale delle librerie, installo una versione di pythorch poco pesante per Git
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copio il file di configurazione
COPY config.py .

# Copio il file winner (uso l'asterisco così se per caso in locale non esiste, Docker non va in errore!)
COPY winner.txt* .

# Copio la cartella dei modelli (dove ci sarà il finetuned scaricato dalla CI/CD)
COPY ./models ./models


# copio la cartella "app" dentro una cartella "app" del container.
COPY ./app ./app

#l'app comunicherà sulla porta 7860
EXPOSE 7860

#lancio uvicorn e la mia app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]