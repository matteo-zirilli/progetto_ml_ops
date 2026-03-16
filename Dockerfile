
FROM python:3.12-slim

# Creo una cartella chiamata /code dentro il container e ci entro
# Da qui in poi, tutti i comandi verranno eseguiti dentro /code.
WORKDIR /code

# Copio il file requirements in /code
COPY requirements.txt .

# Dico al container di eseguire l'installazione da terminale delle librerie
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# copio la cartella "app" dentro una cartella "app" del container.
COPY ./app ./app

#l'app comunicherà sulla porta 8000.
EXPOSE 8000

#lancio uvicorn e la mia app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]