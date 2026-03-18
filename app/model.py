from transformers import pipeline
import os


class SentimentModel():

    """
    Classe per definire dato un testo un analisi sentiment in base al testo ricevuto in input
    """

    def __init__(self):

        """
        Si definisce il metodo costruttore del modello di Sentiment Analysis che salva dentro "model" il modello di Hugging Face ROBERTA
        """

        # Recupero il token dall'ambiente
        hf_token = os.environ.get("HF_TOKEN")
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", token =hf_token)

    def predict(self, text: str):

        """
        Si definisce il metodo predict per restituire la previsione del modello dato un testo
        """

        if text.strip() == "":
            raise ValueError("Stringa vuota: non classificabile come categoria positiva, negativa o neutra")
        else:
            lista_previsione = self.model(text) #self.model(text) restituisce una lista con un solo elemento che è il dizionario in output dal modello
            return lista_previsione[0] #seleziono l'elemento della lista per ritornare solo il dizionario della previsione