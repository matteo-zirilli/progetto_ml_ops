from datasets import load_dataset
from transformers import AutoTokenizer
import os
import config as cf

hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", token =hf_token)




def tokenize_function(examples):
    
    # Il tokenizer prende in input il testo e i due parametri di per eseguire il padding e avere vettori di dimensione uguali, e se è un testo è troppo lungo rispetto ai limiti del modello HF tronca
    results = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True
    )
    
    
    return results




dataset = load_dataset("tweet_eval", "sentiment")


subset = dataset["train"].shuffle(seed=cf.RANDOM_SEED).select(range(100))
#salto il preprocessing qui perché eseguito su notebook di Colab



subset_tokenized = subset.map(
    tokenize_function, 
    batched=True
)



print(subset_tokenized.column_names)



