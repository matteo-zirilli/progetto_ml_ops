from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import os
import config as cf


#importo i metadati da config

dataset_name = cf.metadata["dataset_id"]
split = cf.metadata["splits"]["train_split"]
config_name = cf.metadata["config_name"]
model_baseline = cf.metadata["model"]["model_baseline"]
model_finetuned= cf.metadata["model"]["model_finetuned"]
column_mapping = cf.metadata["column_mapping"]["text_feature"]
num_labels = len(cf.metadata["label_encoding"])

#token di lettura da HF per autenticazione
hf_token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_baseline, token =hf_token)



#FASE DI TOKENIZZAZIONE TWEETS
def tokenize_function(examples):
    
    # Il tokenizer prende in input il testo,  se è un testo è troppo lungo rispetto ai limiti del modello HF tronca
    results = tokenizer(
        examples[column_mapping], 
        truncation=True
    )
    
    
    return results



#scarico il dataset dalla libreria di HF
dataset = load_dataset(dataset_name, config_name)

#seleziono un subset di 500 righe dal train
subset = dataset[split].shuffle(seed=cf.RANDOM_SEED).select(range(5000))
#salto il preprocessing qui perché eseguito su notebook di Colab


#tokenizzo il subset e credo il dizionario
subset_tokenized = subset.map(
    tokenize_function, 
    batched=True
)






#FASE DI TRAINING

#scarico il modello di rete neurale pre addestrato da HF con AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_baseline, num_labels = num_labels,token =hf_token)

# definisco le regole di addestramento (epoche, batch size)

training_args = TrainingArguments(  output_dir = model_finetuned, 
                                    num_train_epochs = 2, 
                                    per_device_train_batch_size = 10,
                                    learning_rate=2e-5,
                                    weight_decay=0.01)

#per far si che il trainer prenda batch di tweet tokenizzati esattamente con la stessa dimensione istanzio la seguente classe

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#sfrutto la classe Trainer per addestrare la rete passandogli il modello dati "tokenizzati" con le regole definite
#istanzio la classe Trainer con gli argomenti definiti finora
trainer = Trainer(  model = model, 
                    args = training_args, 
                    train_dataset = subset_tokenized, 
                    data_collator= data_collator, 
                    processing_class = tokenizer)
#eseguo il train
trainer.train()
trainer.save_model(output_dir = model_finetuned)