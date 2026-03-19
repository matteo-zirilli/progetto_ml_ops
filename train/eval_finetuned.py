from transformers import pipeline

#valuto il modello re-trained per vedere se ancora risponde bene a frasi in input
classificatore = pipeline("text-classification", model="./models/model_fine_tuned", tokenizer = "./models/model_fine_tuned")

text_for_test = "I absolutely loved this project, it was fantastic!"

print(f"Valutazione frase: {text_for_test}\n")
print(classificatore(text_for_test))




#verifico ora le performance sul nuovo set riaddestrato sul set di test del db "tweet_eval"
