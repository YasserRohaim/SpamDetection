from fastapi import FastAPI
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
app=FastAPI()


with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
model = load_model("model.keras")
@app.get("/")
def model_inference(new_email:str):
    

   
    new_sequence = tokenizer.texts_to_sequences([new_email])  
    new_padded_sequence = pad_sequences(new_sequence, maxlen=100) 

    predictions = model.predict(new_padded_sequence)
    predicted_label = (predictions > 0.5).astype(int)  

    prediction = label_encoder.inverse_transform(predicted_label)[0]
  
    return {"prediction": prediction}

