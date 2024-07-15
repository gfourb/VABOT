# app.py
from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(spell.correction(word.lower()) if spell.correction(word.lower()) is not None else word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.70  # Increased threshold to ensure better matching
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return results
    else:
        return []

def get_response(ints, intents_json):
    if ints:
        tag = classes[ints[0][0]]
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    return "I'm sorry, I didn't understand that. Could you please rephrase?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form["message"]
    ints = predict_class(user_input, model)
    response = get_response(ints, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
