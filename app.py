from flask import Flask, request, jsonify, render_template, send_from_directory
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import logging
import sys
import os
import urllib.parse

app = Flask(__name__, static_folder='static')

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

try:
    with open('intents.json', encoding='utf-8') as file:
        intents = json.load(file)
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    model = load_model('chatbot_model.h5')
except Exception as e:
    logger.error(f"Error loading files: {e}")
    raise

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    logger.debug(f"Tokenized and lemmatized sentence: {sentence_words}")
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    logger.debug(f"Bag of words: {bag}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    logger.debug(f"Model prediction: {res}")
    ERROR_THRESHOLD = 0.80  # Increased threshold to ensure better matching
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Prediction results: {results}")
    return results if results else []

def get_response(ints, intents_json):
    if ints:
        tag = classes[ints[0][0]]
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    # Fallback message with clickable options
    fallback_message = "I'm not sure I understood. Maybe one of these topics can help you:<br><a href=\"#\" onclick=\"sendMessageTag('onboarding')\">Onboarding</a><br><a href=\"#\" onclick=\"sendMessageTag('faqs')\">FAQs</a><br><a href=\"#\" onclick=\"sendMessageTag('tools')\">Tools</a>"
    return fallback_message

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form.get("message")
    if not user_input:
        return jsonify({"response": "No input received. Please type a message."})
    
    logger.info(f"Received user input: {user_input}")
    try:
        ints = predict_class(user_input, model)
        logger.info(f"Predicted classes: {ints}")
        response = get_response(ints, intents)
        logger.info(f"Generated response: {response}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        response = "There was an error processing your request. Please try again."

    return jsonify({"response": response})

if __name__ == "__main__":
    # Ensure the application runs with UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    app.run(debug=True)
