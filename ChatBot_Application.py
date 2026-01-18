import nltk
nltk.download('punkt')
nltk.download('punkt_tab')   # <-- this is the missing one
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model

model = load_model('chatbot_Application_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))


def bank_of_words(s,words, show_details=True):
    bag_of_words = [0 for _ in range(len(words))]
    sent_words = nltk.word_tokenize(s)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    for sent in sent_words:
        for i,w in enumerate(words):
            if w == sent:
                bag_of_words[i] = 1
    return np.array(bag_of_words)

def predict_label(s, model):
    pred = bank_of_words(s, words, show_details=False)
    response = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.75
    final_results = [[i, r] for i, r in enumerate(response) if r > ERROR_THRESHOLD]
    final_results.sort(key=lambda x: x[1], reverse=True)

    if not final_results:
        return [{"intent": "fallback", "probability": "1.0"}]

    return [{"intent": labels[r[0]], "probability": str(r[1])} for r in final_results]



def Response(ints, intents_json):
    if not ints:  # if prediction is empty
        return "Sorry, I didn’t understand that. Try asking about drug reactions, blood pressure, hospitals, or pharmacies."

    tags = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tags:
            return random.choice(i['responses'])
    
    # If no tag matched, use fallback
    return "Sorry, I didn’t get that. Can you rephrase?"

def chatbot_response(msg):
    ints = predict_label(msg, model)
    response = Response(ints, intents)
    return response
            
def chat():
    print("Start chat with ChatBot")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        response = chatbot_response(inp)
        print("\n BOT: " + response + '\n\n')

chat()
        
