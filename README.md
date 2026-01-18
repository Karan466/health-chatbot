# 🧠 Healthcare Chatbot using NLP & Deep Learning  

## 📌 Overview  
This project is a simple **healthcare chatbot** built using **Natural Language Processing (NLP)** and **Deep Learning**.  
It can handle small conversations such as greetings, thanks, and healthcare-related queries including:  
- 💊 Adverse drug reactions  
- ❤️ Blood pressure information  
- 🏥 Hospitals guidance  
- 💊 Pharmacies guidance  

The chatbot is trained on a custom **intents.json** dataset and uses a **feedforward neural network** for intent classification.  
It also includes a **fallback mechanism** for unknown queries.  

---

## ⚙️ Features  
- Uses **NLTK** for tokenization and lemmatization  
- Converts text into **Bag-of-Words (BoW)** vectors  
- Trains a **Multilayer Perceptron (MLP)** with Keras + TensorFlow  
- Supports a **fallback intent** when no match is found  
- Lightweight, retrainable with your own dataset  

---

## 🛠️ Tech Stack  
- **Python 3.10+**  
- **NLTK** (Natural Language Toolkit)  
- **TensorFlow / Keras**  
- **NumPy**  
- **Pickle**  

---

## 📂 Project Structure  
```
chatbot-project-source-code/
│
├── Souce Code/
│   ├── ChatBot_Application.py       # Main chatbot application
│   ├── chatBot_model_file.py        # Training script
│   ├── chatbot_Application_model.h5 # Trained model
│   ├── intents.json                 # Intents dataset
│   ├── words.pkl                    # Saved vocabulary
│   ├── labels.pkl                   # Saved intent labels
│
├── questions.txt                    # Example queries to test the bot
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started  

### 1. Clone the repository  
```bash
git clone https://github.com/<your-username>/chatbot-project.git
cd chatbot-project
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources  
Run once in Python:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

### 4. Train the model (if needed)  
```bash
python "chatBot_model_file.py"
```

### 5. Run the chatbot  
```bash
python "ChatBot_Application.py"
```

---

## 💡 Example Conversation  
```
You: hi
BOT: Hello! How can I assist you today?

You: what is normal blood pressure
BOT: Normal blood pressure is around 120/80 mmHg.

You: who is the president of India
BOT: Sorry, I don’t know the answer to that. I can help with drug reactions, blood pressure, hospitals, or pharmacies.
```

---

## 📌 Future Improvements  
- Add more intents (diet, exercise, mental health, etc.)  
- Use **word embeddings** (Word2Vec, GloVe) instead of Bag-of-Words  
- Upgrade to **LSTM/Transformer-based models** for better context understanding  
- Build a **GUI/Web app interface** for real-world usage  

---


