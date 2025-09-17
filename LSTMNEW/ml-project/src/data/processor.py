import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import numpy as np

def load_and_create_dataset(file_path="sentiment_corpus_100.csv"):
    """Creates the initial CSV from the provided corpus list."""

    
    corpus_data = [
    {"text": "This movie was absolutely fantastic!", "sentiment_label": "Positive"},
    {"text": "The service was incredibly slow and rude.", "sentiment_label": "Negative"},
    {"text": "The product arrived on time.", "sentiment_label": "Neutral"},
    {"text": "I'm so happy with my new phone.", "sentiment_label": "Positive"},
    {"text": "What a terrible experience, never again.", "sentiment_label": "Negative"},
    {"text": "It's raining outside today.", "sentiment_label": "Neutral"},
    {"text": "Highly recommend this restaurant.", "sentiment_label": "Positive"},
    {"text": "Feeling quite disappointed with the outcome.", "sentiment_label": "Negative"},
    {"text": "The meeting concluded at 3 PM.", "sentiment_label": "Neutral"},
    {"text": "An amazing performance by the cast.", "sentiment_label": "Positive"},
    {"text": "The food was bland and unappetizing.", "sentiment_label": "Negative"},
    {"text": "The package was delivered yesterday.", "sentiment_label": "Neutral"},
    {"text": "Absolutely loved every moment of it!", "sentiment_label": "Positive"},
    {"text": "Worst customer support ever encountered.", "sentiment_label": "Negative"},
    {"text": "The report is due by Friday.", "sentiment_label": "Neutral"},
    {"text": "Such a delightful surprise!", "sentiment_label": "Positive"},
    {"text": "Completely dissatisfied with the purchase.", "sentiment_label": "Negative"},
    {"text": "The car is parked in the garage.", "sentiment_label": "Neutral"},
    {"text": "A truly inspiring story.", "sentiment_label": "Positive"},
    {"text": "This software is full of bugs.", "sentiment_label": "Negative"},
    {"text": "The temperature is 25 degrees Celsius.", "sentiment_label": "Neutral"},
    {"text": "Couldn't be happier with the results.", "sentiment_label": "Positive"},
    {"text": "My expectations were not met at all.", "sentiment_label": "Negative"},
    {"text": "She wore a blue dress.", "sentiment_label": "Neutral"},
    {"text": "Fantastic value for money.", "sentiment_label": "Positive"},
    {"text": "This is a complete waste of time.", "sentiment_label": "Negative"},
    {"text": "The book has 300 pages.", "sentiment_label": "Neutral"},
    {"text": "So glad I bought this!", "sentiment_label": "Positive"},
    {"text": "I regret buying this product.", "sentiment_label": "Negative"},
    {"text": "He is walking to the store.", "sentiment_label": "Neutral"},
    {"text": "A wonderful experience from start to finish.", "sentiment_label": "Positive"},
    {"text": "The quality is very poor.", "sentiment_label": "Negative"},
    {"text": "The cat is sleeping on the couch.", "sentiment_label": "Neutral"},
    {"text": "Highly recommend this place.", "sentiment_label": "Positive"},
    {"text": "I'm utterly disgusted by this.", "sentiment_label": "Negative"},
    {"text": "The train leaves at 7 AM.", "sentiment_label": "Neutral"},
    {"text": "This made my day!", "sentiment_label": "Positive"},
    {"text": "Never going back there again.", "sentiment_label": "Negative"},
    {"text": "The sun rises in the east.", "sentiment_label": "Neutral"},
    {"text": "Simply brilliant and captivating.", "sentiment_label": "Positive"},
    {"text": "The connection keeps dropping.", "sentiment_label": "Negative"},
    {"text": "Water boils at 100 degrees.", "sentiment_label": "Neutral"},
    {"text": "Exceeded all my expectations.", "sentiment_label": "Positive"},
    {"text": "This is incredibly frustrating.", "sentiment_label": "Negative"},
    {"text": "The capital of France is Paris.", "sentiment_label": "Neutral"},
    {"text": "What a pleasant surprise!", "sentiment_label": "Positive"},
    {"text": "I'm so angry about this situation.", "sentiment_label": "Negative"},
    {"text": "The sky is blue.", "sentiment_label": "Neutral"},
    {"text": "Truly a masterpiece.", "sentiment_label": "Positive"},
    {"text": "This is just unacceptable.", "sentiment_label": "Negative"},
    {"text": "The dog barked loudly.", "sentiment_label": "Neutral"},
    {"text": "Feeling very optimistic about the future.", "sentiment_label": "Positive"},
    {"text": "This news is quite upsetting.", "sentiment_label": "Negative"},
    {"text": "The computer is turned off.", "sentiment_label": "Neutral"},
    {"text": "A truly enjoyable read.", "sentiment_label": "Positive"},
    {"text": "The delay was very annoying.", "sentiment_label": "Negative"},
    {"text": "The light is green.", "sentiment_label": "Neutral"},
    {"text": "So impressed with the quality.", "sentiment_label": "Positive"},
    {"text": "I'm fed up with this problem.", "sentiment_label": "Negative"},
    {"text": "The door is open.", "sentiment_label": "Neutral"},
    {"text": "This made me smile.", "sentiment_label": "Positive"},
    {"text": "Such a disappointing outcome.", "sentiment_label": "Negative"},
    {"text": "The phone rang.", "sentiment_label": "Neutral"},
    {"text": "Absolutely perfect in every way.", "sentiment_label": "Positive"},
    {"text": "This is a terrible idea.", "sentiment_label": "Negative"},
    {"text": "He is reading a book.", "sentiment_label": "Neutral"},
    {"text": "Highly satisfied with the purchase.", "sentiment_label": "Positive"},
    {"text": "The situation is quite grim.", "sentiment_label": "Negative"},
    {"text": "The clock shows 10:30.", "sentiment_label": "Neutral"},
    {"text": "A fantastic addition to my collection.", "sentiment_label": "Positive"},
    {"text": "I'm really upset about this.", "sentiment_label": "Negative"},
    {"text": "The bird is singing.", "sentiment_label": "Neutral"},
    {"text": "This brought me so much joy.", "sentiment_label": "Positive"},
    {"text": "The performance was dreadful.", "sentiment_label": "Negative"},
    {"text": "She is wearing glasses.", "sentiment_label": "Neutral"},
    {"text": "What a delightful experience!", "sentiment_label": "Positive"},
    {"text": "This is a complete rip-off.", "sentiment_label": "Negative"},
    {"text": "The table is made of wood.", "sentiment_label": "Neutral"},
    {"text": "Feeling incredibly grateful.", "sentiment_label": "Positive"},
    {"text": "I'm so frustrated right now.", "sentiment_label": "Negative"},
    {"text": "The window is closed.", "sentiment_label": "Neutral"},
    {"text": "A truly heartwarming story.", "sentiment_label": "Positive"},
    {"text": "This is a major letdown.", "sentiment_label": "Negative"},
    {"text": "The grass is green.", "sentiment_label": "Neutral"},
    {"text": "So happy with the results.", "sentiment_label": "Positive"},
    {"text": "The situation is quite bad.", "sentiment_label": "Negative"},
    {"text": "He opened the box.", "sentiment_label": "Neutral"},
    {"text": "This is a game-changer!", "sentiment_label": "Positive"},
    {"text": "I'm very disappointed.", "sentiment_label": "Negative"},
    {"text": "The coffee is hot.", "sentiment_label": "Neutral"},
    {"text": "Absolutely loving it!", "sentiment_label": "Positive"},
    {"text": "This is a nightmare.", "sentiment_label": "Negative"},
    {"text": "She wrote a letter.", "sentiment_label": "Neutral"},
    {"text": "Couldn't ask for more.", "sentiment_label": "Positive"},
    {"text": "The outcome was terrible.", "sentiment_label": "Negative"},
    {"text": "The car is red.", "sentiment_label": "Neutral"},
    {"text": "A truly remarkable achievement.", "sentiment_label": "Positive"},
    {"text": "This is a huge mistake.", "sentiment_label": "Negative"},
    {"text": "The boy is playing.", "sentiment_label": "Neutral"},
    {"text": "Feeling very positive about this.", "sentiment_label": "Positive"},
    {"text": "I'm quite upset by this.", "sentiment_label": "Negative"},
    {"text": "The movie starts at 8 PM.", "sentiment_label": "Neutral"},
    {"text": "What a wonderful day!", "sentiment_label": "Positive"},
    {"text": "This is just awful.", "sentiment_label": "Negative"},
    {"text": "The phone is ringing.", "sentiment_label": "Neutral"},
    {"text": "So glad I found this.", "sentiment_label": "Positive"},
    {"text": "I'm really annoyed.", "sentiment_label": "Negative"},
    {"text": "The water is cold.", "sentiment_label": "Neutral"},
    {"text": "This is truly amazing.", "sentiment_label": "Positive"},
    {"text": "The service was pathetic.", "sentiment_label": "Negative"},
    {"text": "He closed the door.", "sentiment_label": "Neutral"},
    ]
    df = pd.DataFrame(corpus_data)
    df.to_csv(file_path, index=False)
    print(f"Dataset created and saved to {file_path}")
    return df

def clean_text(text):
    """Applies text cleaning steps: lowercasing, removing punctuation and special chars."""
    text = text.lower() # Lowercasing
    text = re.sub(r'[.,!?"\']', '', text) # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    return text

def preprocess_data(df, max_len=20):
    """Main function to preprocess the entire dataframe."""
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment_label'])
    
    # Split data
    X = df['cleaned_text']
    y = df['sentiment_encoded']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Tokenize text
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    return (X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, tokenizer, label_encoder)

def train_word2vec(sentences, embedding_dim=100):
    """Trains a Word2Vec model on the corpus."""
    # Word2Vec expects a list of lists of tokens
    tokenized_sentences = [s.split() for s in sentences]
    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    return w2v_model

def create_embedding_matrix(w2v_model, tokenizer, embedding_dim=100):
    """Creates an embedding matrix for the Keras Embedding layer."""
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix