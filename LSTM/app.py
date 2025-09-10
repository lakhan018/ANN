# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Keras Tuner for hyperparameter optimization

import keras_tuner as kt

print("TensorFlow Version:", tf.__version__)

# ==============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================
# [cite_start]Corpus data provided in the project description [cite: 3-115]
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

print("Corpus Head:")
print(df.head())
print("\nClass Distribution:")
print(df['sentiment_label'].value_counts(normalize=True) * 100)

# ==============================================================================
# 3. DATA PREPROCESSING AND SPLITTING
# ==============================================================================
# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment_label'])
# Positive: 2, Neutral: 1, Negative: 0
print("\nLabel Mapping:", {i: label for i, label in enumerate(label_encoder.classes_)})


# Split data (70% train, 15% validation, 15% test)
X = df['cleaned_text']
y = df['sentiment_encoded']

# First split: 70% train, 30% temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Second split: 15% val, 15% test from the temp set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# ==============================================================================
# 4. TOKENIZATION AND PADDING
# ==============================================================================
# Hyperparameters for tokenization and padding
vocab_size = 5000
oov_tok = "<OOV>"
max_length = 20
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# One-hot encode the labels for the categorical crossentropy loss function
y_train_cat = to_categorical(y_train, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# ==============================================================================
# 5. WORD EMBEDDINGS (Using GloVe)
# ==============================================================================
# NOTE: Download GloVe embeddings from: https://nlp.stanford.edu/data/glove.6B.zip
# Unzip and place 'glove.6B.100d.txt' in your working directory.
embedding_dim = 100
embeddings_index = {}
try:
    with open('glove.6B.100d.txt', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
except FileNotFoundError:
    print("\nGloVe file not found. Skipping embedding layer initialization.")
    print("Download from: https://nlp.stanford.edu/data/glove.6B.zip")
    embeddings_index = None

if embeddings_index:
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("\nGloVe embedding matrix created.")

# ==============================================================================
# 6. HYPERPARAMETER OPTIMIZATION with Keras Tuner
# ==============================================================================
def build_model(hp, rnn_type='lstm'):
    """Model builder for Keras Tuner."""
    model = Sequential()
    
    # Embedding Layer
    if embeddings_index:
      # Use pre-trained GloVe embeddings
      model.add(Embedding(len(word_index) + 1,
                          embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_length,
                          trainable=False))
    else:
      # Train embeddings from scratch if GloVe is not available
      model.add(Embedding(len(word_index) + 1,
                          embedding_dim,
                          input_length=max_length))

    # RNN Layer (LSTM or GRU)
    rnn_units = hp.Choice('rnn_units', values=[32, 64, 128])
    if rnn_type.lower() == 'lstm':
        model.add(LSTM(units=rnn_units))
    elif rnn_type.lower() == 'gru':
        model.add(GRU(units=rnn_units))

    # Dropout Layer
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate))

    # Dense Layer
    dense_units = hp.Choice('dense_units', values=[16, 32, 64])
    model.add(Dense(units=dense_units, activation='relu'))
    
    # Output Layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# --- LSTM Tuner ---
lstm_tuner = kt.RandomSearch(
    lambda hp: build_model(hp, rnn_type='lstm'),
    objective='val_accuracy',
    max_trials=5,  # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory='tuner_results',
    project_name='lstm_sentiment'
)

# --- GRU Tuner ---
gru_tuner = kt.RandomSearch(
    lambda hp: build_model(hp, rnn_type='gru'),
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='gru_sentiment'
)

# Start the search (using a small number of epochs for speed)
print("\n--- Starting Hyperparameter Search for LSTM ---")
lstm_tuner.search(X_train_pad, y_train_cat, epochs=10, validation_data=(X_val_pad, y_val_cat))
print("\n--- Starting Hyperparameter Search for GRU ---")
gru_tuner.search(X_train_pad, y_train_cat, epochs=10, validation_data=(X_val_pad, y_val_cat))

# Get the best hyperparameters
best_lstm_hp = lstm_tuner.get_best_hyperparameters(num_trials=1)[0]
best_gru_hp = gru_tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n--- Best Hyperparameters Found ---")
print(f"LSTM: Units={best_lstm_hp.get('rnn_units')}, Dropout={best_lstm_hp.get('dropout_rate'):.2f}, LR={best_lstm_hp.get('learning_rate')}, Optimizer={best_lstm_hp.get('optimizer')}")
print(f"GRU: Units={best_gru_hp.get('rnn_units')}, Dropout={best_gru_hp.get('dropout_rate'):.2f}, LR={best_gru_hp.get('learning_rate')}, Optimizer={best_gru_hp.get('optimizer')}")

# ==============================================================================
# 7. FINAL MODEL TRAINING AND EVALUATION
# ==============================================================================
# Build the best models with the optimal hyperparameters
best_lstm_model = lstm_tuner.hypermodel.build(best_lstm_hp)
best_gru_model = gru_tuner.hypermodel.build(best_gru_hp)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Train Best LSTM Model ---
print("\n--- Training Best LSTM Model ---")
history_lstm = best_lstm_model.fit(
    X_train_pad, y_train_cat,
    epochs=50,
    validation_data=(X_val_pad, y_val_cat),
    callbacks=[early_stopping],
    verbose=2
)

# --- Train Best GRU Model ---
print("\n--- Training Best GRU Model ---")
history_gru = best_gru_model.fit(
    X_train_pad, y_train_cat,
    epochs=50,
    validation_data=(X_val_pad, y_val_cat),
    callbacks=[early_stopping],
    verbose=2
)

# Helper function to plot training history
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

plot_history(history_lstm, "LSTM Model Training")
plot_history(history_gru, "GRU Model Training")


# --- Evaluate on Test Set ---
def evaluate_model(model, model_name):
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    # Get predictions
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

evaluate_model(best_lstm_model, "Optimized LSTM Model")
evaluate_model(best_gru_model, "Optimized GRU Model")