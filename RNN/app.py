import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ======================
# 1. Prepare the Dataset
# ======================
data = {
'paragraph': [
"This movie was absolutely fantastic! I loved every minute of it. Highly recommend.",
"The service was terrible and the food was bland. A truly disappointing experience.",
"What a wonderful day! The sun is shining and I feel great.",
"I'm so frustrated with this product. It broke after just one use.",
"An amazing performance by the entire cast. Truly captivating.",
"This book is boring and hard to follow. I couldn't finish it.",
"Such a pleasant surprise! Everything exceeded my expectations.",
"The delivery was late and the item was damaged. Very unhappy.",
"A truly inspiring story that touched my heart.",
"I regret buying this. It's a complete waste of money.",
"Enjoyed the concert immensely. The music was beautiful.",
"The customer support was unhelpful and rude.",
"This new feature is brilliant and very useful.",
"I had a terrible time at the event. It was poorly organized.",
"Highly satisfied with the results. Excellent work!",
"The quality is poor and it feels cheap.",
"A delightful experience from start to finish.",
"This situation is very stressful and upsetting.",
"I'm so happy with my purchase. It's perfect!",
"Absolutely dreadful. Never again.",
"The atmosphere was vibrant and the staff were friendly.",
"This software is buggy and crashes frequently.",
"A truly memorable vacation. So relaxing.",
"The instructions were unclear and confusing.",
"Fantastic value for money. Highly recommended.",
"I'm utterly disgusted by their behavior.",
"What a brilliant idea! It will change everything.",
"This is the worst decision I've ever made.",
"Feeling very optimistic about the future.",
"The problem is persistent and unresolved."
],
'sentiment': [
'positive', 'negative', 'positive', 'negative', 'positive',
'negative', 'positive', 'negative', 'positive', 'negative',
'positive', 'negative', 'positive', 'negative', 'positive',
'negative', 'positive', 'negative', 'positive', 'negative',
'positive', 'negative', 'positive', 'negative', 'positive',
'negative', 'positive', 'negative', 'positive', 'negative'
]
}

df = pd.DataFrame(data)

# Convert labels to binary (0=negative, 1=positive)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# ======================
# 2. Tokenization
# ======================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['paragraph'])
sequences = tokenizer.texts_to_sequences(df['paragraph'])

max_len = 20
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = np.array(df['sentiment'])

# ======================
# 3. Train/Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 4. Build LSTM Model
# ======================
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ======================
# 5. Train
# ======================
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4)

# ======================
# 6. Evaluate the model
# ======================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# 7. Prediction Function

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad_seq)[0][0]
    return "Positive 😀" if pred > 0.5 else "Negative 😡"


# 8. Test with Random Input

print(predict_sentiment("I am very happy with the product, it works great!"))
print(predict_sentiment("This is the worst service I have ever experienced."))

# ======================
# 9. Interactive Prediction
# ======================
while True:
    user_input = input("Enter text for sentiment analysis (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")

# 