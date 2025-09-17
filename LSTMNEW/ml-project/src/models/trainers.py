import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import keras_tuner as kt
import optuna
import numpy as np

# --- Model Building Functions ---

def build_rnn_model(hp, model_type='lstm', vocab_size=None, embedding_dim=None, max_len=None, embedding_matrix=None):
    """Builds either an LSTM or GRU model for Keras Tuner."""
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=hp.Boolean('trainable_embedding') # Fine-tune or freeze
    ))

    # RNN Layers
    num_layers = hp.Int('num_layers', 1, 2)
    for i in range(num_layers):
        rnn_units = hp.Int(f'rnn_units_{i}', min_value=32, max_value=256, step=32)
        return_sequences = (i < num_layers - 1) # Return sequences for all but the last layer
        if model_type == 'lstm':
            model.add(LSTM(rnn_units, return_sequences=return_sequences))
        else:
            model.add(GRU(rnn_units, return_sequences=return_sequences))
    
    # Dense Layers
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5)))
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5)))

    # Output Layer
    model.add(Dense(3, activation='softmax'))

    # Compile model
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Hyperparameter Tuning ---

def tune_with_keras_tuner(model_builder, X_train, y_train, X_val, y_val):
    """Runs hyperparameter search using Keras Tuner."""
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10, # Adjust as needed
        executions_per_trial=1,
        directory='keras_tuner_dir',
        project_name='sentiment_analysis'
    )
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Best Hyperparameters found by Keras Tuner:
    - Trainable Embedding: {best_hps.get('trainable_embedding')}
    - Num Layers: {best_hps.get('num_layers')}
    - Dense Units: {best_hps.get('dense_units')}
    - Dropout 1: {best_hps.get('dropout_1')}
    - Learning Rate: {best_hps.get('learning_rate')}
    - Optimizer: {best_hps.get('optimizer')}
    """)
    return tuner.get_best_models(num_models=1)[0]


def create_optuna_objective(model_type, X_train, y_train, X_val, y_val, vocab_size, embedding_dim, max_len, embedding_matrix):
    """Creates the objective function for Optuna."""
    def objective(trial):
        model = Sequential()
        model.add(Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
            input_length=max_len, trainable=trial.suggest_categorical('trainable_embedding', [True, False])
        ))

        num_layers = trial.suggest_int('num_layers', 1, 2)
        for i in range(num_layers):
            rnn_units = trial.suggest_categorical(f'rnn_units_{i}', [64, 128, 256])
            return_sequences = (i < num_layers - 1)
            if model_type == 'lstm':
                model.add(LSTM(rnn_units, return_sequences=return_sequences))
            else:
                model.add(GRU(rnn_units, return_sequences=return_sequences))
        
        model.add(Dropout(trial.suggest_float('dropout_1', 0.2, 0.5)))
        dense_units = trial.suggest_categorical('dense_units', [32, 64, 128])
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(trial.suggest_float('dropout_2', 0.2, 0.5)))
        model.add(Dense(3, activation='softmax'))

        learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
        optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_train, y_train, epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)],
            verbose=0
        )
        
        return np.max(history.history['val_accuracy'])
    return objective

# --- Final Training and Evaluation ---

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    """Trains the final model and evaluates it on the test set."""
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    print("\n--- Training Final Model ---")
    model.fit(
        X_train, y_train,
        epochs=50, # Higher epochs with early stopping
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        batch_size=64
    )
    
    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model