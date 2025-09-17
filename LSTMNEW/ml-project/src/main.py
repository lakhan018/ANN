from data.processor import load_and_create_dataset, preprocess_data, train_word2vec, create_embedding_matrix
from models.trainer import build_rnn_model, tune_with_keras_tuner, tune_with_optuna, train_and_evaluate_model
import functools

def run_pipeline_for_model(model_type):
    """
    Runs the full ML pipeline for a given model type ('lstm' or 'gru').
    """
    print(f"\n{'='*20} STARTING PIPELINE FOR {model_type.upper()} MODEL {'='*20}")

    # --- 1. Data Processing ---
    df = load_and_create_dataset()
    MAX_LEN = 20
    EMBEDDING_DIM = 100
    (X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, tokenizer, label_encoder) = preprocess_data(df, max_len=MAX_LEN)
    vocab_size = len(tokenizer.word_index) + 1

    # --- 2. Word Embeddings ---
    w2v_model = train_word2vec(df['cleaned_text'], embedding_dim=EMBEDDING_DIM)
    embedding_matrix = create_embedding_matrix(w2v_model, tokenizer, embedding_dim=EMBEDDING_DIM)

    # --- 3. Hyperparameter Tuning ---
    # Method 1: Keras Tuner
    model_builder = functools.partial(
        build_rnn_model, model_type=model_type, vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN, embedding_matrix=embedding_matrix
    )
    best_model_from_kt = tune_with_keras_tuner(
        model_builder, f"{model_type}_kt_tuning", X_train_pad, y_train, X_val_pad, y_val
    )
    
    # Method 2: Optuna
    best_model_from_optuna = tune_with_optuna(
        model_type, X_train_pad, y_train, X_val_pad, y_val, vocab_size,
        EMBEDDING_DIM, MAX_LEN, embedding_matrix
    )
    
    # --- 4. Final Training and Evaluation ---
    # You can choose which tuner's result to proceed with. Here, we'll use Optuna's.
    print(f"\n--- Finalizing model based on OPTUNA's best parameters for {model_type.upper()} ---")
    train_and_evaluate_model(
        best_model_from_optuna, f"{model_type}_optuna_best", X_train_pad, y_train, 
        X_val_pad, y_val, X_test_pad, y_test, label_encoder
    )
    
    print(f"\n{'='*20} PIPELINE FOR {model_type.upper()} MODEL COMPLETE {'='*20}")

if __name__ == '__main__':
    # Run the entire process for the LSTM model
    run_pipeline_for_model('lstm')
    
    # Run the entire process for the GRU model
    run_pipeline_for_model('gru')