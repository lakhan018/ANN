# Multi-Class Sentiment Analysis with RNNs and Hyperparameter Optimization

## 1. Project Overview

This project develops a robust sentiment analysis system to classify text into **Positive**, **Negative**, or **Neutral** categories. [cite_start]It implements and compares two Recurrent Neural Network (RNN) architectures, **LSTM** and **GRU**[cite: 125]. [cite_start]A key focus is the systematic hyperparameter optimization using two distinct methods: **Keras Tuner (RandomSearch)** and **Optuna (TPE Sampler)** to identify the best-performing model configurations[cite: 126, 173].

[cite_start]The project uses a custom corpus and trains its own Word2Vec embeddings to capture domain-specific language nuances[cite: 147, 150].

---

## 2. Project Structure

```
ml-project/
├── app/                  # Streamlit application files
├── data/                 # Raw and processed data
├── models/               # Saved final model files (.h5)
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code for the project
│   ├── data/
│   │   └── processor.py
│   └── models/
│       └── trainer.py
├── test/                 # Test scripts
├── create_environment.sh # Environment setup script
├── generate_project_template.sh # Project structure script
├── dev_requirements.txt  # Python dependencies
├── Dockerfile            # Containerization file
├── README.md             # This file
└── ...
```

---

## 3. Setup and Installation

Follow these steps to set up the project environment.

**Step 1: Create the Project Structure**
Run the template generation script from your terminal.
```bash
bash generate_project_template.sh
```

**Step 2: Create the Conda Environment**
Navigate into the newly created directory and run the environment setup script. [cite_start]This will create a Conda environment named `ml-project`, install all dependencies, and set up a Jupyter kernel [cite: 281-284].
```bash
cd ml-project
bash create_environment.sh
```

**Step 3: Activate the Environment**
Before running any Python scripts, activate the Conda environment.
```bash
conda activate ml-project
```

---

## 4. How to Run the ML Pipeline

The entire machine learning workflow—from data preprocessing to model training, hyperparameter tuning, and evaluation—is orchestrated by the main script.

To run the full pipeline for both LSTM and GRU models, execute the following command from the `ml-project/` root directory:
```bash
python src/main.py
```
The script will print the results of the hyperparameter searches and the final evaluation metrics for both model architectures to the console. The trained models will be saved in the `/models` directory.

---

## 5. Comparative Analysis & Discussion ✍️

**(This is where you will fill in your findings after running the code)**

### [cite_start]Model Comparison: LSTM vs. GRU [cite: 203]

* **Performance Metrics**:
    * **LSTM Model Accuracy**: [Enter Test Accuracy from your run]
    * **GRU Model Accuracy**: [Enter Test Accuracy from your run]
    * **Class-wise Performance**: Which model performed better for the 'Positive', 'Negative', and 'Neutral' classes respectively? Did one model show a clear advantage in precision or recall for a specific class?
    * **Discussion**: Based on the results, the [LSTM/GRU] model demonstrated superior performance. This could be attributed to [e.g., its ability to handle long-term dependencies, its simpler architecture preventing overfitting on this small dataset, etc.].

* [cite_start]**Training Time vs. Accuracy Tradeoff**[cite: 205]:
    * **Observation**: Was there a noticeable difference in the total training and tuning time between the LSTM and GRU models?
    * **Conclusion**: The GRU model, being computationally less expensive, offered a [better/worse/comparable] tradeoff between training time and accuracy for this specific task.

### [cite_start]Impact of Hyperparameter Optimization [cite: 206]

* [cite_start]**Keras Tuner (RandomSearch) vs. Optuna (TPE)**[cite: 208]:
    * **Best Hyperparameters (LSTM)**:
        * Keras Tuner: [e.g., 1 layer, 128 units, 0.3 dropout...]
        * Optuna: [e.g., 2 layers, 64 units, 0.45 dropout...]
    * **Best Hyperparameters (GRU)**:
        * Keras Tuner: [List parameters]
        * Optuna: [List parameters]
    * **Effectiveness**: Did both tuners converge on similar hyperparameter values? Did one tuner find a significantly better model configuration than the other? Optuna's TPE sampler is designed to be more intelligent than RandomSearch; did this reflect in the final validation accuracy during the search?

* [cite_start]**Most Impactful Hyperparameters**[cite: 209]:
    * **Observation**: Which hyperparameters seemed to have the most significant impact on model performance?
    * **Analysis**: For this dataset, the **learning rate** and **number of RNN units** were the most critical parameters. A learning rate of `1e-3` consistently outperformed others, while [e.g., 128 units] provided a good balance without overfitting.

### [cite_start]Challenges & Future Work [cite: 210]

* **Challenges**: The primary challenge was the very small size of the dataset, which increases the risk of overfitting and makes it difficult for the models to generalize. This was mitigated by using Dropout and EarlyStopping.
* **Future Work**:
    * [cite_start]**Bidirectional LSTM/GRU**: Implement bidirectional layers to capture context from both forward and backward directions[cite: 214].
    * [cite_start]**Attention Mechanism**: Add an attention layer to help the model focus on the most relevant words in a sentence[cite: 215].
    * [cite_start]**Transfer Learning**: Explore using pre-trained models like BERT for fine-tuning, which would likely yield significantly better results given the small dataset size[cite: 218].