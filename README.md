# Sentiment and Emotion Analysis in Tweets with Recurrent Neural Networks

## Project Overview

This project explores the field of Natural Language Processing (NLP) to classify sentiment (positive, negative, neutral) and emotion in tweets related to the Dell brand. Four distinct Recurrent Neural Network (RNN) architectures were implemented and compared to evaluate their effectiveness on this task: **LSTM**, **GRU**, **SimpleRNN**, and **Bidirectional LSTM**.

The objective is to demonstrate a complete NLP pipeline, from data cleaning and preprocessing to the training and evaluation of deep learning models.

---

## Dataset

The dataset used is `sentiment-emotion-labelled_Dell_tweets.csv`, which contains a collection of tweets about Dell, each labeled with a specific sentiment and emotion.

---

## Methodology

The project's workflow was structured into the following steps:

1.  **Data Loading and Cleaning:**
    * The data was loaded from the CSV file.
    * A preprocessing function was applied to clean the text of each tweet, which included:
        * Conversion to lowercase.
        * Removal of URLs, mentions (@), hashtags (#), and non-alphabetic characters.
        * Removal of English stopwords (common words like "the," "a," "of").

2.  **Dataset Balancing:**
    * Initial analysis showed that the sentiment classes were imbalanced (e.g., significantly more negative tweets than neutral ones).
    * To prevent the model from being biased towards the majority class, a balanced training set was created by randomly sampling **7,000 examples** from each of the three sentiment classes (positive, negative, and neutral).

3.  **Data Splitting:**
    * The balanced set of 21,000 tweets was split into **80% for training** (16,800 tweets) and **20% for validation** (4,200 tweets).
    * The final test set was formed by combining the validation set with the remaining data from the original dataset that was not used for training, ensuring a robust evaluation on unseen, imbalanced data.

4.  **Tokenization and Padding:**
    * The text was converted into numerical sequences using a `Tokenizer`, considering a maximum vocabulary of the **1,000 most frequent words**.
    * The sequences were standardized to a maximum length of **144 tokens** (the character limit of a tweet) through padding, so that all inputs had the same size.

5.  **Label Encoding:**
    * The sentiment labels (negative, neutral, positive) were converted to a numerical format and subsequently into a *one-hot encoding* format, which is suitable for the `categorical_crossentropy` loss function.

---

## Model Architectures

Four RNN models were trained and evaluated, all sharing a similar base architecture, with the recurrent layer being the only variation:

1.  **LSTM (Long Short-Term Memory):** Effective at learning long-term dependencies.
2.  **GRU (Gated Recurrent Unit):** A simpler and more computationally efficient version of LSTM.
3.  **SimpleRNN:** The most basic RNN architecture.
4.  **Bidirectional LSTM:** Processes the text sequence in both directions (left-to-right and right-to-left), capturing a richer context.

All models used an initial `Embedding` layer and a final `Dense` layer with a `softmax` activation for classification. Regularization techniques such as `Dropout` were also applied to prevent *overfitting*.

---

## Results

After training for 15 epochs, the models yielded the following accuracies on the test set:

| Model | Test Accuracy |
| :--- | :---: |
| **Bidirectional LSTM** | **73.24%** |
| LSTM | 72.02% |
| GRU | 72.00% |
| SimpleRNN | 67.18% |

The **Bidirectional LSTM** achieved the best performance, which suggests that analyzing the context in both directions of the text was beneficial for this task.

---

## How to Run the Project

1.  **Prerequisites:** Ensure you have Python 3 and the following libraries installed:
    ```bash
    pandas
    numpy
    matplotlib
    seaborn
    nltk
    tensorflow
    scikit-learn
    ```
    You can install them with the command:
    `pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn`

2.  **Data Structure:**
    * Place the notebook file (`ProjetoFinal_ET287_v2.ipynb`) and the dataset (`sentiment-emotion-labelled_Dell_tweets.csv`) in the same folder.

3.  **Execution:**
    * Open the notebook in an environment such as Jupyter Notebook or Google Colab.
    * Execute the cells in order to reproduce the analysis and results.

---

## Technologies Used

* **Language:** Python 3
* **Environment:** Jupyter Notebook / Google Colab
* **Main Libraries:**
    * TensorFlow (with Keras) for building the deep learning models.
    * Scikit-learn for preprocessing and evaluation metrics.
    * Pandas and NumPy for data manipulation.
    * NLTK for text processing (stopwords removal).
    * Matplotlib and Seaborn for visualizing the results.
