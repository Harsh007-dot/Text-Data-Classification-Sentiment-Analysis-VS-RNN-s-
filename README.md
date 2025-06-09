# Text Classification: Sentiment Analysis vs RNN

This project compares multiple text classification techniques using the NLTK movie_reviews dataset.
NLTK's movie_reviews dataset:

2000 movie reviews labeled as pos or neg.

Used for training and evaluating all models.

1. Naive Bayes (Traditional Sentiment Analysis)
File Reference: Untitled - Colab.PDF
Steps:
Preprocessing: Tokenize each word from a review and convert to a dictionary where each token is a feature (word: True).

Feature Extraction: Done via extract_features(review) function.

Data Split: 80% training, 20% testing using train_test_split.

Model: nltk.NaiveBayesClassifier.train()

Evaluation: Accuracy calculated using sklearn.metrics.accuracy_score.

⚙️ Sample Accuracy: ~69.75%

2. RNN (Recurrent Neural Network)
File Reference: Untitled - Colab.PDF
Steps:
Tokenization: Use Tokenizer from Keras to convert text to sequences.

Padding: Pad sequences to uniform length using pad_sequences.

Model Architecture:

Embedding layer

SimpleRNN layer

Dense layer with sigmoid activation

Training: 5 epochs with validation_split=0.2

Evaluation: Accuracy on the test set using accuracy_score.

⚙️ Sample Accuracy: ~60.00%

3. LSTM (Long Short-Term Memory)
File Reference: Untitled - Colab.PDF

Steps:
Preprocessing:

Tokenization using word_tokenize

Lemmatization using WordNetLemmatizer

Removal of stop words

Tokenization: Again use Keras Tokenizer

Padding: pad_sequences with max_len = 100

Model:

Embedding layer

LSTM with 64 units

Dense output layer

Training: 10 epochs

Evaluation: Loss and Accuracy

⚙Sample Accuracy: ~63.75%

4.CNN (Convolutional Neural Network)
Steps:
Text Vectorization: Use Keras Tokenizer and pad_sequences

Model Architecture:

Embedding layer

Conv1D layer (filters=250, kernel_size=3)

GlobalMaxPooling1D

Dense output layer

Training: 10 epochs, batch size = 64

Evaluation: Accuracy using model.evaluate()

⚙️ Sample Accuracy: ~69.25%
