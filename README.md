# NLP_Disaster_Tweets


# This notebook goes through different EDA, data cleaning steps, and RNN models for the purpose of this Kaggle competition:
## https://www.kaggle.com/competitions/nlp-getting-started/overview


## Brief Description of the Problem and Data

The challenge involves building a model to predict which Tweets are about real disasters and which ones are not. This is a binary classification problem, where the output is 1 if the tweet is about a real disaster and 0 otherwise. For this assignment - week4 - the requirements are to use only RNN models (example LSTM) to do the text classification.

The data provided is a training set and a test set, both containing tweets with various features. The training data (`train.csv`) includes the following columns:

- `id`: a unique identifier for each tweet (this column was not used within any analysis here)
- `text`: the text of the tweet
- `location`: the location the tweet was sent from (may be blank)
- `keyword`: a particular keyword from the tweet (may be blank)
- `target`: denotes whether a tweet is about a real disaster (1) or not (0)

The training set size and structure are as follows:

- Number of entries: 7,613
- Number of features: 4 ( the id column cannot be used a feature here )

The test set (`test.csv`) has the same structure but without the `target` column. The target column is then predicted and submitted to Kaggle which is the competition part of this assignment.

## Exploratory Data Analysis (EDA)

During EDA, I found:

- The distribution of the `target` variable shows a fairly balanced dataset between disaster and non-disaster tweets.
- Common keywords and locations provided insights into frequent disaster-related terms and areas most discussed in the context of disasters.
- Text data preprocessing included removing URLs, punctuation, converting to lowercase, and removing stopwords to clean up the tweets for analysis.
- Visualizations like word clouds highlighted the most frequent words in the `keyword`, `location`, and `text` columns, confirming the relevance of certain words to disaster-related tweets.

My plan of analysis post-EDA involved preparing the text data for modeling by converting it into a numerical format that a machine learning model could understand, such as through tokenization and padding for RNN input.

## Model Architecture

For the model architecture, a Bidirectional Long Short-Term Memory (Bi-LSTM) network was chosen due to its ability to capture context from both the past and future words in a sentence, which is crucial for understanding the meaning of tweets. The architecture includes:

- An Embedding layer to convert tokenized words into dense vectors of fixed size.
- A Bi-LSTM layer to process the text data in both forward and backward directions.
- A Dense output layer with a sigmoid activation function to classify the tweets as disaster or not.

Word embeddings were utilized to improve model performance, with pre-trained GloVe vectors helping the model understand the semantic meaning of words better. In this case GloVe only slightly improved the performance which suggests something else needs to be improved here. Maybe the text cleaning or the words which are imbedding could improve performance. For further analysis you should try to embed the entire sentence using sentence to Vec or possibly GPT embeddings for greater context understanding. The assumption here is the greater the context understanding the better the prediction will be for the binary class.

Final model architecture:

## Final Model Architecture Summary
This is before the last grid search at the bottom of this notebook. The last grid search below does not significantly improve the accuracy.

- **Embedding Layer**: Utilizes pre-trained GloVe embeddings (`glove.6B.300d.txt`) with an embedding dimension of 300 and an input length of 100 tokens.
- **Bidirectional LSTM Layer**: Contains 50 units to capture both forward and backward dependencies in the text data.
- **Dense Output Layer**: A single unit with a sigmoid activation function for binary classification (disaster or not disaster tweets).
- **Compilation**: The model is compiled with the Adam optimizer and binary cross-entropy loss, measuring accuracy as the performance metric.
- **Training**: Trained on the dataset for 100 epochs, showing progressive improvement in accuracy and adjustments in loss over time. **However, this is a classic case of overfitting. The validation accuracy bounces around at 79 to 81% and goes back below 79 as the epochs go higher than 10.**
- **Performance**: Achieved a best validation accuracy of approximately 81%, demonstrating the model's capability to classify tweets into disaster and non-disaster categories effectively. As already discussed above this is only slightly better compared to the 79% accuracy achieved.


## Results and Analysis

Hyperparameter tuning and experimenting with different RNN architectures (LSTM, GRU, SimpleRNN, Bidirectional LSTM) were conducted to find the optimal setup. The best performing model was a Bidirectional LSTM with specific configurations (dropout, recurrent_dropout, optimizer, etc.).

The model achieved an accuracy of around 81% on the validation set, indicating a reasonably good performance in distinguishing between disaster and non-disaster tweets. The use of pre-trained GloVe embeddings slightly improved accuracy, suggesting the benefit of leveraging external knowledge.

## Conclusion

This week 4 mini Kaggle project showed that RNNs, specifically Bi-LSTMs with GloVe embeddings, are effective for text classification tasks like predicting disaster-related tweets. While the model performed well, further improvements could involve exploring more advanced NLP techniques, deeper architectures, or alternative word embeddings (example: BERT or GPT ADA, etc).

Future work could also include more sophisticated data preprocessing and augmentation strategies to improve model robustness and performance.

Note that in my experience using GPT word embeddings it is not necessary to remove punctuation and other stop words as this provides GPT a greater context of undertanding if left in. That is why I believe using this technique of GPT word embeddings could improve this binary classifier. GPT is able to make use of the punctuation and other parts of the text very well.
