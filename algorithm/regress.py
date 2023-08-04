import array
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfTransform:
    def __init__(self):
        pass

    """
    Transform data to to mutiple features, count fec
    TfidfVectorizer
    
    """
    def transform(self, data):
        self.data_origin = data
        self.vectorizer = TfidfVectorizer( min_df=0.1, max_df=0.9, max_features = 200000, ngram_range = (1, 1))
        advancedtrain = self.vectorizer.fit_transform(data)
        self.data_transformed = advancedtrain.toarray()
        return self.data_transformed
    
    def inverse_transform(self, data):
        return self.vectorizer.inverse_transform(data)

import torch
from torch import embedding
from torch.nn.functional import pad
from underthesea import classify, word_tokenize, sent_tokenize
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from helper import isNumpy

class PhobertTransform:
    max_len = 300
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_hidden_state(self, outputs, type='view'):
        # 2 way for get hidden state: outputs[0], outputs['last_hidden_state']
        with torch.no_grad():
            if type == 'view':
                hiddenSize = outputs[0].size()[0]
                # flatten dimension 1 and 2, and hold batch dimension 0
                embeddings = outputs[0].view(hiddenSize, -1).numpy()
            elif type == 'mean':
                hiddenSize = outputs[0].size()[0]
                embeddings = outputs[0].mean(axis=1).numpy()

        return embeddings

    def vectorize_mean(self, sentences, type_extract='mean'):
        embeddingArr = []
        for sentence in sentences:
            try:
                encodedText = self.tokenizer.encode_plus(
                    sentence, padding=True, truncation=True, 
                    add_special_tokens=True,
                    return_attention_mask=True,
                    max_length=self.max_len, return_tensors='pt')
            except Exception as e:
                print('e', e, sentence)
                return None
            
            encodedTextInputIds = encodedText['input_ids']
            embeddedTitle = self.model(torch.tensor(encodedTextInputIds))
            embedding = self.extract_hidden_state(embeddedTitle, type_extract)
            embeddingArr.append(embedding.ravel())
        return embeddingArr

    def fit(self, x_train, y_train):
        self.x_train_raw = np.array(x_train) if isNumpy(x_train) else x_train
        self.y_train_raw = np.array(y_train) if isNumpy(y_train) else y_train
        print('fit', self.x_train_raw.shape, self.y_train_raw.shape)
        self.x_train = np.array(self.vectorize_mean(self.x_train_raw))
        self.y_train = self.y_train_raw.ravel()
        print('fit_1:', self.y_train)
        self.regressor = SVR(kernel='linear')
        self.regressor.fit(self.x_train, self.y_train)  # ravel y_train to convert to 1D array

    def predict(self, x_test):
        x_test = self.vectorize_mean(x_test)
        result = self.regressor.predict(x_test)
        return result
    
    @staticmethod
    def trim_words(sentence, num_word=200):
        # Split the sentence into words
        words = sentence.split()
        # Trim the sentence to 5 words
        trimmed_sentence = " ".join(words[:num_word])
        return trimmed_sentence


    """
    Transform data to to mutiple features, count fec
    TfidfVectorizer
    
    """
    def transform(self, data):
        self.data_origin = data
        self.data_transformed = self.vectorize_mean(data)
        return self.data_transformed

class DataTransform:
    def __init__(self, transformer):
        self.transformer = transformer

    def transform(self, data):
        self.data_transformed = self.transformer.transform(data)
        return self.data_transformed

    
class RegressionStock:
    def __init__(self):
        pass

    def use_regression(self, model_type='linear'):
        if model_type == 'linear':
            self.regressor = LinearRegression()
        elif model_type == 'logistic':
            self.regressor = LogisticRegression()
        elif model_type == 'randomforest':
            self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'nbsvm':
            self.regressor = NBSVM(C=0.01)

    def fit(self, x_train, y_train):
        if isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray):
            self.x_train = x_train
            self.y_train = y_train
        else:
            self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        self.train()

    def train(self):
        self.regressor.fit(self.x_train, self.y_train)
    
    def tranform_to_predict(self, data):
        return data

    def predict(self, data):
        result = self.regressor.predict(data)
        return result