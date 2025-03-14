import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.util import ngrams

stop_words = set(stopwords.words('english'))
stop_words.add("said")
stemmer = LancasterStemmer()

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.conditionals = None
        self.V = None
        self.smoothening = 0

    def plot_word_clouds(self, df, class_col="Class Index", text_col="Tokenized Description"):
        classes = [1 , 2 , 3 , 4]        
        for label in classes:
            words = ' '.join([' '.join(text) for text in df[df[class_col] == label][text_col]])
            wordcloud = WordCloud(width=800, height=400, background_color='white' , stopwords=stop_words).generate(words)
            
            plt.figure(figsize=(8, 4))
            plt.title(f"Word Cloud for Class {label}")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()


    def bigram_f(self , text):
        words = []
        for i in range(len(text) - 1):
            words.append(text[i] + " " + text[i+1])
        return words    

    def activation(self, label, text):
        log_act = 0.0
        for word in text:
            prob = self.conditionals[label].get(word, (1.0 * self.smoothening) / (self.smoothening * self.V))
            log_act += np.log(prob)
        return log_act
    
    def activation_bigram(self, label, text):
        log_act = 0.0
        for word in self.bigram_f(text):
            prob = self.conditionals[label].get(word, (1.0 * self.smoothening) / (self.smoothening * self.V))
            log_act += np.log(prob)
        return log_act

    def activation_stemmed(self, label, text):
        log_act = 0.0
        for word in text:
            word = stemmer.stem(word)
            if word not in stop_words:
                prob = self.conditionals[label].get(word, (1.0 * self.smoothening) / (self.smoothening * self.V))
                log_act += np.log(prob)
        return log_act



    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        num_words = [0.0, 0.0, 0.0, 0.0, 0.0]
        conditionals = [{} for _ in range(5)] 
        prior = np.zeros(5)  
        self.smoothening = smoothening

        for _, row in df.iterrows():
            label = row[class_col]
            num_words[label] += len(row[text_col])
            prior[label] += 1.0

            for word in row[text_col]:
                if word in conditionals[label]:
                    conditionals[label][word] += 1.0
                else:
                    conditionals[label][word] = 1.0

        prior /= prior.sum()
        self.prior = prior
        
        V = len(conditionals[1]) + len(conditionals[2]) + len(conditionals[3]) + len(conditionals[4])
        self.V = V

        for i in range(1, len(conditionals)):
            for word, count in conditionals[i].items():
                conditionals[i][word] = (count + smoothening) / (num_words[i] + smoothening * V)

        self.conditionals = conditionals

    def fit_stemmed(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        num_words = [0.0, 0.0, 0.0, 0.0, 0.0]
        conditionals = [{} for _ in range(5)] 
        prior = np.zeros(5)  
        self.smoothening = smoothening

        for _, row in df.iterrows():
            label = row[class_col]
            num_words[label] += len(row[text_col])
            prior[label] += 1.0

            for word in row[text_col]:
                word = stemmer.stem(word)
                if word in conditionals[label]:
                    conditionals[label][word] += 1.0
                else:
                    conditionals[label][word] = 1.0

        prior /= prior.sum()
        self.prior = prior
        
        V = len(conditionals[1]) + len(conditionals[2]) + len(conditionals[3]) + len(conditionals[4])
        self.V = V

        for i in range(1, len(conditionals)):
            for word, count in conditionals[i].items():
                conditionals[i][word] = (count + smoothening) / (num_words[i] + smoothening * V)

        self.conditionals = conditionals

    def bigram_fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        num_bigrams = [0.0, 0.0, 0.0, 0.0, 0.0]
        conditionals = [{} for _ in range(5)] 
        prior = np.zeros(5)  
        self.smoothening = smoothening

        for _, row in df.iterrows():
            label = row[class_col]

            bigrams = self.bigram_f(row[text_col])
            print(bigrams)
            num_bigrams[label] += len(bigrams)
            prior[label] += 1.0

            for bigram in bigrams:
                stemmed_bigram = (stemmer.stem(bigram[0]), stemmer.stem(bigram[1]))
                
                if stemmed_bigram in conditionals[label]:
                    conditionals[label][stemmed_bigram] += 1.0
                else:
                    conditionals[label][stemmed_bigram] = 1.0

        prior /= prior.sum()
        self.prior = prior
        
        V = sum(len(conditionals[i]) for i in range(5))
        self.V = V


        for i in range(5):
            for bigram, count in conditionals[i].items():
                conditionals[i][bigram] = (count + smoothening) / (num_bigrams[i] + smoothening * V)

        self.conditionals = conditionals



    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        predictions = []
        print("predict routine")
        for description in df[text_col]:
            predicted = 0
            max_act = -np.inf
            for label in range(1, 5):
                act = self.activation(label=label, text=description)
                if act > max_act:
                    predicted = label
                    max_act = act
            predictions.append(predicted)

        df[predicted_col] = predictions
        mis = 0
        for v in (df[predicted_col] - df["Class Index"]):
            if v != 0:
                mis += 1
        
        print(mis)
    
    def bigram_predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        predictions = []
        print("predict routine")
        for description in df[text_col]:
            predicted = 0
            max_act = -np.inf
            for label in range(1, 5):
                act = self.activation_bigram(label=label, text=description)
                if act > max_act:
                    predicted = label
                    max_act = act
            predictions.append(predicted)

        df[predicted_col] = predictions
        mis = 0
        for v in (df[predicted_col] - df["Class Index"]):
            if v != 0:
                mis += 1
        
        print(mis)


def tokeniser(class_col , text_col):
    df = pd.read_csv("../data/train.csv")
    df2 = pd.read_csv("../data/test.csv")
    df["Tokenized Description"] = df["Description"].apply(lambda x : x.split())
    df2["Tokenized Description"] = df2["Description"].apply(lambda x : x.split())
    nb = NaiveBayes()
    nb.bigram_fit(df=df , smoothening = 0.1)
    nb.bigram_predict(df2)


tokeniser("Class Index" , "Description")