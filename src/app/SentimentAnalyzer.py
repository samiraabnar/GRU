import numpy as np
import datetime
import sys
import pickle
sys.path.append('../../../')

from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *

from GRU.src.GRU2LwEmSentenceBased import *


class SentimentAnalyzer(object):
    def __init__(self):
        self.init_train_data()
        self.init_model()



    def init_train_data(self):
        self.train = {}
        self.test = {}
        self.train["sentences"], self.train["sentiments"], self.word_to_index, self.index_to_word, self.labels_count = DataPrep.load_one_hot_sentiment_data("../../data/sentiment/trainsentence_and_label.txt")
        #self.dev["sentences"], self.dev["sentiments"] = DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/devsentence_and_label.txt",self.word_to_index, self.index_to_word,self.labels_count)
        self.test["sentences"], self.test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/testsentence_and_label.txt",self.word_to_index, self.index_to_word,self.labels_count)

        self.vocab_size = len(self.index_to_word)

    def init_model(self):
        self.model = GRU2LwEmSentenceBased(input_dim=self.vocab_size, embedding_dim=300, output_dim=self.labels_count, hidden_dim1= 256, hidden_dim2=256)


    def test_model(self,num_examples_seen):
        pc_sentiment = np.zeros((len(self.test["sentences"]),self.labels_count))
        for i in np.arange(len(self.test["sentences"])):
            pc_sentiment[i] = self.model.predict(self.test["sentences"][i])

        correct = 0.0
        for i in np.arange(len(self.test["sentences"])):
            if np.argmax(pc_sentiment[i]) == np.argmax(self.test["sentiments"][i]):
                correct += 1

        accuracy = correct / len(self.test["sentences"])

        print("Accuracy: %f" %accuracy)


    def train_model(self):

            learning_rate = 0.001
            nepoch = 2
            decay = 0.9
            epochs_per_callback = 1

            expected_outputs = []
            for i in np.arange(len(self.train["sentences"])):
                s_out = np.zeros((len(self.train["sentences"][i]),self.labels_count),dtype=np.float32)
                s_out[-1] = self.train["sentiments"][i]
                expected_outputs.append(s_out)


            self.model.train_with_sgd(self.train["sentences"],expected_outputs, learning_rate, nepoch, decay, epochs_per_callback,self.test_model)

    def save(self):
        self.model.save_model_parameters_theano("FirstTrainedModel_0.txt")
        with open('dict_0' + '.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f)

    def load(self):
        with open('test' + '.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)

def prepare_data():
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/train.txt")
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/dev.txt")
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/test.txt")

if __name__ == '__main__':
    SA = SentimentAnalyzer()
    print("training ... ")
    SA.train_model()
    SA.save()

