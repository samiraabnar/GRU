import numpy as np
import datetime
import sys
import pickle
sys.path.append('../../../')

from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *

from GRU.src.GRU2LwEmSentenceBased import *
from Util.util.visual.Visualizer import *
from Util.util.math.MathUtil import *

import matplotlib.pyplot as plt
import time


class SentimentAnalyzer(object):



    def init_train_data(self):
        self.train = {}
        self.test = {}
        self.train["sentences"], self.train["sentiments"], self.word_to_index, self.index_to_word, self.labels_count = DataPrep.load_one_hot_sentiment_data("../../data/sentiment/trainsentence_and_label_binary.txt")
        #self.dev["sentences"], self.dev["sentiments"] = DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/devsentence_and_label.txt",self.word_to_index, self.index_to_word,self.labels_count)
        self.test["sentences"], self.test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/testsentence_and_label_binary.txt",self.word_to_index, self.index_to_word,self.labels_count)

        self.vocab_size = len(self.index_to_word)

    def init_model(self):
        self.model = GRU2LwEmSentenceBased(input_dim=self.vocab_size, embedding_dim=300, output_dim=self.labels_count, hidden_dim1= 128, hidden_dim2=128)

    def test_model(self,num_examples_seen):
        pc_sentiment = np.zeros((len(self.test["sentences"]),self.labels_count))
        for i in np.arange(len(self.test["sentences"])):
            pc_sentiment[i] = self.model.predict(self.test["sentences"][i])

        correct = 0.0
        for i in np.arange(len(self.test["sentences"])):
            if np.argmax(pc_sentiment[i]) == np.argmax(self.test["sentiments"][i]):
                correct += 1

        accuracy = correct / len(self.test["sentences"])

        print("Accuracy on test: %f" %accuracy)


        pc_sentiment = np.zeros((len(self.train["sentences"]),self.labels_count))
        for i in np.arange(len(self.train["sentences"])):
            pc_sentiment[i] = self.model.predict(self.train["sentences"][i])

        correct = 0.0
        for i in np.arange(len(self.train["sentences"])):
            if np.argmax(pc_sentiment[i]) == np.argmax(self.train["sentiments"][i]):
                correct += 1

        accuracy = correct / len(self.train["sentences"])

        print("Accuracy on train: %f" %accuracy)

    def train_model(self):
        self.init_train_data()
        self.model = GRU2LwEmSentenceBased(input_dim=self.vocab_size, embedding_dim=300, output_dim=self.labels_count, hidden_dim1= 100, hidden_dim2=100)

        learning_rate = 0.001
        nepoch = 5
        decay = 0.9
        epochs_per_callback = 1

        expected_outputs = []
        for i in np.arange(len(self.train["sentences"])):
            s_out = np.zeros((len(self.train["sentences"][i]),self.labels_count),dtype=np.float32)
            s_out[-1] = self.train["sentiments"][i]
            expected_outputs.append(s_out)
        print("training ... ")
        self.model.train_with_sgd(self.train["sentences"],expected_outputs, learning_rate, nepoch, decay, epochs_per_callback,self.test_model)

    def save(self):
        self.model.save_model_parameters_theano("FirstTrainedModel_binary_5.txt")
        with open('dict_binary_5' + '.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f)

    def load(self):
        with open('dict_0' + '.pkl', 'rb') as f:
            self.word_to_index = pickle.load(f)

        self.index_to_word = [""] * len(self.word_to_index.keys())

        for item in self.word_to_index.keys():
            self.index_to_word[self.word_to_index[item]] = item

        self.vocab_size = len(self.index_to_word)
        self.model = GRU2LwEmSentenceBased.load_model_parameters_theano("FirstTrainedModel_2.txt.npz")
        self.labels_count = self.model.outputDim()



    @staticmethod
    def model_analyzer():
        parameters = GRU2LwEmSentenceBased.load_model_parameters_only("FirstTrainedModel_2.txt.npz")
        labels_count = parameters["output_bias"].shape[0]
        word_to_index = {}
        index_to_word = []
        with open('dict_0' + '.pkl', 'rb') as f:
            word_to_index = pickle.load(f)

        index_to_word = [""] * len(word_to_index.keys())

        for item in word_to_index.keys():
            index_to_word[word_to_index[item]] = item

        vocab_size = len(index_to_word)
        #Visualizer.plot_vector([(parameters["V"][0],'blue'),(parameters["V"][1],'green'),(parameters["V"][2],'red'),(parameters["V"][3],'orange'),(parameters["V"][4],'yellow')])

        test = {}
        test["sentences"], test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../../data/sentiment/analys.txt",word_to_index, index_to_word,labels_count)

        pc_sentiment = np.zeros((len(test["sentences"]),len(parameters["output_bias"])))

        for sent in test["sentences"]:
            s_1 = np.zeros(parameters["U_update"][0].shape[0])
            s_2 = np.zeros(parameters["U_update"][1].shape[0])
            o_t = []

            """ plt.ion()
            plt.bar(np.arange(100)+0.5,np.zeros(100),width=1,color='blue')
            plt.show()
            plt.pause(0.0001)"""
            update_gate_2_values = []
            update_gate_2_words = []

            reset_gate_1s = []
            update_gate_1s = []

            reset_gate_2s = []
            update_gate_2s = []

            hidden_state_1s = []
            hidden_state_2s = []

            for word in sent:
                x_e = parameters["Embedding"].dot(word.T)

                update_gate_1 = MathUtil.sigmoid(parameters["U_update"][0].dot(x_e) + parameters["W_update"][0].dot(s_1) + parameters["b_update"][0])
                reset_gate_1 = MathUtil.sigmoid(parameters["U_reset"][0].dot(x_e) + parameters["W_reset"][0].dot(s_1) + parameters["b_reset"][0])

                c_1 = np.tanh(parameters["U_candidate"][0].dot(x_e) + parameters["W_candidate"][0].dot(s_1 * reset_gate_1) + parameters["b_candidate"][0])
                s_1 = (1 - update_gate_1) * c_1 + update_gate_1 * s_1

                # GRU Layer 2
                update_gate_2 = MathUtil.sigmoid(parameters["U_update"][1].dot(s_1) + parameters["W_update"][1].dot(s_2) + parameters["b_update"][1])
                reset_gate_2 = MathUtil.sigmoid(parameters["U_reset"][1].dot(s_1) + parameters["W_reset"][1].dot(s_2) + parameters["b_reset"][1])
                c_2 = np.tanh(parameters["U_candidate"][1].dot(s_1) + parameters["W_candidate"][1].dot(s_2 * reset_gate_2) + parameters["b_candidate"][1])
                s_2 = (1 - update_gate_2) * c_2 + update_gate_2 * s_2

                o_l2 = MathUtil.softmax(parameters["V"].dot(s_2) + parameters["output_bias"])
                o_l1 = MathUtil.softmax(parameters["V"].dot(s_1) + parameters["output_bias"])


                """ plt.clf()
                #plt.bar(np.arange(0,200,2),np.abs(parameters["V"][4]),width=2,color='yellow')
                plt.bar(np.arange(0,200,2),s_2,width=1,color='blue')
                plt.bar(np.arange(0,200,2)+1,update_gate_2,width=1,color='red')
                plt.draw()
                plt.pause(0.0001)
                name = input("")"""
                print(index_to_word[np.argmax(word)]+" --> Sentiment So far ol2: " + str(np.argmax(o_l2))+ " ol1: "+str(np.argmax(o_l1)))

                ug1 = np.mean(update_gate_1 * parameters["V"], axis=1)
                rg1 = np.mean(reset_gate_1 * parameters["V"], axis=1)
                ug2 = np.mean(update_gate_2 * parameters["V"], axis=1)
                rg2 = np.mean(reset_gate_2 * parameters["V"], axis=1)

                reset_gate_1s.append(reset_gate_1)
                update_gate_1s.append(update_gate_1)

                reset_gate_2s.append(reset_gate_2)
                update_gate_2s.append(update_gate_2)


                hidden_state_1s.append(s_1)
                hidden_state_2s.append(s_2)





                """plt.clf()
                plt.bar(np.arange(0,10,2),ug1,width=1,color='blue')
                plt.bar(np.arange(0,10,2)+1,ug2,width=1,color='red')

                plt.draw()
                plt.pause(0.0001)
                name = input("")
                """

                #plt.bar(np.arange(0,200,2),np.abs(parameters["V"][4]),width=2,color='yellow
                # ')
                #plt.bar(np.arange(0,200,2),update_gate_2,width=1,color='blue')
                #plt.bar(np.arange(0,200,2)+1,reset_gate_2,width=1,color='red')
                #plt.draw()
                #plt.pause(0.0001)
                #name = input("")
            #plt.clf()
            #plt.plot(np.arange(1,len(update_gate_2_values)*10,10),update_gate_2_values,"ro")
            #plt.xticks(np.arange(1,len(update_gate_2_values)*10,10),update_gate_2_words,rotation='vertical')
            #plt.show()

           # plt.clf()

           # plt.show()
            print("reset gates 1: ")
            print(reset_gate_1s)
            print("update gates 1: ")
            print(update_gate_1s)
            print("hidden state 1: ")
            print(hidden_state_1s)
            name = input("Continue? ")
            print("----------------------------------------")








def prepare_data():
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/train.txt")
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/dev.txt")
    FileUtil.get_sentence_and_label_from_tree_annotation("../../data/sentiment/trees/test.txt")

if __name__ == '__main__':
    SA = SentimentAnalyzer()
    #SentimentAnalyzer.model_analyzer()
    SA.train_model()
    SA.save()

