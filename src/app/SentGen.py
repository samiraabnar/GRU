import numpy as np
import datetime
import sys
sys.path.append('../../../')

from Util.util.data.DataPrep import *
from GRU.src.GRU2LwE import *

class SentGen(object):

    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.x_train, self.y_train, self.word_to_index, self.index_to_word = \
            DataPrep.load_data_reditcomment("../../data/reddit-comments-2015-08.csv", vocabulary_size= self.vocab_size)

        self.model = GRU2LwE(input_dim=vocab_size,embedding_dim=48, output_dim=vocab_size, hidden_dim1=20, hidden_dim2=20)




    def print_sentence(self,s, index_to_word):
        sentence_str = [index_to_word[x] for x in s[1:-1]]
        print(" ".join(sentence_str))

    def generate_sentence(self, index_to_word, word_to_index, min_length=5):
        # We start the sentence with the start token
        new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
            one_hot_new_sentence = np.eye(len(index_to_word))[new_sentence]
            next_word_probs = self.model.predict(one_hot_new_sentence)[-1]
            samples = np.random.multinomial(1, next_word_probs)
            sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
            # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
            # And: We don't want sentences with UNKNOWN_TOKEN's
            if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
                return None
        if len(new_sentence) < min_length:
            return None
        return new_sentence

    def generate_sentences(self, n, index_to_word, word_to_index):
        for i in range(n):
            sent = None
            while not sent:
                sent = self.generate_sentence(index_to_word, word_to_index)
            self.print_sentence(sent, index_to_word)

    def sgd_callback(self, num_examples_seen,x_train,y_train,index_to_word,word_to_index):
        dt = datetime.now().isoformat()
        loss = self.model.calculate_loss(x_train[:10000], y_train[:10000])
        print("\n%s (%d)" % (dt, num_examples_seen))
        print("--------------------------------------------------")
        print("Loss: %f" % loss)
        self.generate_sentences(10, index_to_word, word_to_index)
        print("\n")
        sys.stdout.flush()

    def learn_to_make_sentences(self, nepoch=10, learning_rate=0.001):
        for epoch in range(nepoch):
            self.model.train_with_sgd(self.x_train, self.y_train,learning_rate, 1, 0.9,
            1, self.sgd_callback,self.x_train , self.y_train,self.index_to_word, self.word_to_index)



if __name__ == '__main__':
      sg = SentGen(vocab_size=1000)
      print("Starting to learn ....")
      sg.learn_to_make_sentences()

