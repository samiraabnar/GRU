import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import sys
import os
import time
from datetime import datetime
import csv
import itertools
import nltk
import sys
import io
import array
from datetime import datetime

sys.path.append('../../')
from Util.util.data.DataPrep import *


class GRU2LwEmSentenceBased(object):

    def __init__(self, input_dim, embedding_dim, output_dim, hidden_dim1= 128, hidden_dim2=128, bptt_truncate=-1):

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim


        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        Embedding = np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.embedding_dim, self.input_dim))

        U_update = {}
        W_update = {}
        b_update = {}
        U_reset = {}
        W_reset = {}
        b_reset = {}
        U_candidate = {}
        W_candidate = {}
        b_candidate = {}

        U_update[0] = np.random.uniform(-np.sqrt(1./self.embedding_dim), np.sqrt(1./self.embedding_dim), (self.hidden_dim1, self.embedding_dim))
        W_update[0] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim1, self.hidden_dim1))
        b_update[0] = np.zeros((self.hidden_dim1))
        U_reset[0] = np.random.uniform(-np.sqrt(1./self.embedding_dim), np.sqrt(1./self.embedding_dim), (self.hidden_dim1, self.embedding_dim))
        W_reset[0] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim1, self.hidden_dim1))
        b_reset[0] = np.zeros((self.hidden_dim1))
        U_candidate[0] = np.random.uniform(-np.sqrt(1./self.embedding_dim), np.sqrt(1./self.embedding_dim), (self.hidden_dim1, self.embedding_dim))
        W_candidate[0] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim1, self.hidden_dim1))
        b_candidate[0] = np.zeros((self.hidden_dim1))


        U_update[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim2, self.hidden_dim1))
        W_update[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim2), np.sqrt(1./self.hidden_dim2), (self.hidden_dim2, self.hidden_dim2))
        b_update[1] = np.zeros((self.hidden_dim2))
        U_reset[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim2, self.hidden_dim1))
        W_reset[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim2), np.sqrt(1./self.hidden_dim2), (self.hidden_dim2, self.hidden_dim2))
        b_reset[1] = np.zeros((self.hidden_dim2))
        U_candidate[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim1), np.sqrt(1./self.hidden_dim1), (self.hidden_dim2, self.hidden_dim1))
        W_candidate[1] = np.random.uniform(-np.sqrt(1./self.hidden_dim2), np.sqrt(1./self.hidden_dim2), (self.hidden_dim2, self.hidden_dim2))
        b_candidate[1] = np.zeros((self.hidden_dim2))

        V = np.random.uniform(-np.sqrt(1./self.hidden_dim2), np.sqrt(1./self.hidden_dim2), (self.output_dim, self.hidden_dim2))
        output_bias = np.zeros(self.output_dim)

        # Theano: Created shared variables

        self.U_update = {}
        self.U_reset = {}
        self.U_candidate = {}
        self.W_update = {}
        self.W_reset = {}
        self.W_candidate = {}
        self.b_update = {}
        self.b_reset = {}
        self.b_candidate = {}

        self.Embedding = theano.shared(name='E', value=Embedding.astype(theano.config.floatX))

        self.U_update[0] = theano.shared(name='U', value=U_update[0].astype(theano.config.floatX))
        self.W_update[0] = theano.shared(name='W', value=W_update[0].astype(theano.config.floatX))
        self.b_update[0] = theano.shared(name='b', value=b_update[0].astype(theano.config.floatX))

        self.U_reset[0] = theano.shared(name='U', value=U_reset[0].astype(theano.config.floatX))
        self.W_reset[0] = theano.shared(name='W', value=W_reset[0].astype(theano.config.floatX))
        self.b_reset[0] = theano.shared(name='b', value=b_reset[0].astype(theano.config.floatX))

        self.U_candidate[0] = theano.shared(name='U', value=U_candidate[0].astype(theano.config.floatX))
        self.W_candidate[0] = theano.shared(name='W', value=W_candidate[0].astype(theano.config.floatX))
        self.b_candidate[0] = theano.shared(name='b', value=b_candidate[0].astype(theano.config.floatX))

        self.U_update[1] = theano.shared(name='U', value=U_update[1].astype(theano.config.floatX))
        self.W_update[1] = theano.shared(name='W', value=W_update[1].astype(theano.config.floatX))
        self.b_update[1] = theano.shared(name='b', value=b_update[1].astype(theano.config.floatX))

        self.U_reset[1] = theano.shared(name='U', value=U_reset[1].astype(theano.config.floatX))
        self.W_reset[1] = theano.shared(name='W', value=W_reset[1].astype(theano.config.floatX))
        self.b_reset[1] = theano.shared(name='b', value=b_reset[1].astype(theano.config.floatX))

        self.U_candidate[1] = theano.shared(name='U', value=U_candidate[1].astype(theano.config.floatX))
        self.W_candidate[1] = theano.shared(name='W', value=W_candidate[1].astype(theano.config.floatX))
        self.b_candidate[1] = theano.shared(name='b', value=b_candidate[1].astype(theano.config.floatX))

        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.output_bias = theano.shared(name='c', value=output_bias.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(Embedding.shape).astype(theano.config.floatX))

        self.mU_update = {}
        self.mW_update = {}
        self.mb_update = {}
        self.mU_reset = {}
        self.mW_reset = {}
        self.mb_reset = {}
        self.mU_candidate = {}
        self.mW_candidate = {}
        self.mb_candidate = {}


        self.mU_update[0] = theano.shared(name='mU', value=np.zeros(U_update[0].shape).astype(theano.config.floatX))
        self.mU_reset[0] = theano.shared(name='mU', value=np.zeros(U_reset[0].shape).astype(theano.config.floatX))
        self.mU_candidate[0] = theano.shared(name='mU', value=np.zeros(U_candidate[0].shape).astype(theano.config.floatX))

        self.mW_update[0] = theano.shared(name='mU', value=np.zeros(W_update[0].shape).astype(theano.config.floatX))
        self.mW_reset[0] = theano.shared(name='mU', value=np.zeros(W_reset[0].shape).astype(theano.config.floatX))
        self.mW_candidate[0] = theano.shared(name='mU', value=np.zeros(W_candidate[0].shape).astype(theano.config.floatX))

        self.mb_update[0] = theano.shared(name='mU', value=np.zeros(b_update[0].shape).astype(theano.config.floatX))
        self.mb_reset[0] = theano.shared(name='mU', value=np.zeros(b_reset[0].shape).astype(theano.config.floatX))
        self.mb_candidate[0] = theano.shared(name='mU', value=np.zeros(b_candidate[0].shape).astype(theano.config.floatX))

        self.mU_update[1] = theano.shared(name='mU', value=np.zeros(U_update[1].shape).astype(theano.config.floatX))
        self.mU_reset[1] = theano.shared(name='mU', value=np.zeros(U_reset[1].shape).astype(theano.config.floatX))
        self.mU_candidate[1] = theano.shared(name='mU', value=np.zeros(U_candidate[1].shape).astype(theano.config.floatX))

        self.mW_update[1] = theano.shared(name='mU', value=np.zeros(W_update[1].shape).astype(theano.config.floatX))
        self.mW_reset[1] = theano.shared(name='mU', value=np.zeros(W_reset[1].shape).astype(theano.config.floatX))
        self.mW_candidate[1] = theano.shared(name='mU', value=np.zeros(W_candidate[1].shape).astype(theano.config.floatX))

        self.mb_update[1] = theano.shared(name='mU', value=np.zeros(b_update[1].shape).astype(theano.config.floatX))
        self.mb_reset[1] = theano.shared(name='mU', value=np.zeros(b_reset[1].shape).astype(theano.config.floatX))
        self.mb_candidate[1] = theano.shared(name='mU', value=np.zeros(b_candidate[1].shape).astype(theano.config.floatX))

        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mOutputBias = theano.shared(name='mc', value=np.zeros(output_bias.shape).astype(theano.config.floatX))

        self.__theano_build__()

    def __theano_build__(self):
        E, V, U_update, U_reset, U_candidate, W_update, W_reset, W_candidate, b_update, b_reset, b_candidate, output_bias = \
            self.Embedding, self.V, self.U_update, self.U_reset, self.U_candidate, \
                                   self.W_update, self.W_reset, self.W_candidate, self.b_update, self.b_reset,\
                                   self.b_candidate, self.output_bias

        x = T.imatrix('x').astype(theano.config.floatX)
        y = T.imatrix('y').astype(theano.config.floatX)

        def forward_prop_step(x_t, s_1_prev, s_2_prev):

            # Word Embeding layer
            x_e = E.dot(x_t.T)
            x_e = x_e.astype(theano.config.floatX)

            # GRU Layer 1
            update_gate_1 = T.nnet.hard_sigmoid(U_update[0].dot(x_e) + W_update[0].dot(s_1_prev) + b_update[0])
            reset_gate_1 = T.nnet.hard_sigmoid(U_reset[0].dot(x_e) + W_reset[0].dot(s_1_prev) + b_reset[0])
            c_1 = T.tanh(U_candidate[0].dot(x_e) + W_candidate[0].dot(s_1_prev * reset_gate_1) + b_candidate[0])
            s_1 = (T.ones_like(update_gate_1) - update_gate_1) * c_1 + update_gate_1 * s_1_prev

            # GRU Layer 2
            update_gate_2 = T.nnet.hard_sigmoid(U_update[1].dot(s_1) + W_update[1].dot(s_2_prev) + b_update[1])
            reset_gate_2 = T.nnet.hard_sigmoid(U_reset[1].dot(s_1) + W_reset[1].dot(s_2_prev) + b_reset[1])
            c_2 = T.tanh(U_candidate[1].dot(s_1) + W_candidate[1].dot(s_2_prev * reset_gate_2) + b_candidate[1])
            s_2 = (T.ones_like(update_gate_2) - update_gate_2) * c_2 + update_gate_2 * s_2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_2) + output_bias)[0]

            print_update_gate_1 = theano.printing.Print('UpdateGate_1 for ')
            print_update_gate_1(update_gate_1)

            print_update_gate_2 = theano.printing.Print('UpdateGate_2 for ')
            print_update_gate_2(update_gate_2)

            return [o_t, s_1, s_2]

        [o, s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim1)),
                          dict(initial=T.zeros(self.hidden_dim2))])

        prediction = T.argmax(o)
        o_error = T.sum(T.nnet.categorical_crossentropy(o[-1], y[-1]))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dU_update = {}
        dU_reset = {}
        dU_candidate = {}

        dW_update = {}
        dW_reset = {}
        dW_candidate = {}

        db_update = {}
        db_reset = {}
        db_candidate = {}

        dE = T.grad(cost, E)

        dU_update[0] = T.grad(cost, U_update[0])
        dU_reset[0] = T.grad(cost, U_reset[0])
        dU_candidate[0] = T.grad(cost, U_candidate[0])

        dW_update[0] = T.grad(cost, W_update[0])
        dW_reset[0] = T.grad(cost, W_reset[0])
        dW_candidate[0] = T.grad(cost, W_candidate[0])

        db_update[0] = T.grad(cost, b_update[0])
        db_reset[0] = T.grad(cost, b_reset[0])
        db_candidate[0] = T.grad(cost, b_candidate[0])

        dU_update[1] = T.grad(cost, U_update[1])
        dU_reset[1] = T.grad(cost, U_reset[1])
        dU_candidate[1] = T.grad(cost, U_candidate[1])

        dW_update[1] = T.grad(cost, W_update[1])
        dW_reset[1] = T.grad(cost, W_reset[1])
        dW_candidate[1] = T.grad(cost, W_candidate[1])

        db_update[1] = T.grad(cost, b_update[1])
        db_reset[1] = T.grad(cost, b_reset[1])
        db_candidate[1] = T.grad(cost, b_candidate[1])


        dV = T.grad(cost, V)
        dOutputBias = T.grad(cost, output_bias)

        # Assign functions
        self.predict = theano.function([x], o[-1])
        self.predict_class = theano.function([x], prediction)
        self.calculate_cost = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE,
                                             dU_update[0], dU_reset[0],dU_candidate[0],
                                             dW_update[0], dW_reset[0],dW_candidate[0],
                                             db_update[0], db_reset[0],db_candidate[0],

                                             dU_update[1], dU_reset[1],dU_candidate[1],
                                             dW_update[1], dW_reset[1],dW_candidate[1],
                                             db_update[1], db_reset[1],db_candidate[1],

                                             dV, dOutputBias])

        # SGD parameters
        learning_rate = T.scalar('learning_rate').astype(theano.config.floatX)
        decay = T.scalar('decay').astype(theano.config.floatX)

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2

        mU_update = {}
        mU_reset = {}
        mU_candidate = {}

        mW_update = {}
        mW_reset = {}
        mW_candidate = {}

        mb_update = {}
        mb_reset = {}
        mb_candidate = {}

        mU_update[0] = decay * self.mU_update[0] + (1 - decay) * dU_update[0] ** 2
        mU_reset[0] = decay * self.mU_reset[0] + (1 - decay) * dU_reset[0] ** 2
        mU_candidate[0] = decay * self.mU_candidate[0] + (1 - decay) * dU_candidate[0] ** 2

        mW_update[0] = decay * self.mW_update[0] + (1 - decay) * dW_update[0] ** 2
        mW_reset[0] = decay * self.mW_reset[0] + (1 - decay) * dW_reset[0] ** 2
        mW_candidate[0] = decay * self.mW_candidate[0] + (1 - decay) * dW_candidate[0] ** 2

        mb_update[0] = decay * self.mb_update[0] + (1 - decay) * db_update[0] ** 2
        mb_reset[0] = decay * self.mb_reset[0] + (1 - decay) * db_reset[0] ** 2
        mb_candidate[0] = decay * self.mb_candidate[0] + (1 - decay) * db_candidate[0] ** 2


        mU_update[1] = decay * self.mU_update[1] + (1 - decay) * dU_update[1] ** 2
        mU_reset[1] = decay * self.mU_reset[1] + (1 - decay) * dU_reset[1] ** 2
        mU_candidate[1] = decay * self.mU_candidate[1] + (1 - decay) * dU_candidate[1] ** 2

        mW_update[1] = decay * self.mW_update[1] + (1 - decay) * dW_update[1] ** 2
        mW_reset[1] = decay * self.mW_reset[1] + (1 - decay) * dW_reset[1] ** 2
        mW_candidate[1] = decay * self.mW_candidate[1] + (1 - decay) * dW_candidate[1] ** 2

        mb_update[1] = decay * self.mb_update[1] + (1 - decay) * db_update[1] ** 2
        mb_reset[1] = decay * self.mb_reset[1] + (1 - decay) * db_reset[1] ** 2
        mb_candidate[1] = decay * self.mb_candidate[1] + (1 - decay) * db_candidate[1] ** 2

        mV = decay * self.mV + (1 - decay) * dV ** 2
        mOutputBias = decay * self.mOutputBias + (1 - decay) * dOutputBias ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U_update[0], U_update[0] - learning_rate * dU_update[0] / T.sqrt(mU_update[0] + 1e-6)),
                     (U_reset[0], U_reset[0] - learning_rate * dU_reset[0] / T.sqrt(mU_reset[0] + 1e-6)),
                     (U_candidate[0], U_candidate[0] - learning_rate * dU_candidate[0] / T.sqrt(mU_candidate[0] + 1e-6)),
                     (W_update[0], W_update[0] - learning_rate * dW_update[0] / T.sqrt(mW_update[0] + 1e-6)),
                     (W_reset[0], W_reset[0] - learning_rate * dW_reset[0] / T.sqrt(mW_reset[0] + 1e-6)),
                     (W_candidate[0], W_candidate[0] - learning_rate * dW_candidate[0] / T.sqrt(mW_candidate[0] + 1e-6)),
                     (b_update[0], b_update[0] - learning_rate * db_update[0] / T.sqrt(mb_update[0] + 1e-6)),
                     (b_reset[0], b_reset[0] - learning_rate * db_reset[0] / T.sqrt(mb_reset[0] + 1e-6)),
                     (b_candidate[0], b_candidate[0] - learning_rate * db_candidate[0] / T.sqrt(mb_candidate[0] + 1e-6)),

                     (U_update[1], U_update[1] - learning_rate * dU_update[1] / T.sqrt(mU_update[1] + 1e-6)),
                     (U_reset[1], U_reset[1] - learning_rate * dU_reset[1] / T.sqrt(mU_reset[1] + 1e-6)),
                     (U_candidate[1], U_candidate[1] - learning_rate * dU_candidate[1] / T.sqrt(mU_candidate[1] + 1e-6)),
                     (W_update[1], W_update[1] - learning_rate * dW_update[1] / T.sqrt(mW_update[1] + 1e-6)),
                     (W_reset[1], W_reset[1] - learning_rate * dW_reset[1] / T.sqrt(mW_reset[1] + 1e-6)),
                     (W_candidate[1], W_candidate[1] - learning_rate * dW_candidate[1] / T.sqrt(mW_candidate[1] + 1e-6)),
                     (b_update[1], b_update[1] - learning_rate * db_update[1] / T.sqrt(mb_update[1] + 1e-6)),
                     (b_reset[1], b_reset[1] - learning_rate * db_reset[1] / T.sqrt(mb_reset[1] + 1e-6)),
                     (b_candidate[1], b_candidate[1] - learning_rate * db_candidate[1] / T.sqrt(mb_candidate[0] + 1e-6)),

                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (output_bias, output_bias - learning_rate * dOutputBias / T.sqrt(mOutputBias + 1e-6)),
                     (self.mE, mE),

                     (self.mU_update[0], mU_update[0]),
                     (self.mU_reset[0], mU_reset[0]),
                     (self.mU_candidate[0], mU_candidate[0]),
                     (self.mW_update[0], mW_update[0]),
                     (self.mW_reset[0], mW_reset[0]),
                     (self.mW_candidate[0], mW_candidate[0]),
                     (self.mb_update[0], mb_update[0]),
                     (self.mb_reset[0], mb_reset[0]),
                     (self.mb_candidate[0], mb_candidate[0]),

                     (self.mU_update[1], mU_update[1]),
                     (self.mU_reset[1], mU_reset[1]),
                     (self.mU_candidate[1], mU_candidate[1]),
                     (self.mW_update[1], mW_update[1]),
                     (self.mW_reset[1], mW_reset[1]),
                     (self.mW_candidate[1], mW_candidate[1]),
                     (self.mb_update[1], mb_update[1]),
                     (self.mb_reset[1], mb_reset[1]),
                     (self.mb_candidate[1], mb_candidate[1]),

                     (self.mV, mV),
                     (self.mOutputBias, mOutputBias)
                    ])






    def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
        callback_every=10000, callback=None, *args):
        num_examples_seen = 0
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
                num_examples_seen += 1
                # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(num_examples_seen, *args)
        return model

    def calculate_total_loss(self, X, Y):
        return np.sum([self.calculate_cost(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

    def save_model_parameters_theano(model, outfile):
        np.savez(outfile,
            Embedding=model.Embedding.get_value(),
            U_update_0=model.U_update[0].get_value(),
            U_update_1=model.U_update[1].get_value(),
            U_reset_0=model.U_reset[0].get_value(),
            U_reset_1=model.U_reset[1].get_value(),
            U_candidate_0=model.U_candidate[0].get_value(),
            U_candidate_1=model.U_candidate[1].get_value(),

            W_update_0=model.W_update[0].get_value(),
            W_update_1=model.W_update[1].get_value(),

            W_reset_0=model.W_reset[0].get_value(),
            W_reset_1=model.W_reset[1].get_value(),
            W_candidate_0=model.W_candidate[0].get_value(),
            W_candidate_1=model.W_candidate[1].get_value(),
            b_update_0=model.b_update[0].get_value(),
            b_update_1=model.b_update[1].get_value(),
            b_reset_0=model.b_reset[0].get_value(),
            b_reset_1=model.b_reset[1].get_value(),
            b_candidate_0=model.b_candidate[0].get_value(),
            b_candidate_1=model.b_candidate[1].get_value(),
            V=model.V.get_value(),
            output_bias=model.output_bias.get_value())
        print("Saved model parameters to %s." % outfile)




    @staticmethod
    def load_model_parameters_theano(path):
        modelFile = np.load(path)
        E, U_update_0, U_update_1,U_reset_0, U_reset_1,U_candidate_0,U_candidate_1, W_update_0, W_update_1,W_reset_0,W_reset_1,W_candidate_0,W_candidate_1,b_update_0,b_update_1,b_reset_0,b_reset_1,b_candidate_0,b_candidate_1, V, ob = \
                           modelFile["Embedding"], modelFile["U_update_0"], modelFile["U_update_1"],modelFile["U_reset_0"], modelFile["U_reset_1"],modelFile["U_candidate_0"], modelFile["U_candidate_1"],\
                                                   modelFile["W_update_0"],modelFile["W_update_1"],modelFile["W_reset_0"], modelFile["W_reset_1"], modelFile["W_candidate_0"], modelFile["W_candidate_1"],\
                                                   modelFile["b_update_0"],modelFile["b_update_1"],modelFile["b_reset_0"],modelFile["b_reset_1"],modelFile["b_candidate_0"],modelFile["b_candidate_1"],\
                                                   \
                                                   \
                           modelFile["V"],  modelFile["output_bias"]
        hidden_dim, word_dim = E.shape[0], E.shape[1]
        print("Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim))
        sys.stdout.flush()
        model = GRU2LwEmSentenceBased(input_dim=E.shape[1],embedding_dim=E.shape[0], hidden_dim1=U_update_0.shape[0],hidden_dim2=U_update_1.shape[0],output_dim=V.shape[0])
        model.Embedding.set_value(E)
        model.U_update[0].set_value(U_update_0)
        model.U_reset[0].set_value(U_reset_0)
        model.U_candidate[0].set_value(U_candidate_0)
        model.W_update[0].set_value(W_update_0)
        model.W_reset[0].set_value(W_reset_0)
        model.W_candidate[0].set_value(W_candidate_0)
        model.b_update[0].set_value(b_update_0)
        model.b_reset[0].set_value(b_reset_0)
        model.b_candidate[0].set_value(b_candidate_0)
        model.U_update[1].set_value(U_update_1)
        model.U_reset[1].set_value(U_reset_1)
        model.U_candidate[1].set_value(U_candidate_1)
        model.W_update[1].set_value(W_update_1)
        model.W_reset[1].set_value(W_reset_1)
        model.W_candidate[1].set_value(W_candidate_1)
        model.b_update[1].set_value(b_update_1)
        model.b_reset[1].set_value(b_reset_1)
        model.b_candidate[1].set_value(b_candidate_1)

        model.V.set_value(V)
        model.output_bias.set_value(ob)
        return model

if __name__ == '__main__':

   # model = GRU2LwEmSentenceBased(input_dim=2,embedding_dim=10, output_dim=2)
   # model.save_model_parameters_theano("test")
    model1 = GRU2LwEmSentenceBased.load_model_parameters_theano("test.npz")
#    learning_rate = 0.001
#    x_train = [[1,2],[1,1]]
#    y_train = [[2,4],[2,2]]
    # Print SGD step time
#    t1 = time.time()
#    model.sgd_step(x_train, y_train, learning_rate)

#    t2 = time.time()
#    print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
#    sys.stdout.flush()

    # We do this every few examples to understand what's going on


    #for epoch in range(NEPOCH):
    #  model.train_with_sgd(x_train, y_train,index_to_word, word_to_index, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    #    callback_every=PRINT_EVERY, callback=model.sgd_callback)
