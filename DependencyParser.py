import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()"""
                
            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size,Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.float32, shape=[Config.batch_size,parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens])
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.reshape(embed, [Config.batch_size, -1])
            #self.keep_prob = tf.placeholder(tf.float32)

            #splitting the train inputs
            self.train_input_words, self.train_input_pos, self.train_input_labels = tf.split(self.train_inputs,[18,18,12],1)
            embed_words = tf.nn.embedding_lookup(self.embeddings, self.train_input_words)
            embed_pos = tf.nn.embedding_lookup(self.embeddings, self.train_input_pos)
            embed_labels= tf.nn.embedding_lookup(self.embeddings, self.train_input_labels)
            embed_words = tf.reshape(embed_words, [Config.batch_size, -1])
            embed_pos = tf.reshape(embed_pos, [Config.batch_size, -1])
            embed_labels = tf.reshape(embed_labels, [Config.batch_size, -1])



            weights_input = tf.Variable(
                tf.random_normal([Config.hidden_size, Config.n_Tokens*Config.embedding_size],
                                    stddev=0.1))

            """weights_input3 = tf.Variable(
                tf.random_normal([Config.hidden_size, Config.hidden_size],
                                 stddev=0.1))"""

            biases_input = tf.Variable(tf.zeros([Config.hidden_size]))
            weights_output = tf.Variable(
                tf.random_normal([parsing_system.numTransitions(),Config.hidden_size],
                                    stddev=0.1))
            #self.train_labels = tf.reshape(self.train_labels,[Config.batch_size,-1])

            #2) Call forward_pass and get predictions
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)

            ##forward pass function call for 2 hidden layers
            # self.prediction = self.forward_pass_2_layers(embed, weights_input, weights_input3, biases_input,
            # weights_output)

            ### forward pass function call for 3 hidden layers
            # self.prediction = self.forward_pass_3_layers(embed, weights_input, weights_input2, weights_input3, biases_input,
            # weights_output)

            ### forward pass function call for seperate(parallel) hidden layers
            # self.prediction = self.forward_pass_parallel(embed_words,embed_pos,embed_labels, weights_input, biases_input, weights_output)



            """3) Implement the loss function described in the paper
             - lambda is defined at Config.lam"""
            ##Removing the non fesible transitions
            cond = tf.equal(self.train_labels,-1)
            true_transitions = tf.reshape(tf.multiply(tf.ones([Config.batch_size*parsing_system.numTransitions()], tf.float32),0.0),[Config.batch_size,parsing_system.numTransitions()])
            new_preds = tf.where(cond,true_transitions,self.prediction)
            new_labels = tf.where(cond, true_transitions,self.train_labels)

            ##Calculating loss and regularization
            ##Loss : cross entropy loss
            regularizations = 0.5*Config.lam * (tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(
                embed) + tf.nn.l2_loss(biases_input))
            self.loss = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=new_preds,labels=new_labels))+regularizations)

            
            #===================================================================



            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print("Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        h = tf.pow(tf.matmul(weights_input,tf.transpose(embed)),3)
        pred = tf.matmul(weights_output, h)
        return tf.transpose(pred)

        ### forward pass function for 2 hidden layers
    """def forward_pass_2_layers(self, embed, weights_input, weights_input2, biases_inpu, weights_output):
        h = tf.pow(tf.matmul(weights_input, tf.transpose(embed)), 3)
        layer1 = tf.nn.dropout(h,self.keep_prob)
        weights_input3 = tf.Variable(
                tf.random_normal([Config.hidden_size, Config.hidden_size],
                                 stddev=0.1))
        h2 = tf.pow(tf.matmul(weights_input2, layer1), 3)
        pred2 = tf.matmul(weights_output, h2)
        return tf.transpose(pred2)"""

    ###forward pass function for 3 hidden layers
    """def forward_pass_3_layers(self, embed, weights_input, weights_input2, weights_input3, biases_inpu, weights_output):
        h = tf.pow(tf.matmul(weights_input, tf.transpose(embed)), 3)
        layer1 = tf.nn.dropout(h,self.keep_prob)
        h2 = tf.pow(tf.matmul(weights_input2, layer1), 3)
        layer2 = tf.nn.dropout(h2,self.keep_prob)
        h3 = tf.pow(tf.matmul(weights_input3, layer2),3)
        pred3 = tf.matmul(weights_output, h3)
        return tf.transpose(pred3)"""

    ###forward pass function for parallel hidden layers
    def forward_pass_parallel(self, embed_words,embed_pos,embed_labels, weights_input, biases_inpu, weights_output):

        weights_words, weights_pos, weights_labels = tf.split(weights_input,[18*50.18*50,12*50],1)
        h1 = tf.add(tf.pow(tf.matmul(weights_words,tf.transpose(embed_words)),3), tf.pow(tf.matmul(weights_pos,tf.transpose(embed_pos)),3))
        h = tf.add(h1, tf.pow(tf.matmul(weights_labels,tf.transpose(embed_labels)),3))
        pred = tf.matmul(weights_output,h)
        return tf.transpose(pred)


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    ### Collect the words, pos tags of words, labels of words
    ### Words: include the left and the right children.

    features = []
    features1 = []
    s1 = c.getStack(0)
    features.append(s1)
    s2 = c.getStack(1)
    features.append(s2)
    s3 = c.getStack(2)
    features.append(s3)
    b1 = c.getBuffer(0)
    features.append(b1)
    b2 = c.getBuffer(1)
    features.append(b1)
    b3 = c.getBuffer(2)
    features.append(b3)
    lc1 = c.getLeftChild(s1,1)
    features.append(lc1)
    lc2 = c.getLeftChild(s2,1)
    features.append(lc2)
    lc3 = c.getLeftChild(s3,1)
    features.append(lc3)
    rc1 = c.getRightChild(s1,1)
    features.append(rc1)
    rc2 = c.getRightChild(s2,1)
    features.append(rc2)
    rc3 = c.getRightChild(s3,1)
    features.append(rc3)
    left_lc1 = c.getLeftChild(lc1,1)
    features.append(left_lc1)
    left_lc2 = c.getLeftChild(lc2,1)
    features.append(left_lc2)
    left_lc3 = c.getLeftChild(lc3,1)
    features.append(left_lc3)
    right_rc1 = c.getRightChild(rc1,1)
    features.append(right_rc1)
    right_rc2 = c.getRightChild(rc2,1)
    features.append(right_rc2)
    right_rc3 = c.getRightChild(rc3,1)
    features.append(right_rc3)

    for word in features:
        features1.append(getWordID(c.getWord(word)))

    for word in features:
        features1.append(getPosID(c.getPOS(word)))

    label_vec = features[6:]
    for word in label_vec:
        features1.append(getLabelID(c.getLabel(word)))

    return features1


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])
            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print(i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
            if(c.tree == trees[i]):
                print("arc standard success")
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print(parsing_system.rootLabel)

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

