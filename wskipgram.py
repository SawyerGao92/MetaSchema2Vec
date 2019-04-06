#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Leo Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.

import os.path as path
import numpy as np
import math
import sys
from utils import sigmoid
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import sparse_ops
from sklearn.metrics import f1_score, roc_auc_score
import plot
import tqdm
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class VocabItem(object):
    def __init__(self, name, id, type=None):
        self._name = name
        self._type = type
        self._id = id
        self._code = None  # Huffman encoding

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def code(self):
        return self._code


class Vocab(object):
    def __init__(self, walks_dir):
        vocab_index_by_id = {}
        vocab_index_by_name = {}
        vocab_types = set()
        walks_size = 0  # node num in walks

        with open(walks_dir, 'r') as fi:
            for line in tqdm.tqdm(fi):
                tokens = line.split()
                for token in tokens:
                    if token not in vocab_index_by_name:
                        vocab_index_by_id[len(vocab_index_by_id)] = VocabItem(name=token, id=len(vocab_index_by_id),
                                                                              type=token.split("-")[0])
                        vocab_index_by_name[token] = VocabItem(name=token, id=len(vocab_index_by_name),
                                                               type=token.split("-")[0])
                        vocab_types.add(token.split("-")[0])
                    walks_size += 1

        vocab_nodepair_index_by_name = {}
        vocab_nodepair_index_by_id = {}
        for i in vocab_types:
            for j in vocab_types:
                vocab_nodepair_index_by_name[i + '-' + j] = len(vocab_nodepair_index_by_name)
                vocab_nodepair_index_by_id[len(vocab_nodepair_index_by_id)] = i + '-' + j

        self._vocab_index_by_id = vocab_index_by_id
        self._vocab_index_by_name = vocab_index_by_name
        self._vocab_nodepair_index_by_name = vocab_nodepair_index_by_name
        self._vocab_nodepair_index_by_id = vocab_nodepair_index_by_id
        self._vocab_types = vocab_types
        self._walks_size = walks_size  # node num in walks

        # 从游走文件中节点数量进行排序，小于min_count的节点送入删去，并送入<unk> (unknown)
        # self.__sort(min_count=0)
        time.sleep(0.1)
        print 'Total nodes in random walks: %d' % self.walks_size
        print 'Total unique nodes: %d' % len(vocab_index_by_id)
        print 'Total nodepairs: %d' % len(vocab_nodepair_index_by_id)

    def nodepair_name2id(self, name):
        return self._vocab_nodepair_index_by_name[name]

    def nodepair_id2name(self, id):
        return self._vocab_nodepair_index_by_id[id]

    def node_name2id(self, name):
        if type(name) is str:
            return self._vocab_index_by_name[name].id
        elif type(name) is list:
            return [self._vocab_index_by_name[token].id for token in name]
            # if token in self else self.vocab_hash['<unk>']
        else:
            raise TypeError('name type error!')

    def node_id2name(self, id):
        if type(id) is int:
            return self._vocab_index_by_id[id].name
        elif type(id) is list:
            return [self._vocab_index_by_id[token].name for token in id]
        else:
            raise TypeError('name type error!')

    def __getitem__(self, key):
        if type(key) is int:
            return self._vocab_index_by_id[key]
        elif type(key) is str:
            return self._vocab_index_by_name[key]
        else:
            raise TypeError('Key type error!')

    def __len__(self):
        return len(self._vocab_index_by_id)

    def __iter__(self):
        return iter(self._vocab_index_by_id)

    def __contains__(self, key):
        if type(key) is int:
            return key in self._vocab_index_by_id
        elif type(key) is str:
            return key in self._vocab_index_by_name
        else:
            raise TypeError('Key type error!')

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # 既然删去部分节点，重新排序号
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print '\nUnknown vocab size:', count_unk

    @property
    def nodes_size(self):
        return len(self._vocab_index_by_id)

    @property
    def walks_size(self):
        return self._walks_size

    @walks_size.setter
    def walks_size(self, value):
        self._walk_size = value

    @property
    def nodepairs_size(self):
        return len(self._vocab_nodepair_index_by_id)

    def savetxt(self):
        print 1


class WeightedSkipGram(object):
    # file, min_count, window, skipgram_weight, learning_rate, net_vocab=None, neg_table=None, nn0=None, nn1=None,
    def build(self, embedding_size, num_neg, batch_size=128, vocabulary_size=10000, transmissibility_size=100,
              learning_rate=0.01, classifier_type='node', label_size=None):
        label_size = 1 if classifier_type == 'link' else label_size
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('inputs'):
                nodea = tf.placeholder(tf.int32, shape=[batch_size, ])
                nodeb = tf.placeholder(tf.int32, shape=[batch_size, ])
                linkab_type = tf.placeholder(tf.int32, shape=[batch_size])
                nodea_label = tf.placeholder(tf.float32, shape=[batch_size, label_size])
                self.nodea = nodea
                self.nodeb = nodeb
                self.linkab_type = linkab_type
                self.nodea_label = nodea_label

            with tf.device('/cpu:0'):
                with tf.name_scope('embeddings'):
                    # X in nce
                    embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    nodea_embed = tf.nn.embedding_lookup(embeddings, nodea)
                    nodeb_embed = tf.nn.embedding_lookup(embeddings, nodeb)
                    self.embeddings = embeddings

                with tf.name_scope('weights'):
                    # theta in nce
                    nce_weights = tf.Variable(tf.truncated_normal(
                        [vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
                    # weight in classifier
                    classifier_weights = {
                        'h1': tf.Variable(tf.random_normal([embedding_size, 100])),
                        'h2': tf.Variable(tf.random_normal([100, 100])),
                        'out': tf.Variable(tf.random_normal([100, label_size]))
                    }

                with tf.name_scope('biases'):
                    # theta in nce
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
                    # bias in classifier
                    classifier_biases = {
                        'b1': tf.Variable(tf.random_normal([100])),
                        'b2': tf.Variable(tf.random_normal([100])),
                        'out': tf.Variable(tf.random_normal([label_size]))
                    }

                with tf.name_scope('transmissibility_weights'):
                    # target weight in nce, weights cannot minus 0.
                    transmissibility_weights = tf.Variable(
                        tf.random_uniform([transmissibility_size, 1], 0.0, 1.0),
                        dtype=tf.float32)
                    normalized_transmissibility_weights = tf.add(tf.div(
                        tf.div(
                            tf.subtract(transmissibility_weights, tf.reduce_min(transmissibility_weights)),
                            tf.subtract(tf.reduce_max(transmissibility_weights),
                                        tf.reduce_min(transmissibility_weights))
                        ),
                        tf.constant(2.0)), tf.constant(0.5))
                    # normalized_transmissibility_weights = tf.div(
                    #     tf.subtract(transmissibility_weights, tf.reduce_min(transmissibility_weights)),
                    #     tf.subtract(tf.reduce_max(transmissibility_weights), tf.reduce_min(transmissibility_weights)))
                    transmission = tf.nn.embedding_lookup(normalized_transmissibility_weights, linkab_type)
                    self.transmissibility_weights = normalized_transmissibility_weights

            with tf.name_scope('nce'):
                with tf.name_scope('nce_loss'):
                    nce_loss = tf.reduce_mean(
                        self._weighted_nce_loss(weights=nce_weights,
                                                biases=nce_biases,
                                                inputs=nodea_embed,
                                                labels=nodeb,
                                                transmissibility=transmission,
                                                num_sampled=num_neg,
                                                num_classes=vocabulary_size))
                    tf.summary.scalar('nce_loss', nce_loss)
                with tf.name_scope('nce_optimizer'):
                    var_list = [embeddings, nce_weights, nce_biases]
                    nce_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(nce_loss,
                                                                                                var_list=var_list)
                self.nce_loss = nce_loss
                self.nce_optimizer = nce_optimizer

            if classifier_type is not None or classifier_type is not '':
                print 'Metaschema2vec runnning in supervised mode...'
                with tf.name_scope('classifier'):
                    # hidden1 = tf.nn.relu(tf.matmul(nodea_embed, weights) + biases)
                    with tf.name_scope('classifier_loss'):
                        if classifier_type == 'node':
                            layer_0 = nodea_embed
                        elif classifier_type == 'link':
                            layer_0 = tf.multiply(nodea_embed, nodeb_embed)
                        else:
                            raise TypeError('classifier type error!')

                        layer_1 = tf.nn.relu(
                            tf.add(tf.matmul(layer_0, classifier_weights['h1']), classifier_biases['b1']))
                        layer_2 = tf.nn.relu(
                            tf.add(tf.matmul(layer_1, classifier_weights['h2']), classifier_biases['b2']))
                        logits = tf.matmul(layer_2, classifier_weights['out']) + classifier_biases['out']
                        classifier_prediction = tf.nn.sigmoid(logits)
                        # error = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_layer, labels=nodea_label)
                        # class_weight = tf.stack([tf.constant([50, 50, 50], dtype=tf.float32)] * batch_size)
                        # classifier_loss = tf.reduce_mean(tf.multiply(error, class_weight))
                        classifier_loss = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=nodea_label))

                    tf.summary.scalar('classifier_loss', classifier_loss)
                    with tf.name_scope('classifier_optimizer'):
                        # freeze embeddings untrainable
                        # variables_to_train = tf.trainable_variables()
                        # variables_to_train.remove(embeddings)
                        var_list = [classifier_weights['h1'], classifier_weights['h2'], classifier_weights['out'],
                                    classifier_biases['b1'], classifier_biases['b2'],
                                    classifier_biases['out']]
                        classifier_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
                            classifier_loss,
                            var_list=var_list)
                    self.classifier_loss = classifier_loss
                    self.classifier_prediction = classifier_prediction
                    self.classifier_optimizer = classifier_optimizer

                with tf.name_scope('transmissibility'):
                    with tf.name_scope('transmissibility_optimizer'):
                        var_list = [embeddings]
                        transmissibility_1_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(
                            classifier_loss, var_list=var_list)
                        var_list = [transmissibility_weights]
                        transmissibility_2_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(
                            nce_loss, var_list=var_list)
                    self.transmissibility_1_optimizer = transmissibility_1_optimizer
                    self.transmissibility_2_optimizer = transmissibility_2_optimizer

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()
        return graph, init

    def train(self, walks_dir, vocab, graph, init, window_size, batch_size=128, classifier_data=None, classifier_type=None, log_dir=None):
        tr_data, val_data, tr_label, val_label = classifier_data
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Open a writer to write summaries.
            if log_dir is not None:
                writer = tf.summary.FileWriter(log_dir, session.graph)

            # initialize all variables
            init.run()
            print('Initialized')

            for epoch in range(30):
                # separator
                print '--> epoch: ' + str(epoch) + ' <---------------------------------------------------------------------------------'
                if classifier_type is not None or classifier_type is not '':
                    print '----> train-transmissibility---------'
                    self._train_transmissibility(session, tr_data, tr_label, walks_dir, vocab, window_size, batch_size, classifier_type)
                print '----> train-embedding----------------'
                self._train_embedding(session, walks_dir, vocab, window_size, batch_size)

                if classifier_type is not None or classifier_type is not '':
                    print '----> train-classifier---------------'
                    self._train_classifier(session, tr_data, tr_label, batch_size, classifier_type)
                    print '----> test-classifier----------------'
                    self._test_classifier(session, val_data, val_label, batch_size, classifier_type)

            # export results
            embedding_dict = dict()
            for i, embedding in enumerate(session.run(self.embeddings).tolist()):
                embedding_dict[vocab.node_id2name(id=i)] = embedding
        return embedding_dict

    def _train_classifier(self, session, tr_data, tr_label, batch_size, classifier_type='node'):
        classifier_loss = 0
        train_batch = zip(range(0, len(tr_data), batch_size), range(batch_size, len(tr_data) + 1, batch_size))
        batch_num = 0
        for epoch_c in range(30):
            for j, (start, end) in enumerate(train_batch):
                batch_num += 1
                if classifier_type == 'node':
                    feed_dict = {self.nodea: tr_data[start:end], self.nodea_label: tr_label[start:end]}
                elif classifier_type == 'link':
                    feed_dict = {self.nodea: tr_data[0][start:end], self.nodeb: tr_data[1][start:end], self.nodea_label: tr_label[start:end]}
                _, loss_val = session.run([self.classifier_optimizer, self.classifier_loss],
                                          feed_dict=feed_dict)  # the loss before training
                classifier_loss += loss_val
            sys.stdout.flush()
            sys.stdout.write(
                "\rTraining classifier of epoch %d, %d batch, loss: %f" % (
                    epoch_c, batch_num, classifier_loss))
            classifier_loss = 0
        sys.stdout.write("\n")

    def _test_classifier(self, session, val_data, val_label, batch_size, classifier_type='node'):
        all_prediction = []
        all_label = []
        test_batch = zip(range(0, len(val_label), batch_size), range(batch_size, len(val_label) + 1, batch_size))
        for i, (start, end) in enumerate(test_batch):
            if classifier_type == 'node':
                feed_dict = {self.nodea: val_data[start:end], self.nodea_label: val_label[start:end]}
            elif classifier_type == 'link':
                feed_dict = {self.nodea: val_data[0][start:end], self.nodeb: val_data[1][start:end],
                             self.nodea_label: val_label[start:end]}
            prediction = session.run(self.classifier_prediction, feed_dict=feed_dict)
            all_prediction.append(prediction)
            all_label.append(val_label[start:end])
        all_prediction = np.concatenate(all_prediction, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        # all_prediction_int = np.int64(all_prediction >= 0.5)
        # macro_f1 = f1_score(y_true=all_label, y_pred=all_prediction_int, average='macro')
        # micro_f1 = f1_score(y_true=all_label, y_pred=all_prediction_int, average='micro')
        # f1 = f1_score(y_true=all_label, y_pred=all_prediction_int)
        # print 'macro f1: ' + str(macro_f1) + '; micro f1: ' + str(micro_f1)  # + '; f1: ' + str(f1)
        # print 'positive samples in labels: ' + str(val_label.sum()) + ' ; positive predictions: ' + str(all_prediction.sum())
        print 'auc: ' + str(roc_auc_score(y_true=all_label, y_score=all_prediction))

    def _train_transmissibility(self, session, tr_data, tr_label, walks_dir, vocab, window_size, batch_size, classifier_type='node'):
        # 1. back to embedding
        classifier_loss = 0
        train_batch = zip(range(0, len(tr_label), batch_size), range(batch_size, len(tr_label) + 1, batch_size))
        batch_num = 0
        for epoch_c in range(10):
            for j, (start, end) in enumerate(train_batch):
                batch_num += 1
                if classifier_type == 'node':
                    feed_dict = {self.nodea: tr_data[start:end], self.nodea_label: tr_label[start:end]}
                elif classifier_type == 'link':
                    feed_dict = {self.nodea: tr_data[0][start:end], self.nodeb: tr_data[1][start:end], self.nodea_label: tr_label[start:end]}
                _, loss_val = session.run([self.transmissibility_1_optimizer, self.classifier_loss],
                                          feed_dict=feed_dict)  # the loss before training
                classifier_loss += loss_val
            sys.stdout.flush()
            sys.stdout.write(
                "\rTraining 1 transmissibility of epoch %d, %d batch, loss: %f" % (
                    epoch_c, batch_num, classifier_loss))
            classifier_loss = 0
        sys.stdout.write("\n")

        # 2. embedding to transmissibility
        breaka = False
        batch_embedding_nodea = []
        batch_embedding_nodeb = []
        batch_embedding_linktype = []
        average_loss = 0
        for epoch_t in range(1):
            window_position = 0
            with open(walks_dir) as f:
                for line in f:
                    sent = vocab.node_name2id(line.strip().split())
                    sent_types = [li.split('-')[0] for li in line.strip().split()]
                    for sent_pos, token in enumerate(sent):
                        token_type = sent_types[sent_pos]

                        current_win = np.random.randint(low=1, high=window_size + 1)
                        context_start = max(sent_pos - current_win, 0)
                        context_end = min(sent_pos + current_win + 1, len(sent))
                        context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]
                        context_type = sent_types[context_start:sent_pos] + sent_types[
                                                                            sent_pos + 1: context_end]

                        for context_node, context_node_type in zip(context, context_type):
                            batch_embedding_nodea.append(context_node)
                            batch_embedding_nodeb.append(token)
                            batch_embedding_linktype.append(
                                vocab.nodepair_name2id(context_node_type + '-' + token_type))

                            if len(batch_embedding_nodea) == batch_size:
                                feed_dict = {self.nodea: batch_embedding_nodea,
                                             self.nodeb: batch_embedding_nodeb,
                                             self.linkab_type: batch_embedding_linktype}
                                _, loss_val, la, lo = session.run(
                                    [self.transmissibility_2_optimizer, self.nce_loss, self.labels, self.logits],
                                    feed_dict=feed_dict)
                                if loss_val < 0:
                                    print ' '
                                average_loss += loss_val
                                batch_embedding_nodea = []
                                batch_embedding_nodeb = []
                                batch_embedding_linktype = []

                        window_position += 1
                        if window_position % 10000 == 0:
                            sys.stdout.flush()
                            sys.stdout.write(
                                "\rTraining 2 transmissibility of epoch %d: %d of %d walking nodes, loss: %f" % (
                                    epoch_t, window_position, vocab.walks_size, average_loss))
                            average_loss = 0
                    #     if window_position == 10000:
                    #         breaka = True
                    #         break
                    # if breaka is True:
                    #     break
        sys.stdout.write("\n")
        print 'current transmissibility weight is :' + str([vocab.nodepair_id2name(i) for i in range(vocab.nodepairs_size)])
        print session.run(self.transmissibility_weights).tolist()

    def _train_embedding(self, session, walks_dir, vocab, window_size, batch_size):
        breaka = False
        average_loss = 0
        batch_embedding_nodea = []
        batch_embedding_nodeb = []
        batch_embedding_linktype = []
        for epoch_e in range(10):
            window_position = 0
            with open(walks_dir) as f:
                for line in f:
                    sent = vocab.node_name2id(line.strip().split())
                    sent_types = [li.split('-')[0] for li in line.strip().split()]
                    for sent_pos, token in enumerate(sent):
                        token_type = sent_types[sent_pos]

                        current_win = np.random.randint(low=1, high=window_size + 1)
                        context_start = max(sent_pos - current_win, 0)
                        context_end = min(sent_pos + current_win + 1, len(sent))
                        context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]
                        context_type = sent_types[context_start:sent_pos] + sent_types[
                                                                            sent_pos + 1: context_end]

                        for context_node, context_node_type in zip(context, context_type):
                            batch_embedding_nodea.append(context_node)
                            batch_embedding_nodeb.append(token)
                            batch_embedding_linktype.append(
                                vocab.nodepair_name2id(context_node_type + '-' + token_type))

                            if len(batch_embedding_nodea) == batch_size:
                                feed_dict = {self.nodea: batch_embedding_nodea,
                                             self.nodeb: batch_embedding_nodeb,
                                             self.linkab_type: batch_embedding_linktype}
                                _, loss_val = session.run([self.nce_optimizer, self.nce_loss],
                                                          feed_dict=feed_dict)
                                average_loss += loss_val
                                batch_embedding_nodea = []
                                batch_embedding_nodeb = []
                                batch_embedding_linktype = []

                        window_position += 1
                        if window_position % 10000 == 0:
                            sys.stdout.flush()
                            sys.stdout.write(
                                "\rTraining embedding of epoch %d: %d of %d walking nodes, loss: %f" % (
                                    epoch_e, window_position, vocab.walks_size, average_loss))
                            average_loss = 0
                    #     if window_position == 10000:
                    #         breaka = True
                    #         break
                    # if breaka is True:
                    #     break
        sys.stdout.write("\n")

    def _weighted_nce_loss(self,
                           weights,
                           biases,
                           labels,
                           inputs,
                           num_sampled,
                           num_classes,
                           transmissibility=None,
                           num_true=1,
                           sampled_values=None,
                           remove_accidental_hits=False,
                           partition_strategy="mod",
                           name="nce_loss"):
        logits, labels = self._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
            num_sampled=num_sampled,
            num_classes=num_classes,
            transmissibility=transmissibility,
            num_true=num_true,
            sampled_values=sampled_values,
            subtract_log_q=True,
            remove_accidental_hits=remove_accidental_hits,
            partition_strategy=partition_strategy,
            name=name)
        self.labels = labels
        self.logits = logits
        sampled_losses = self.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, name="sampled_losses")
        # sampled_losses is batch_size x {true_loss, sampled_losses...}
        # We sum out true and sampled losses.
        return self._sum_rows(sampled_losses)

    def _compute_sampled_logits(self, weights, biases, labels, inputs, num_sampled, num_classes, transmissibility,
                                num_true=1,
                                sampled_values=None, subtract_log_q=True, remove_accidental_hits=False,
                                partition_strategy="mod", name=None, seed=None):
        if isinstance(weights, variables.PartitionedVariable):
            weights = list(weights)
        if not isinstance(weights, list):
            weights = [weights]

        with ops.name_scope(name, "compute_sampled_logits",
                            weights + [biases, inputs, labels]):
            if labels.dtype != dtypes.int64:
                labels = math_ops.cast(labels, dtypes.int64)
            if labels.shape.ndims == 1:
                labels = array_ops.expand_dims(labels, -1)
            labels_flat = array_ops.reshape(labels, [-1])

            # Sample the negative labels.
            #   sampled shape: [num_sampled] tensor
            #   true_expected_count shape = [batch_size, 1] tensor
            #   sampled_expected_count shape = [num_sampled] tensor
            #   num_sampled 字典大小
            if sampled_values is None:
                sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
                    true_classes=labels,
                    num_true=num_true,
                    num_sampled=num_sampled,
                    unique=True,
                    range_max=num_classes,
                    seed=seed)
            # NOTE: pylint cannot tell that 'sampled_values' is a sequence
            # pylint: disable=unpacking-non-sequence
            sampled, true_expected_count, sampled_expected_count = (
                array_ops.stop_gradient(s) for s in sampled_values)
            # pylint: enable=unpacking-non-sequence
            sampled = math_ops.cast(sampled, dtypes.int64)

            # labels_flat is a [batch_size * num_true] tensor
            # sampled is a [num_sampled] int tensor
            all_ids = array_ops.concat([labels_flat, sampled], 0)

            # Retrieve the true weights and the logits of the sampled weights.

            # weights shape is [num_classes, dim]
            # 128个相似节点对和 5个非相似节点(也就是128*5个非相似节点对)
            all_w = embedding_ops.embedding_lookup(
                weights, all_ids, partition_strategy=partition_strategy)

            # true_w shape is [batch_size * num_true, dim] - > 128 * 100
            true_w = array_ops.slice(all_w, [0, 0],
                                     array_ops.stack(
                                         [array_ops.shape(labels_flat)[0], -1]))
            # 5 * 100
            sampled_w = array_ops.slice(
                all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])

            # inputs has shape [batch_size, dim]
            # sampled_w has shape [num_sampled, dim]
            # Apply X*W', which yields [batch_size, num_sampled]
            # 128个输入节点分别和这5个非相似节点,进行比较, 128 * 5, 表示节点a和节点b的相似度.
            sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

            # Retrieve the true and sampled biases, compute the true logits, and
            # add the biases to the true and sampled logits.
            all_b = embedding_ops.embedding_lookup(
                biases, all_ids, partition_strategy=partition_strategy)
            # true_b is a [batch_size * num_true] tensor
            # sampled_b is a [num_sampled] float tensor
            true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
            sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

            # inputs shape is [batch_size, dim]
            # true_w shape is [batch_size * num_true, dim]
            # row_wise_dots is [batch_size, num_true, dim]
            dim = array_ops.shape(true_w)[1:2]
            new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
            row_wise_dots = math_ops.multiply(
                array_ops.expand_dims(inputs, 1),
                array_ops.reshape(true_w, new_true_w_shape))
            # We want the row-wise dot plus biases which yields a
            # [batch_size, num_true] tensor of true_logits.
            dots_as_matrix = array_ops.reshape(row_wise_dots,
                                               array_ops.concat([[-1], dim], 0))
            true_logits = array_ops.reshape(self._sum_rows(dots_as_matrix), [-1, num_true])
            true_b = array_ops.reshape(true_b, [-1, num_true])
            # 相似节点对,对比结果是128*1;非相似节点对,对比结果是128*5
            true_logits += true_b
            sampled_logits += sampled_b

            if remove_accidental_hits:
                acc_hits = candidate_sampling_ops.compute_accidental_hits(
                    labels, sampled, num_true=num_true)
                acc_indices, acc_ids, acc_weights = acc_hits

                # This is how SparseToDense expects the indices.
                acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
                acc_ids_2d_int32 = array_ops.reshape(
                    math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
                sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                                  "sparse_indices")
                # Create sampled_logits_shape = [batch_size, num_sampled]
                sampled_logits_shape = array_ops.concat(
                    [array_ops.shape(labels)[:1],
                     array_ops.expand_dims(num_sampled, 0)], 0)
                if sampled_logits.dtype != acc_weights.dtype:
                    acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
                sampled_logits += sparse_ops.sparse_to_dense(
                    sparse_indices,
                    sampled_logits_shape,
                    acc_weights,
                    default_value=0.0,
                    validate_indices=False)

            if subtract_log_q:
                # Subtract log of Q(l), prior probability that l appears in sampled.
                true_logits -= math_ops.log(true_expected_count)
                sampled_logits -= math_ops.log(sampled_expected_count)

            # Construct output logits and labels. The true labels/logits start at col 0.
            out_logits = array_ops.concat([true_logits, sampled_logits], 1)

            # true_logits is a float tensor, ones_like(true_logits) is a float
            # tensor of ones. We then divide by num_true to ensure the per-example
            # labels sum to 1.0, i.e. form a proper probability distribution.
            out_labels = array_ops.concat([
                transmissibility,  # array_ops.ones_like(true_logits) / num_true,  #
                array_ops.zeros_like(sampled_logits)
            ], 1)

            return out_logits, out_labels

    def _sum_rows(self, x):
        """Returns a vector summing up each row of the matrix x."""
        # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
        # a matrix.  The gradient of _sum_rows(x) is more efficient than
        # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
        # we use _sum_rows(x) in the nce_loss() computation since the loss
        # is mostly used for training.
        cols = array_ops.shape(x)[1]
        ones_shape = array_ops.stack([cols, 1])
        ones = array_ops.ones(ones_shape, x.dtype)
        return array_ops.reshape(math_ops.matmul(x, ones), [-1])

    def sigmoid_cross_entropy_with_logits(self,  # pylint: disable=invalid-name
                                          _sentinel=None,
                                          labels=None,
                                          logits=None,
                                          name=None):
        """Computes sigmoid cross entropy given `logits`.

        Measures the probability error in discrete classification tasks in which each
        class is independent and not mutually exclusive.  For instance, one could
        perform multilabel classification where a picture can contain both an elephant
        and a dog at the same time.

        For brevity, let `x = logits`, `z = labels`.  The logistic loss is

              z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
            = (1 - z) * x + log(1 + exp(-x))
            = x - x * z + log(1 + exp(-x))

        For x < 0, to avoid overflow in exp(-x), we reformulate the above

              x - x * z + log(1 + exp(-x))
            = log(exp(x)) - x * z + log(1 + exp(-x))
            = - x * z + log(1 + exp(x))

        Hence, to ensure stability and avoid overflow, the implementation uses this
        equivalent formulation

            max(x, 0) - x * z + log(1 + exp(-abs(x)))

        `logits` and `labels` must have the same type and shape.

        Args:
          _sentinel: Used to prevent positional parameters. Internal, do not use.
          labels: A `Tensor` of the same type and shape as `logits`.
          logits: A `Tensor` of type `float32` or `float64`.
          name: A name for the operation (optional).

        Returns:
          A `Tensor` of the same shape as `logits` with the componentwise
          logistic losses.

        Raises:
          ValueError: If `logits` and `labels` do not have the same shape.
        """
        # pylint: disable=protected-access
        nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                                 labels, logits)
        # pylint: enable=protected-access

        with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
            logits = ops.convert_to_tensor(logits, name="logits")
            labels = ops.convert_to_tensor(labels, name="labels")
            try:
                labels.get_shape().merge_with(logits.get_shape())
            except ValueError:
                raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                                 (logits.get_shape(), labels.get_shape()))

            # The logistic loss formula from above is
            #   x - x * z + log(1 + exp(-x))
            # For x < 0, a more numerically stable formula is
            #   -x * z + log(1 + exp(x))
            # Note that these two expressions can be combined into the following:
            #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
            # To allow computing gradients at zero, we define custom versions of max and
            # abs functions.
            zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
            cond = (logits >= zeros)
            relu_logits = array_ops.where(cond, logits, zeros)
            neg_abs_logits = array_ops.where(cond, -logits, logits)
            return math_ops.add(
                relu_logits - logits * labels,
                math_ops.log1p(math_ops.exp(neg_abs_logits)),
                name=name)
