#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Leo Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.

import codecs
import os
import re
import sys
import math
import xml.sax
import numpy as np
import networkx as nx
import time, datetime
import nltk
from keras.utils import to_categorical
from itertools import groupby
from sklearn.metrics import f1_score

reload(sys)
sys.setdefaultencoding('utf8')


# class NodeID():
#     def __init__(self, name, type, id=None):
#         self.name = name
#         self.type = type
#         self.id = id
#         self.quantity = 1
#         self.time = None
#
#
# class NetVocab():
#     def __init__(self):
#         # index the informtion based on their real name.
#         self.vocab = dict()
#
#     def add(self, NodeID):
#         if NodeID.name in self.vocab:
#             self.vocab[NodeID.name].quantity += 1
#         else:
#             self.vocab[NodeID.name] = NodeID
#
#     def generate_id(self):
#         i = 0
#         for name in self.vocab.keys():
#             self.vocab[name].id = i
#             i += 1
#
#     def filter_for_rare(self, min_count):
#         for name, nodeid in self.vocab.items():
#             if nodeid.quantity < min_count:
#                 del self.vocab[name]
#
#     def __getitem__(self, idorname):
#         return self.vocab[idorname]
#
#     def __len__(self):
#         return len(self.vocab)
#
#     def __iter__(self):
#         return iter(self.vocab.items())
#
#     def __contains__(self, key):
#         return key in self.vocab
#
#     def indices(self, tokens):
#         return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]


class Dataflow(object):
    """
    将网络导入networkx
    """

    def __init__(self, rawfilename=None):
        self.netname = None
        self.adj = list()
        self.label = dict()
        self.vocab = dict()
        self.netx = nx.MultiGraph()
        self.walks = list()
        self.embeddings = dict()

    def read_and_process_raw(self, filename):
        raise NotImplementedError

    def read_adj(self, dirfile):
        adj = list()
        with open(dirfile, 'r') as f:
            for line in f:
                adj.append(line.strip().split(' '))
        self.adj = adj

    def read_vocab(self, dir):
        vocab = dict()
        with codecs.open(dir, 'r', encoding='utf-8') as f:
            cur_meta = None
            for line in f:
                if re.match("---", line):
                    cur_meta = line[re.match("---:", line).end():].strip()
                    vocab[cur_meta] = NetVocab()
                else:
                    spline = line.strip().split(',')
                    vocab[cur_meta].add(NodeID(spline[0], spline[2], spline[1]))
        self.vocab = vocab

    def read_walks(self, file):
        walks = list()
        with open(file, 'r') as f:
            for line in f:
                spline = line.strip().split(' ')
                walks.append(spline)
        self.walks = walks

    def read_embedding(self, file):
        embedding = dict()
        with open(file, 'r') as f:
            for line in f:
                spline = line.split(' ')
                embedding[spline[0]] = map(float, spline[1:])
        self.embeddings = embedding

    def read_netx(self):
        """
        import adj to networkx
        :return:
        """
        for line in self.adj:
            self.netx.add_node(line[0], type=line[0].split("-")[0])
            self.netx.add_node(line[1], type=line[1].split("-")[0])
            self.netx.add_edge(line[0], line[1], key=line[2])

    def get_classfier_data(self, classifier_type, vocab):
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.model_selection import train_test_split
        import random

        if classifier_type == 'node':
            question_nodes = [node for (node, t) in data.netx.nodes(data='type') if t == 'question']
            topk_label = self.get_top_k(self.label, top_k=7)

            batch_classifier_node = []
            batch_classifier_nodelabel = []
            for n in question_nodes:
                try:
                    batch_classifier_node.append(vocab.node_name2id(n))
                    batch_classifier_nodelabel.append(topk_label[n])
                except:
                    pass
            batch_classifier_nodelabel = MultiLabelBinarizer().fit_transform(batch_classifier_nodelabel)
            classifier_data = train_test_split(batch_classifier_node, batch_classifier_nodelabel, test_size=0.2,
                                               random_state=1)
        elif classifier_type == 'link':
            author_nodes = [node for (node, t) in self.netx.nodes(data='type') if t == 'author']
            venue_nodes = [venue for (venue, t) in self.netx.nodes(data='type') if t == 'venue']
            labels_set = set([(i[0], i[1]) for i in self.label])

            batch_classifier_node = []
            batch_classifier_nodelabel = []
            for a in author_nodes:
                for v in venue_nodes:
                    try:
                        if (a, v) not in labels_set:
                            if random.randint(0, 10) <= 6:
                                continue
                        batch_classifier_node.append((vocab.node_name2id(a), vocab.node_name2id(v)))
                        batch_classifier_nodelabel.append([int((a, v) in labels_set)])
                    except:
                        pass
            tr_data, val_data, tr_label, val_label = train_test_split(batch_classifier_node, batch_classifier_nodelabel, test_size=0.2,
                                               random_state=1)
            classifier_data = [zip(*tr_data), zip(*val_data), tr_label, val_label]
        return classifier_data

    def mkdir(self, dirname):
        '''如果文件夹不存在，则生成'''
        dirname = os.path.dirname(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print 'Dir %s --> create success' % dirname
        print 'Writing to %s' % dirname

    def write_adj(self, dir):
        """adj输出"""
        self.mkdir(dir)
        with open(dir, 'a') as f:
            for line in self.adj:
                for item in line:
                    f.write(str(item))
                    f.write(" ")
                f.write("\n")

    def write_label(self, filedir):
        """label 输出"""
        if self.label is None:
            return
        self.mkdir(filedir)
        if type(self.label) == dict:
            with open(filedir, 'a') as f:
                for question, tags in self.label.items():
                    f.write(str(question))
                    f.write(":")
                    for tag in tags:
                        f.write(str(tag))
                        f.write(" ")
                    f.write("\n")
        elif type(self.label) == list:
            with open(filedir, 'a') as f:
                for line in self.label:
                    for item in line:
                        f.write(str(item))
                        f.write(" ")
                    f.write("\n")
        else:
            raise NotImplementedError('This type of label is not supported. ')

    def read_label(self, filedir, container='dict'):
        if container == 'dict':
            label = dict()
            with open(filedir, 'r') as f:
                for line in f:
                    spline = line.split(':')
                    label[spline[0]] = set(spline[1].strip().split(' '))
            self.label = label
        elif container == 'list':
            label = list()
            with open(filedir, 'r') as f:
                for line in f:
                    spline = line.strip().split(' ')
                    spline[-1] = int(spline[-1])
                    label.append(spline)
            self.label = label
        else:
            raise NotImplementedError('This type of container is not supported. ')

    def label_reverse(self, q_label):
        """
        reserve question-label to label-question
        """
        label_q = dict()
        for q, labels in q_label.items():
            for label in labels:
                if label not in label_q:
                    label_q[label] = set()
                label_q[label].add(q)
        return label_q

    def get_top_k(self, label, top_k):
        """
        : turn label->question and question->label to the top n label
        :param label_sourcedata:
        :param top_num:
        :return:
        """
        q_label = label
        label_q = self.label_reverse(label)
        label_and_qnum = [(l, len(list(q))) for l, q in label_q.items()]
        top_k_label, _ = zip(*sorted(label_and_qnum, reverse=True, key=lambda x: x[1])[:top_k])
        for l, q in label_q.items():
            if l not in top_k_label:
                label_q.pop(l)
        for q, l in q_label.items():
            q_label[q] = filter(lambda x: x in top_k_label, l)
        return q_label

    def write_vocab(self, dir):
        self.mkdir(dir)
        with codecs.open(dir, 'a', encoding='utf-8') as f:
            for t, v in self.vocab.items():
                f.write('---:' + str(t) + '\n')
                for _, nodeid in v:
                    line = str(nodeid.name) + "," + str(nodeid.id) + "," + str(nodeid.type) + "\n"
                    f.write(line)

    def write_walks(self, dir):
        sys.stdout.flush()
        self.mkdir(dir)
        with open(dir, 'a') as f:
            for walk in self.walks:
                f.write(" ".join(walk))
                f.write("\n")

    def write_embedding(self, dir, embeddings):
        self.mkdir(dir)
        with open(dir, 'w+') as f:
            for token, vector in embeddings.items():
                word = token.replace(' ', '_')
                vector_str = ' '.join([str(s) for s in vector])
                f.write('%s %s\n' % (word, vector_str))

    @property
    def nodes_info(self):
        # netx nodes
        types_in_netx = [[n, t] for n, t in self.netx.nodes(data='type')]
        types_in_netx.sort(key=lambda x: x[1])
        types_in_netx = groupby(types_in_netx, key=lambda x: x[1])

        types_num = dict()
        for key, group in types_in_netx:
            types_num[key] = len(list(group))
        return types_num

    @property
    def edges_info(self):
        # netx edges
        types_in_netx = [[u, v, t] for u, v, t in self.netx.edges(keys=True)]
        types_in_netx.sort(key=lambda x: x[2])
        types_in_netx = groupby(types_in_netx, key=lambda x: x[2])

        types_num = dict()
        for key, group in types_in_netx:
            types_num[key] = len(list(group))
        return types_num

class Dataflow_StackOverflow(Dataflow):
    def __init__(self):
        Dataflow.__init__(self)
        # self.vocab = {'usr': NetVocab(), 'question': NetVocab(), 'tag': NetVocab()}
        self.netname = 'StackOverflow'
        self.netname_extension = '(small).csv'
        self.metaschema = {'question': {('ask', 'usr'): 0.5, ('answer', 'usr'): 0.5},
                           'usr': {('ask', 'question'): 0.7, ('answer', 'question'): 0.3}}
        self.tag_usage = 'label'  # tag as label or adj?
        self.LABEL_CONTAINER = 'dict'

    def read_and_process_raw(self, filename):
        """清理raw数据
        清理raw数据,将处理后数据放到bclean中.
        tag: adj / label, tag可以作为label也可以作为adj
        """
        adj_usr_q = set()
        assert self.tag_usage == 'adj' or self.tag_usage == 'label'
        adj_q_tags = set() if self.tag_usage == 'adj' else dict()

        with open(filename) as f:
            next(f)
            for line in f:
                spline = line.replace('\"', '').split(',')
                if spline[2] == 'NA' or spline[-3] == 'NA':
                    continue

                ##########
                q = int(spline[1])  # 问题
                ask_usr = int(spline[2])  # 提问人
                q_time = datetime.datetime.utcfromtimestamp(int(spline[4])).strftime("%Y-%m-%d")  # 提问时间
                ans_usr = int(spline[-3])  # 回答人
                tags = spline[5: -6]  # 问题标签
                ans_time = datetime.datetime.utcfromtimestamp(int(spline[-1])).strftime("%Y-%m-%d")  # 回答时间
                ##########

                # 添加以name为主打的邻接表
                adj_usr_q.add(('usr' + '-' + str(ask_usr), 'question' + '-' + str(q), "ask", q_time))
                adj_usr_q.add(('usr' + '-' + str(ans_usr), 'question' + '-' + str(q), "answer", ans_time))

                if self.tag_usage == 'adj':
                    for tag in tags:
                        adj_q_tags.add(
                            ('question' + '-' + str(q), 'tag' + '-' + str(tag).replace(' ', '_'), 'belong', q_time))
                elif self.tag_usage == 'label':
                    for tag in tags:
                        if 'question' + '-' + str(q) not in adj_q_tags:
                            adj_q_tags['question' + '-' + str(q)] = set()
                        adj_q_tags['question' + '-' + str(q)].add('tag' + '-' + str(tag).replace(' ', '_'))

        # 输出结果
        if self.tag_usage == 'adj':
            self.adj = sorted(list(adj_usr_q) + list(adj_q_tags), key=lambda x: (x[-1]))
            self.label = None
        elif self.tag_usage == 'label':
            self.adj = adj_usr_q
            self.label = adj_q_tags
        return True


class Dataflow_DBLP(Dataflow):
    def __init__(self):
        Dataflow.__init__(self)
        self.netname = 'DBLP'
        self.netname_extension = '.xml'
        self.metaschema = {'paper': {('cite', 'paper'): 0.25, ('include', 'term'): 0.25, ('write', 'author'): 0.25,
                                     ('publish', 'venue'): 0.25},
                           'term': {('include', 'paper'): 1},
                           'author': {('write', 'paper'): 1},
                           'venue': {('publish', 'paper'): 1}}
        self.POINT = 2010
        self.GEL_LABEL = 'yes' if type(self.POINT) == int else 'no'  # split adj as label, yes or no?
        self.LABEL_CONTAINER = 'list'

    def read_and_process_raw(self, filename):
        '''目的是得到adj vocab label'''
        # 创建一个新的解析器对象并返回
        parser = xml.sax.make_parser()
        # 关闭命名空间
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)

        # 提取点
        handler = DblpNodeHandler()
        parser.setContentHandler(handler)
        # 解析XML文档,得到节点表格
        parser.parse(filename)
        self.adj = handler.adj
        self.label = handler.label


class DblpNodeHandler(Dataflow_DBLP, xml.sax.ContentHandler):
    def __init__(self):
        Dataflow_DBLP.__init__(self)
        self.level = 0
        self.adj = set()
        self.label = set()
        self.progressBar = 0
        self.PaperTag = {"article", "inproceedings"}
        self.VenueTag = {'tods', 'tois', 'tkde', 'vldbj', 'tkdd', 'aei', 'dke', 'dmkd', 'ejis', 'geoinformatica', 'ipm',
                         'is', 'isci', 'jasist', 'jws', 'kis', 'tweb', 'dpd', 'iam', 'ipl', 'ir', 'ijcis', 'ijgis',
                         'ijis', 'ijkm', 'ijswis', 'jcis', 'jdm', 'jgitm', 'jiis', 'jsis', 'sigmod', 'sigkdd', 'sigir',
                         'vldb', 'icde', 'cikm', 'pods', 'dasfaa', 'ecml-pkdd', 'iswc', 'icdm', 'icdt', 'edbt', 'cidr',
                         'sdm', 'wsdm', 'dexa', 'ecir', 'webdb', 'er', 'mdm', 'ssdbm', 'waim', 'sstd', 'pakdd', 'apweb',
                         'wise', 'eswc'}

    def startDocument(self):
        print 'In Xml File Parsing!'

    def endDocument(self):
        self.adj = sorted(list(self.adj), key=lambda x: (x[-1]))
        self.label = sorted(list(self.label), key=lambda x: (x[-1]))
        sys.stdout.write("\n")

    def startElement(self, name, attrs):
        self.level += 1
        self.ERContent = ''
        if self.level == 2:
            keys = attrs["key"].split("/")
            if name not in self.PaperTag or keys[0] == 'dblpnote' or keys[1] not in self.VenueTag:
                self.threeLevelSwitch = False
                return
            _, venue, author = keys
            self.threeLevelSwitch = True
            self.paper = attrs["key"].replace(' ', '_')
            self.venue = venue.replace(' ', '_')
            self.authors = set()
            self.terms = set()
            self.time = None
            self.cites = set()
            # progressBar
            self.progressBar += 1
            if self.progressBar % 100 == 0:
                sys.stdout.write("\r>>> iter:" + str(self.progressBar))
                sys.stdout.flush()

    def endElement(self, name):
        # 三级菜单
        if self.level == 3:
            if self.threeLevelSwitch == True and not self.ERContent == '...':
                if name == 'author':
                    self.authors.add(self.ERContent.replace(' ', '_'))
                elif name == 'title':
                    try:
                        sentence = nltk.word_tokenize(self.ERContent)
                    except LookupError:
                        print "Fail to find nltk modules, downloading..."
                        nltk.download('punkt')
                        nltk.download('averaged_perceptron_tagger')
                        sentence = nltk.word_tokenize(self.ERContent)
                    sentence = nltk.pos_tag(sentence)
                    grammar = "NP: {<NNP><NNP>}"
                    cp = nltk.RegexpParser(grammar)
                    result = cp.parse(sentence)
                    keywords = [list(i)[0][0] + ' ' + list(i)[1][0] for i in result if type(i) == nltk.tree.Tree]
                    self.terms = set(map(lambda x: x.replace(' ', '_'), keywords))
                elif name == 'year':
                    self.time = int(self.ERContent)
                elif name == 'cite':
                    self.cites.add(self.ERContent)
        # the end of 2nd menu.
        if self.level == 2:
            if self.threeLevelSwitch:
                adj_temp = set()

                if self.GEL_LABEL == 'no' or (self.GEL_LABEL == 'yes' and self.time <= self.POINT):
                    adj_temp.add(('paper-' + self.paper, 'venue-' + self.venue, "publish", self.time))
                    [adj_temp.add(('paper-' + self.paper, 'paper-' + ci, "cite", self.time)) for ci in self.cites]
                    [adj_temp.add(('paper-' + self.paper, 'term-' + te, "include", self.time)) for te in self.terms]
                    [adj_temp.add(('author-' + au.replace(' ', '_'), 'paper-' + self.paper, "write", self.time)) for au
                     in self.authors]
                    self.adj = self.adj | adj_temp
                elif self.GEL_LABEL == 'yes' and self.time > self.POINT:
                    [adj_temp.add(('author-' + au.replace(' ', '_'), 'venue-' + self.venue, "submit", self.time)) for au
                     in self.authors]
                    self.label = self.label | adj_temp
        self.level -= 1

    def characters(self, content):
        self.ERContent += content
        return


if __name__ == "__main__":
    so = Dataflow_StackOverflow()
