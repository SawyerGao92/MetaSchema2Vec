#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Bruce Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from sklearn.decomposition import PCA
plt.style.use('ggplot')



def plot_embedding(embedding):
    name, embedding = zip(*list(embedding.items()))

    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(embedding)

    usr_embedding = list()
    question_embedding = list()

    for n, e in zip(name, pca_embedding):
        if re.match("usr", n):
            usr_embedding.append(e)
        elif re.match("question", n):
            question_embedding.append(e)

    x, y = zip(*pca_embedding)
    # u_x, u_y = zip(*usr_embedding)
    # q_x, q_y = zip(*question_embedding)
    # color = np.arctan2(y, x)
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    '''s: 点大小'''
    ax.scatter(x[:40000], y[:40000], s=10, c='C0', alpha=0.5)
    # l1 = ax.scatter(u_x[:200], u_y[:200], s=30, c='C0', alpha=0.4)
    # l2 = ax.scatter(q_x[:200], q_y[:200], s=30, c='C1', alpha=0.4)
    # ax.set_title('representations ')
    # ax.legend((l1, l2), ('User', 'Question'), scatterpoints=1, loc='upper left', ncol=1, fontsize=13)
    ax.grid(True)

    # 不显示坐标轴的值
    plt.xticks(())
    plt.yticks(())

    plt.show()

def plot_scatter():
    # Load a numpy record array from yahoo csv data with fields date, open, close,
    # volume, adj_close from the mpl-data/example directory. The record array
    # stores the date as an np.datetime64 with a day unit ('D') in the date column.
    with cbook.get_sample_data('goog.npz') as datafile:
        price_data = np.load(datafile)['price_data'].view(np.recarray)
    price_data = price_data[-250:]  # get the most recent 250 trading days

    delta1 = np.diff(price_data.adj_close) / price_data.adj_close[:-1]

    # Marker size in units of points^2
    volume = (15 * price_data.volume[:-2] / price_data.volume[0]) ** 2
    close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

    fig, ax = plt.subplots()
    ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)


    ax.set_title('representations')

    ax.grid(True)
    fig.tight_layout()

    plt.show()

def plot_question_classcification(embedding_dict, top_label_questions, num_nodes=5000):
    name, embedding = zip(*list(embedding_dict.items())[:num_nodes])
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(embedding)
    pca_embedding_dict = dict(zip(name, pca_embedding))

    top_1_q_label = dict()
    for label, questions in top_label_questions.items():
         for q in questions:
             top_1_q_label[q] = label

    label_embedding_dict = dict()
    label_embedding_dict['no label'] = list()
    for name, embedding in pca_embedding_dict.items():
        if re.match("question", name):
            # 根据name找到label
            if name in top_1_q_label:
                label = top_1_q_label[name]
                if label not in label_embedding_dict:
                    label_embedding_dict[label] = list()
                label_embedding_dict[label].append(embedding)
            else:
                label_embedding_dict['no label'].append(embedding)

    color = ['C0', 'C1', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    l = list()
    for i, [name, embeddings] in enumerate(label_embedding_dict.items()):
        x, y = zip(*embeddings)
        if name == 'no label':
            l.append(ax.scatter(x[:], y[:], s=6, c='grey', alpha=0.4))
    id = 0
    for i, [name, embeddings] in enumerate(label_embedding_dict.items()):
        x, y = zip(*embeddings)
        if name != 'no label':
            l.append(ax.scatter(x[:], y[:], s=40, c=color[i], alpha=0.4))
            id += 1

    # 'C#', '.NET', 'Java', 'ASP.NET', 'JavaScript'
    ax.legend(l[1:]+[l[0]], ('C#', '.NET', 'Java', 'asp.net', 'Other labels'), scatterpoints=1, loc='upper left', ncol=1, fontsize=10)
    # ax.grid(True)

    # 不显示坐标轴的值
    plt.xticks(())
    plt.yticks(())

    plt.show()


def plot_f1(f1):
    macro_f1, micro_f1 = zip(*f1)
    x = range(0, len(macro_f1))
    plt.plot(x, macro_f1, color='green', marker='o', label='macro_f1')
    plt.plot(x, micro_f1, color='blue', marker='o', label='micro_f1')
    plt.legend()

    plt.xlabel('iteration times')
    plt.ylabel('f1')
    plt.show()

def plot_line(line):
    x = range(0, len(line))
    plt.plot(x, line, color='red', marker='o')
    plt.legend()

    plt.xlabel('value')
    plt.ylabel('times')
    plt.show()