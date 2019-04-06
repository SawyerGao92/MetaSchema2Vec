#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Leo Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.
import os.path as path
import argparse


class FileSettings(object):
    def __init__(self, dataname, dataname_extension):
        # prefix = path.splitext(dataname)[0]
        self.FOLDER = FOLDER = ['aRaw', 'bAdj', 'cWalk', 'dEmbedding']

        self.PATH = PATH = {'raw': path.join(FOLDER[0], dataname, dataname + dataname_extension),
                            'adj': path.join(FOLDER[1], dataname, dataname + '.adj'),
                            'label': path.join(FOLDER[1], dataname, dataname + '.label'),
                            'walk': path.join(FOLDER[2], dataname, dataname + '.walk'),
                            'embedding': path.join(FOLDER[3], dataname, dataname + '.embedding')
                            }
        self.GROUP = GROUP = [['raw'], ['adj', 'label'], ['walk'], ['embedding']]

        self.GROUP_EXIST = [[path.exists(PATH['raw'])],
                            [path.exists(PATH['adj']), path.exists(PATH['label'])],
                            [path.exists(PATH['walk'])],
                            [path.exists(PATH['embedding'])]
                            ]


def parse_args():
    parser = argparse.ArgumentParser(description="Meta.Schema.2.Vec")
    # global parameter
    parser.add_argument('-d', '--data', help='Data abbreviation', default='DBLP',
                        choices=['StackOverflow', 'DBLP'], dest='d')
    parser.add_argument('-s', '--start-position', help='The start position in the flow', default=2, type=int, dest='s')
    # random walk parameter
    parser.add_argument('--number_walks', help='number of walks per node', default=10, type=int)
    parser.add_argument('--walk_length', help='walk length per node', default=50, type=int)
    parser.add_argument('--trans_prob', help='walk length per node', choices=['average', 'invnodes', 'invedges'],
                        default='average', type=str)

    # representation learning parameter
    parser.add_argument('-i', '--embedding_size', help='Dimensionality of node embeddings', default=100, type=int,
                        dest='i')
    parser.add_argument('-b', '--batch_size', help='batch size', default=128, type=int, dest='b')
    parser.add_argument('-w', '--window', help='Max window length', default=5, type=int, dest='w')
    parser.add_argument('-n', '--num_neg', help='Number of negative examples', default=5, type=int, dest='n')
    parser.add_argument('-l', '--learning_rate', help='learning rate', default=0.01, type=float, dest='l')
    parser.add_argument('-aw', '--auto_weight', help='supervised learning weight while learning embedding',
                        default=True, type=bool, dest='aw')
    parser.add_argument('-c', '--classifier_type', help='Node classfication/Link prediction', default='link',
                        choices=['node', 'link'], dest='c')
    args = parser.parse_args()
    return args
