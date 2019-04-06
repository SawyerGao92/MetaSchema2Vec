#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Leo Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.
import dataflow
import sys
import time
from settings import parse_args, FileSettings
from utils import checkLocalFiles
from randomwalk import MetaSchemaRandomWalk
from wskipgram import WeightedSkipGram, Vocab
from plot import plot_embedding, plot_question_classcification, plot_f1, plot_line


def main(args):
    print("Fetching Data: %s" % args.d)
    # init dataflow
    if args.d == "StackOverflow":
        data = dataflow.Dataflow_StackOverflow()
    elif args.d == "DBLP":
        data = dataflow.Dataflow_DBLP()
    else:
        sys.exit("Can't process this type of data, you need increase support in dataflow.")
    # init file settings
    fs = FileSettings(data.netname, data.netname_extension)
    # check loacal files
    checkLocalFiles(args.s, fs)

    # open main process
    # 1.raw->adj(adj, vocab, label, adj_for_gem)
    if args.s == 1:
        print '*******************************************\n' \
              'Start the 1st contact!'
        data.read_and_process_raw(fs.PATH['raw'])
        data.write_adj(fs.PATH['adj'])
        data.write_label(fs.PATH['label'])
        exit()

    # 2.adj->walk
    if args.s <= 2:
        if args.s == 2:
            data.read_adj(fs.PATH['adj'])
            data.read_label(fs.PATH['label'], data.LABEL_CONTAINER)
        print '*******************************************\n' \
              'Enter the 2nd contact!'
        data.read_netx()
        print data.nodes_info
        print data.edges_info
        data.walks = MetaSchemaRandomWalk(data.netx, data.metaschema, 'average', args.number_walks, args.walk_length,
                                          data.nodes_info, data.edges_info)
        data.write_walks(fs.PATH['walk'])
        exit()

    # 3.walk->embedding + classification
    if args.s <= 3:
        if args.s == 3:
            data.read_adj(fs.PATH['adj'])
            data.read_netx()
            data.read_label(fs.PATH['label'], data.LABEL_CONTAINER)
        print '*******************************************\n' \
              'Enter the 3rd contact!'

        # start training
        sg = WeightedSkipGram()
        vocab = Vocab(walks_dir=fs.PATH['walk'])  # 需要先构建一个vocab, 因为建模型和训练都需要
        # prepare supervised training data
        classifier_data = data.get_classfier_data(args.c, vocab)

        graph, init = sg.build(embedding_size=args.i,
                               num_neg=args.n,
                               batch_size=args.b,
                               vocabulary_size=vocab.nodes_size,
                               transmissibility_size=vocab.nodepairs_size,
                               learning_rate=args.l,
                               classifier_type=args.c,
                               label_size=1)
        data.embeddings = sg.train(walks_dir=fs.PATH['walk'], vocab=vocab,
                                   graph=graph, init=init, window_size=args.w, batch_size=args.b,
                                   classifier_data=classifier_data,
                                   log_dir='log',
                                   classifier_type=args.c)
        data.write_embedding(fs.PATH['embedding'], data.embeddings)


if __name__ == "__main__":
    time_start = time.time()
    print 'WELCOME! \n*******************************************\n' \
          'Meta.Schema.2.Vec \n 1 -- raw->adj \n 2 -- adj->walk \n 3 -- walk->embedding  ' \
          '\n*******************************************'
    args = parse_args()
    main(args)
    print 'FINISH! Run time=%.2fs' % (time.time() - time_start)
