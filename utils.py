#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Bruce Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.
import os
import time
import math
import sys
from randomwalk import alias_sampling, transprob_setting
from settings import FileSettings


def checkLocalFiles(flow_start_p, fs):
    """
    a : raw data
    b : clean adj data
    c : walks
    d : embedding
    :return:
    """
    print 'CHECKING FILES!'
    if not all(fs.GROUP_EXIST[:flow_start_p]):
        print 'Lack of inputs, SYSTEM TERMINATION! \n'
        exit()

    for folder, group, exis in zip(fs.FOLDER[flow_start_p:], fs.GROUP[flow_start_p:], fs.GROUP_EXIST[flow_start_p:]):
        if any(exis) is True:
            print("%s already exist. I'll delete it in 5 minutes. Ctrl-C to abort." % folder)
            time.sleep(5)
            try:
                [os.system('rm %s' % fs.PATH[g]) for i, g in enumerate(group) if exis[i] is True]
            except ValueError:
                print ValueError
    print 'OVER CHECK!'


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def interval01(z):
    if z > 1:
        return 1.0
    elif z < 0:
        return 0.0
    else:
        return z


def test_metapathsrandomwalk(netx, metaschema, number_walks, walk_length, nodes_info, edges_info):
    APA = ['author', 'paper', 'author']
    APPA = ['author', 'paper', 'paper', 'author']
    # PA = ['paper', 'author']
    APVPA = ['author', 'paper', 'venue', 'paper', 'author']
    # VPA = ['venue', 'paper', 'author']
    APTPA = ['author', 'paper', 'term', 'paper', 'author']
    # TPA = ['term', 'paper', 'author']
    metapaths = [APA, APPA, APVPA, APTPA]
    walks = list()
    length_filter = 5
    total_edges = set()
    for i, j in netx.edges():
        total_edges.add((i, j))
        total_edges.add((j, i))

    total_nodes = set()
    for i in netx.nodes():
        total_nodes.add(i)

    cur_search_edges = set()
    cur_search_nodes = set()
    tran_times = 0
    while True:
        for batch_u0, batch_t0 in netx.nodes('type'):
            if len(cur_search_edges) > 0.99 * len(total_edges):
                print '\n total need ' + str(tran_times) + 'to search all edges!'
                return walks
            # if len(cur_search_nodes) > 0.99 * len(total_nodes):
            #     print '\n total need ' + str(tran_times) + 'to search all nodes!'
            #     return walks

            if tran_times % 100 == 0 and tran_times != 0:
                from itertools import groupby
                cur_search_nodes = list(cur_search_nodes)
                cur_search_nodes.sort(key=lambda x: x[0])
                groupby_cur_search_nodes = groupby(cur_search_nodes, key=lambda x: x[0])
                num_in_netx = dict()
                for key, group in groupby_cur_search_nodes:
                    num_in_netx[key] = len(list(group))

                sys.stdout.write(
                    "\r>>> tran_times:" + str(tran_times) + " ;edge search ratio:" + str(
                        len(cur_search_edges)) + '/' + str(
                        len(total_edges)) + " ;node search ratio:" + str(len(cur_search_nodes)) + '/' + str(
                        len(total_nodes)) + ' ;term:' + str(num_in_netx['term']) + ' ;paper:' + str(
                        num_in_netx['paper']) + ' ;venue:' + str(num_in_netx['venue']) + ' ;author:' + str(
                        num_in_netx['author']))
                sys.stdout.flush()
                cur_search_nodes = set(cur_search_nodes)
            cur_search_nodes.add(tuple(batch_u0.split('-')))
            # 多个metapath 选择一个
            for m in metapaths:
                if batch_t0 not in m[0]:
                    continue
                for _ in range(0, number_walks):
                    walk = [batch_u0]
                    cur_n = batch_u0
                    i = 0
                    for _ in range(0, walk_length):
                        i += 1
                        if i == len(m):
                            i = 0
                        ngb_t = m[i]
                        ngbs_types_and_probs = [[v, netx.nodes[v]['type'], 1] for u, v, t in
                                                netx.edges(cur_n, keys=True) if
                                                netx.nodes[v]['type'] == ngb_t and v not in walk]
                        if len(ngbs_types_and_probs) is 0:
                            # print 'ImpasseDirection'
                            break
                        ngbs, types, probs = zip(*ngbs_types_and_probs)
                        rand_id = alias_sampling(probs)

                        cur_search_edges.add((cur_n, ngbs[rand_id]))
                        cur_search_edges.add((ngbs[rand_id], cur_n))

                        cur_n = ngbs[rand_id]
                        walk.append(cur_n)

                        cur_search_nodes.add(tuple(cur_n.split('-')))

                        tran_times += 1
                    if len(walk) > length_filter:
                        walks.append(walk)
    time.sleep(0.1)
    return walks


def test_metaschemarandomwalks(netx, metaschema, number_walks, walk_length, nodes_info, edges_info):
    walks = list()
    length_filter = 5
    total_edges = set()
    for i, j in netx.edges():
        total_edges.add((i, j))
        total_edges.add((j, i))

    total_nodes = set()
    for i in netx.nodes():
        total_nodes.add(i)

    cur_search_edges = set()
    cur_search_nodes = set()
    tran_times = 0

    metaschema = transprob_setting(metaschema, 'invedges', nodes_info, edges_info)

    while True:
        for batch_u0, batch_t0 in netx.nodes('type'):
            if len(cur_search_edges) > 0.99 * len(total_edges):
                print '\n total need ' + str(tran_times) + 'to search all edges!'
                return walks
            if len(cur_search_nodes) > 0.99 * len(total_nodes):
                print '\n total need ' + str(tran_times) + 'to search all nodes!'
                return walks

            if tran_times % 100 == 0 and tran_times != 0:
                from itertools import groupby
                cur_search_nodes = list(cur_search_nodes)
                cur_search_nodes.sort(key=lambda x: x[0])
                groupby_cur_search_nodes = groupby(cur_search_nodes, key=lambda x: x[0])
                num_in_netx = dict()
                for key, group in groupby_cur_search_nodes:
                    num_in_netx[key] = len(list(group))

                sys.stdout.write(
                    "\r>>> tran_times:" + str(tran_times) + " ;edge search ratio:" + str(
                        len(cur_search_edges)) + '/' + str(
                        len(total_edges)) + " ;node search ratio:" + str(len(cur_search_nodes)) + '/' + str(
                        len(total_nodes)) + ' ;term:' + str(num_in_netx['term']) + ' ;paper:' + str(
                        num_in_netx['paper']) + ' ;venue:' + str(num_in_netx['venue']) + ' ;author:' + str(
                        num_in_netx['author']))
                sys.stdout.flush()
                cur_search_nodes = set(cur_search_nodes)

            if batch_t0 not in metaschema:
                continue
            cur_search_nodes.add(tuple(batch_u0.split('-')))
            for _ in range(0, number_walks):
                walk = [batch_u0]
                cur_n = batch_u0
                cur_t = batch_t0
                for _ in range(0, walk_length):
                    ngb_t = metaschema[cur_t]
                    ngbs_types_and_probs = [[v, netx.nodes[v]['type'], ngb_t[(t, netx.nodes[v]['type'])]] for u, v, t in
                                            netx.edges(cur_n, keys=True) if
                                            (t, netx.nodes[v]['type']) in ngb_t and v not in walk]
                    if len(ngbs_types_and_probs) is 0:
                        # print 'ImpasseDirection'
                        break
                    ngbs, types, probs = zip(*ngbs_types_and_probs)
                    rand_id = alias_sampling(probs)

                    cur_search_edges.add((cur_n, ngbs[rand_id]))
                    cur_search_edges.add((ngbs[rand_id], cur_n))

                    cur_n = ngbs[rand_id]
                    cur_t = types[rand_id]
                    walk.append(cur_n)
                    tran_times += 1

                    cur_search_nodes.add(tuple(cur_n.split('-')))
                if len(walk) > length_filter:
                    walks.append(walk)
        time.sleep(0.1)
    return walks
