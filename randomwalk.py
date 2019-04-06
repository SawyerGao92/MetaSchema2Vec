#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-
# Author: Leo Gao
# Email: gao19920804@126.com
# Copyright 2018 Gxy. All Rights Reserved.
from tqdm import tqdm
import time


def MetaSchemaRandomWalk(netx, metaschema, transprob_setcriterion, number_walks, walk_length, node_info, edge_info):
    walks = list()
    length_filter = 5
    metaschema = transprob_setting(metaschema, transprob_setcriterion, node_info, edge_info)
    for batch_u0, batch_t0 in tqdm(netx.nodes('type')):
        if batch_t0 not in metaschema:
            continue
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
                cur_n = ngbs[rand_id]
                cur_t = types[rand_id]
                walk.append(cur_n)
            if len(walk) > length_filter:
                walks.append(walk)
    time.sleep(0.1)
    return walks


def transprob_setting(metaschema, transprob_setcriterion, node_info, edge_info):
    if transprob_setcriterion == 'average':
        for u, value in metaschema.items():
            prob = 1.0 / len(value)
            for v in value.keys():
                metaschema[u][v] = prob
        return metaschema
    elif transprob_setcriterion == 'invnodes':
        for u, value in metaschema.items():
            sum_num = sum([node_info[v] for (r, v) in value.keys()])
            inv_sum_num = sum([sum_num - node_info[v] for (r, v) in value.keys()])
            for (r, v) in value.keys():
                metaschema[u][(r, v)] = 1 if inv_sum_num == 0 else (sum_num - node_info[v]) / float(inv_sum_num)
        return metaschema
    elif transprob_setcriterion == 'invedges':
        for u, value in metaschema.items():
            sum_num = sum([edge_info[r] for (r, v) in value.keys()])
            inv_sum_num = sum([sum_num - edge_info[r] for (r, v) in value.keys()])
            for (r, v) in value.keys():
                metaschema[u][(r, v)] = 1 if inv_sum_num == 0 else (sum_num - edge_info[r]) / float(inv_sum_num)
        return metaschema
    else:
        raise NotImplementedError('This type of transprob_setcriterion is not supported. ')


def alias_sampling(probs):
    import numpy as np
    import numpy.random as npr

    def alias_setup(probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] - (1.0 - q[small])

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    def alias_draw(J, q):
        K = len(J)
        # Draw from the overall uniform mixture.
        kk = int(np.floor(npr.rand() * K))
        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if npr.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    J, q = alias_setup(probs)
    return alias_draw(J, q)
