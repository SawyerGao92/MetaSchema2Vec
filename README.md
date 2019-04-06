# MetaSchema2Vec

This repository provides a reference implementation of *metaschema2vec* as described in the paper:

> MetaSchema2Vec: Learning Flexible Representations of Highly Heterogeneous Networks

The *metaschema2vec* algorithm first introduces the concept of meta-schema in the data sampling phase and perform meta-schema-based random walks to sample similar node pairs. Then, it constructs a weighted skip-gram model to capture the different transmissibilities of different types of node pairs in the representation learning phase. 

## Usage

```shell
python main.py --<FLAG>=VALUE`
```

where the flags are defined as :

```
# global parameter
parser.add_argument('-d', '--data', help='Data abbreviation', default='DBLP', choices=['StackOverflow', 'DBLP'], dest='d')
parser.add_argument('-s', '--start-position', help='The start position in the flow', default=2, type=int, dest='s')
# random walk parameter
parser.add_argument('--number_walks', help='number of walks per node', default=20, type=int)
parser.add_argument('--walk_length', help='walk length per node', default=50, type=int)
parser.add_argument('--trans_prob', help='walk length per node', choices=['average', 'invnodes', 'invedges'], default='average', type=str)
# representation learning parameter
parser.add_argument('-i', '--embedding_size', help='Dimensionality of node embeddings', default=100, type=int, dest='i')
parser.add_argument('-b', '--batch_size', help='batch size', default=128, type=int, dest='b')
parser.add_argument('-w', '--window', help='Max window length', default=5, type=int, dest='w')
parser.add_argument('-n', '--num_neg', help='Number of negative examples', default=5, type=int, dest='n')
parser.add_argument('-l', '--learning_rate', help='learning rate', default=0.01, type=float, dest='l')
parser.add_argument('-aw', '--auto_weight', help='supervised learning weight while learning embedding', default=True, type=bool, dest='aw')
parser.add_argument('-c', '--classifier_type', help='Node classfication/Link prediction', default='link', choices=['node', 'link'], dest='c')
```

*Note:* This is only a reference implementation of the *metaschema2vec* algorithm and could benefit from several performance enhancement schemes, some of which are discussed in the paper.