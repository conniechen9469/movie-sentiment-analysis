#coding=utf-8
from gensim.models import Word2Vec, KeyedVectors

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logging.info("running %s" % ' '.join(sys.argv))

type = 'char_num'
# type = 'char_num_punc'

# Corpus
train_file = 'data/train_{}.tsv'.format(type)
test_file = 'data/test_{}.tsv'.format(type)

files = [train_file, test_file]
sentences = []
for file in files:
    corpus = open(file, 'r').readlines()[1:]
    for line in corpus:
        sentences.append(line.split('\t')[2].split())  # sentences should be a list of lists of tokens

# Train Model
model = Word2Vec(sentences, size=20, window=5, sg=0, hs=1, negative=5, workers=4)
model.save('data/word2vec_20_{}.bin'.format(type))
model.wv.save_word2vec_format('data/word2vec_20_{}_vec.bin'.format(type), binary=True)


# Load Model
# model = Word2Vec.load('data/word2vec_20_char_num.bin')
# word_vectors = KeyedVectors.load_word2vec_format('data/word2vec_20_char_num_vec.bin', binary=True)


# Test
# results = model.most_similar([u'good'], topn=10)
# for result in results:
#     print result

# ('great', 0.8133276104927063)
# ('real', 0.7902841567993164)
# ('there', 0.7582378387451172)
# ('nice', 0.7552671432495117)
# ('passably', 0.7376144528388977)
# ('weird', 0.7301462292671204)
# ('big', 0.7087181806564331)
# ('loaded', 0.6998113393783569)
# ('different', 0.6919705867767334)
# ('that', 0.6861714124679565)
