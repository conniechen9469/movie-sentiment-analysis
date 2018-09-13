# coding=utf-8

# import nltk
# nltk.download()

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

in_train_file = 'data/train.tsv'
in_test_file = 'data/test.tsv'

out_train_punc_file = 'data/train_char_num_punc.tsv'
out_test_punc_file = 'data/test_char_num_punc.tsv'

out_train_no_punc_file = 'data/train_char_num.tsv'
out_test_no_punc_file = 'data/test_char_num.tsv'

# all words are in lower case

# corpus exists ":"， we don't keep them
# only keep char, numbers, comma, period, question mark, exclamatory mark,
# single quote and single whitespace
non_nums_english_punc = re.compile('[^a-zA-Z0-9,.?!\-\'\ ]')
non_nums_english = re.compile('[^a-zA-Z0-9\-\'\ ]')

# here we replace "..." with "."
dotn = re.compile('\.{3}')

# stop words
stop_words = set(stopwords.words('english'))

# lemmatization object
lemma = WordNetLemmatizer()


def preprocess_train(in_file, out_file, pattern):
    print("preprocess from {} to {}".format(in_file, out_file))
    fo = open(in_file, 'r')
    fs = open(out_file, 'w')

    corpus = fo.readlines()

    cols = corpus[0]
    fs.writelines(cols)

    corpus = corpus[1:]

    cnt = 0
    for line in corpus:
        pid, sid, phrase, sentiment = line.split('\t')

        out_phrase = re.sub(pattern, '', phrase)
        out_phrase = re.sub(dotn, '.', out_phrase)
        out_phrase = out_phrase.lower()

        # out_phrase_words = [lemma.lemmatize(word)
        #                     if word not in stop_words else ' '
        #                     for word in word_tokenize(out_phrase)]

        out_phrase_words = [lemma.lemmatize(word) for word in word_tokenize(out_phrase)]

        out_phrase = ' '.join(out_phrase_words)
        out_phrase = ' '.join(out_phrase.split())

        if out_phrase == '':
            out_phrase = '.'

        result = [pid, sid, out_phrase, sentiment]
        out_line = '\t'.join(result)
        fs.writelines(out_line)

        cnt += 1
        if cnt % 10000 == 0:
            print("     {} lines processed.".format(cnt))
    print("     finished, total {} lines processed.".format(cnt))

    fo.close()
    fs.close()


def preprocess_test(in_file, out_file, pattern):
    print("preprocess from {} to {}".format(in_file, out_file))
    fo = open(in_file, 'r')
    fs = open(out_file, 'w')

    corpus = fo.readlines()

    cols = corpus[0]
    fs.writelines(cols)

    corpus = corpus[1:]

    cnt = 0
    for line in corpus:
        pid, sid, phrase = line.split('\t')

        out_phrase = re.sub(pattern, '', phrase)
        out_phrase = re.sub(dotn, '.', out_phrase)
        out_phrase = out_phrase.lower()

        # out_phrase_words = [lemma.lemmatize(word)
        #                     if word not in stop_words else ' '
        #                     for word in word_tokenize(out_phrase)]

        out_phrase_words = [lemma.lemmatize(word) for word in word_tokenize(out_phrase)]

        out_phrase = ' '.join(out_phrase_words)
        out_phrase = ' '.join(out_phrase.split())

        if out_phrase == '':
            out_phrase = '.'

        result = [pid, sid, out_phrase]

        out_line = '\t'.join(result)
        fs.writelines(out_line+'\n')

        cnt += 1
        if cnt % 10000 == 0:
            print("     {} lines processed.".format(cnt))
    print("     finished, total {} lines processed.".format(cnt))

    fo.close()
    fs.close()


preprocess_train(in_train_file, out_train_punc_file, non_nums_english_punc)
preprocess_train(in_train_file, out_train_no_punc_file, non_nums_english)

preprocess_test(in_test_file, out_test_punc_file, non_nums_english_punc)
preprocess_test(in_test_file, out_test_no_punc_file, non_nums_english)







