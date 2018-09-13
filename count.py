
# import os

# data_path = 'data'
# data_files = os.listdir(data_path)


# train = pd.read_csv('data/train.tsv', sep='\t')
#
# # count word frequency
# phrase_corpus = train['Phrase']
#
# word_dict = dict()
# for phrase in phrase_corpus:
#     words = phrase.split()
#     for word in words:
#         if word in word_dict:
#             word_dict[word] += 1
#         else:
#             word_dict[word] = 1
#
#
# print("dict size: {}".format(len(word_dict)))  # dict size: 18226


# ____________
import pandas as pd

# type = 'char_num'
type = 'char_num_punc'

train_file = 'data/train_{}.tsv'.format(type)
test_file = 'data/test_{}.tsv'.format(type)

files = [train_file, test_file]
word_dict = dict()
sentence_length_dict = dict()
for file in files:
    f = open(file, 'r')
    corpus = f.readlines()
    for line in corpus:
        sentence = line.split('\t')[2]
        words = sentence.split()

        sentence_len = len(words)
        if sentence_len in sentence_length_dict:
            sentence_length_dict[sentence_len] += 1
        else:
            sentence_length_dict[sentence_len] = 1

        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1


f = open('data/count_{}.txt'.format(type), 'w')
f.writelines('dict_size: {}\n'.format(len(word_dict)))
f.writelines('#############################\n')

lengths = sorted(sentence_length_dict, key=sentence_length_dict.get)
f.writelines('maximum sentence length: {}\n'.format(max(lengths)))
f.writelines('minimum sentence length: {}\n'.format(min(lengths)))

f.writelines('length\tfrequency\n')
for sentence_len in lengths:
    f.writelines(str(sentence_len)+'\t'+str(sentence_length_dict[sentence_len])+'\n')
f.writelines('#############################\n')

f.writelines('words\tfrequency\n')
words = sorted(word_dict, key=word_dict.get, reverse=True)  # sort by frequency
for word in words:
    f.writelines(word+'\t'+str(word_dict[word])+'\n')




