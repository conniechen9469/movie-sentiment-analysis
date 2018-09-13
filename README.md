# movie-sentiment-analysis
卡狗小分队


### data

这个文件夹里面分成原先的语料和我预处理的语料、预训练的词向量。
原先的语料：
注意这个语料是用句法树代表整个影评句子的，包括根、中间节点和叶子节点都要打上标签。
- train.tsv：训练集
- test.tsv：测试集
- sampleSubmission.csv：提交的格式

我后面处理的语料：
preprocess.py产生的：
- train_char_num_punc.tsv：有英文，数字和正常标点符号的训练语料
- train_char_num.tsv：只有英文和数字的训练语料
- test_char_num_punc.tsv：有英文，数字和正常标点符号的测试语料
- test_char_num.tsv：只有英文和数字的测试语料

count.py产生的：
- count_char_num.txt：只有英文和数字的语料统计，包括词典大小，句子长度，词频
- count_char_num_punc.txt: 有英文、数字和正常标点符号的语料统计，包括词典大小，句子长度，词频

vectors_word2vec.py预训练的词向量：
- word2vec_20_char_num.bin：用的只有字母和数字的语料，训练的word2vec(cbow）模型
- word2vec_20_char_num_vec.bin：，用的只有字母和数字的语料, 训练的word2vec(cbow）模型的词向量，二进制的KeyVectors读写
具体使用看文件里被我注释掉那一段代码。


### preprocess.py
这个文件预处理训练和测试语料，得到两种语料：
- 有英文，数字和正常标点符号的语料
- 只有英文和数字的语料

具体的步骤是：
- 去除掉非目标字符（非英文、数字等）
- 分词
- 词性还原
在这里我并没有去除停用词，因为我试过除去停用词之后句子看起来不是很方便训练，而且句法树很多分支都已经是空白的了。

### vectors_word2vec.py
直接用的gensim，word2vec(cbow）模型，向量大小20（字典大小|D|大概是17k左右，词向量取log2|D|，这个是一种经验值，不一定是对的），窗口5，用的softmax huffman，负采样5.




