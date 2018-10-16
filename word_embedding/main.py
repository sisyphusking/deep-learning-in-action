from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
import os

path = './data/wiki.zh.word.text'
if not os.path.exists(path):
    t0 = int(time.time())
    sentences = LineSentence(path)
    model = Word2Vec(sentences, size=128, window=5, min_count=5, workers=4)
    print(' 训练耗时{} s'.format(int(time.time()) - t0))
    model.save('./data/gensim_128')
model = Word2Vec.load('./data/gensim_128')
# 相关词
items = model.wv.most_similar('人工智能')
for i, item in enumerate(items):
    print(i, item[0], item[1])

# 语义类比
print('=' * 20)
items = model.wv.most_similar(positive=['纽约', '中国'], negative=['北京'])
for i, item in enumerate(items):
    print(i, item[0], item[1])
# 不相关词
print('=' * 20)
print(model.wv.doesnt_match(['早餐', '午餐', '晚餐', '手机']))
计算相关度
print('=' * 20)
print(model.wv.similarity('男人', '女人'))
