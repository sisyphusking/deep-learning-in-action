# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

with open('./data/tf_128.pkl', 'rb') as fr:
    data = pickle.load(fr)
    final_embeddings = data['embeddings']
    word2id = data['word2id']
    id2word = data['id2word']

word_indexs = []
count = 0
plot_only = 200
for i in range(1, len(id2word)):
    if len(id2word[i]) > 1:
        word_indexs.append(i)
        count += 1
        if count == plot_only:
            break

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[word_indexs, :])
labels = [id2word[i] for i in word_indexs]

plt.figure(figsize=(15, 12))
for i, label in enumerate(labels):
    x, y = two_d_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), ha='center', va='top', fontproperties='Microsoft YaHei')
plt.savefig('./images/词向量降维可视化.png')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./data/tf_128.meta')
saver.restore(sess, tf.train.latest_checkpoint('.'))
graph = tf.get_default_graph()
valid = graph.get_tensor_by_name('valid:0')
similarity = graph.get_tensor_by_name('MatMul_1:0')

word = '数学'
sim = sess.run(similarity, feed_dict={valid: [word2id[word]]})
top_k = 10
nearests = (-sim[0, :]).argsort()[1: top_k + 1]
s = 'Nearest to %s:' % word
for k in range(top_k):
    s += ' ' + id2word[nearests[k]]
print(s)


# 计算相关度
def cal_sim(w1, w2):
    return np.dot(final_embeddings[word2id[w1]], final_embeddings[word2id[w2]])


print(cal_sim('男人', '女人'))


# 相关词
word = '数学'
sim = [[id2word[i], cal_sim(word, id2word[i])] for i in range(len(id2word))]
sim.sort(key=lambda x:x[1], reverse=True)
top_k = 10
for i in range(top_k):
    print(sim[i + 1])


# 不相关词
def find_mismatch(words):
    vectors = [final_embeddings[word2id[word]] for word in words]
    scores = {word: np.mean([cal_sim(word, w) for w in words]) for word in words}
    scores = sorted(scores.items(), key=lambda x:x[1])
    return scores[0][0]


print(find_mismatch(['早餐', '午餐', '晚餐', '手机']))
