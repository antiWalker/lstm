#!/usr/bin/env python
# coding: utf-8

# 在我之前的博客中我们介绍了文本的多分类的方法,我们还尝试了各种分类模型,比如朴素贝叶斯、逻辑回归、支持向量机和随机森林等并且都取得了非常不错的效果。今天我们使用深度学习中的LSTM（Long Short-Term Memory）长短期记忆网络，它是一种时间循环神经网络，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件。
# LSTM 已经在科技领域有了多种应用。基于 LSTM 的系统可以学习翻译语言、控制机器人、图像分析、文档摘要、语音识别图像识别、手写识别、控制聊天机器人、预测疾病、点击率和股票、合成音乐等等任务。今天我们用它来实现一下文本多分类，相信会取得较好的效果。

# # 数据
#
# 我们的数据来自于互联网，你可以在这里下载,数据中包含了10 个类别（书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店），共 6 万多条评论数据 首先查看一下我们的数据，这些数据都是来自于电商网站的用户评价数据,我们想要把不同评价数据分到不同的分类中去,且每条数据只能对应10个类中的一个类。
#
# 数据下载地址:https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb
#

# In[1]:


import matplotlib.pyplot as plt
# coding=utf-8
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import re

# In[2]:


df = pd.read_csv('./data/online_shopping_10_cats.csv')
df = df[['cat', 'review']]
print("数据总量: %d ." % len(df))
df.sample(10)

# In[ ]:


print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
df[df.isnull().values == True]
df = df[pd.notnull(df['review'])]

# In[ ]:


d = {'cat': df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
df_cat

# In[ ]:


df_cat.plot(x='cat', y='count', kind='bar', legend=False, figsize=(8, 5))
plt.title("类目分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)


# 我们将cat转换成了Id(0到9),由于我们的评价内容都是中文,所以要对中文进行一些预处理工作,这包括删除文本中的标点符号,特殊符号,还要删除一些无意义的常用词(stopword),因为这些词和符号对系统分析预测文本的内容没有任何帮助,反而会增加计算的复杂度和增加系统开销,所有在使用这些文本数据之前必须要将它们清理干净。

# In[72]:


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 加载停用词
stopwords = stopwordslist("./data/chineseStopWords.txt")

# 中文停用词包含了很多日常使用频率很高的常用词,如 吧，吗，呢，啥等一些感叹词等,这些高频常用词无法反应出文本的主要意思,所以要被过滤掉。

# In[73]:


# 删除除字母,数字，汉字以外的所有符号
df = pd.read_csv("df.csv", encoding='utf-8', dtype=str)
df = df.astype(str)
# # LSTM建模 
#
# 数据预处理完成以后，接下来我们要开始进行LSTM的建模工作：
#
# * 我们要将cut_review数据进行向量化处理,我们要将每条cut_review转换成一个整数序列的向量
# * 设置最频繁使用的50000个词
# * 设置每条 cut_review最大的词语数为250个(超过的将会被截去,不足的将会被补0)

# In[ ]:


# 设置最频繁使用的50000个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)
MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cut_review'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))

# 训练和测试的数据集都准备好以后,接下来我们要定义一个LSTM的序列模型:
#
# * 模型的第一次是嵌入层(Embedding)，它使用长度为100的向量来表示每一个词语
# * SpatialDropout1D层在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合
# * LSTM层包含100个记忆单元
# * 输出层为包含10个分类的全连接层
# * 由于是多分类，所以激活函数设置为'softmax'
# * 由于是多分类, 所以损失函数为分类交叉熵categorical_crossentropy

# In[ ]:


X = tokenizer.texts_to_sequences(df['cut_review'].values)
# 填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# 多类标签的onehot展开
Y = pd.get_dummies(df['cat_id']).values

print(X.shape)
print(Y.shape)

# In[ ]:


# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# In[ ]:


# 定义模型
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# model.save('model', save_format='tf')
# model.save('my_model.h5')
epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# In[ ]:

model.save('mymodel.h5')
