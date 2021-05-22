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


# coding=utf-8
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import jieba as jb
import re
from keras.models import load_model



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
#df = pd.read_csv("./df.csv")
df = pd.read_csv("df.csv",encoding='utf-8',dtype=str)
df = df.astype(str)
# print(df.head())
# print(df.dtypes)
# df['clean_review'] = df['review'].apply(remove_punctuation)
#df.sample(10)

# 我们过滤掉了review中的标点符号和一些特殊符号，并生成了一个新的字段 clean_review。接下来我们要在clean_review的基础上进行分词,把每个评论内容分成由空格隔开的一个一个单独的词语。

# In[ ]:


# 分词，并过滤停用词
# df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
print(df.dtypes)
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
#print('共有 %s 个不相同的词语.' % len(word_index))


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    model = load_model("/Users/wjn/pythonpro/lstm/mymodel.h5")
    #model.summary()
    pred = model.predict(padded)
    cat_id = pred.argmax(axis=1)[0]
    cat_id_df1 = pd.read_csv("./cat_id_df.csv")
    return cat_id_df1[cat_id_df1.cat_id == cat_id]['cat'].values[0]


# In[ ]:


print(predict('苹果好吃又不贵，已经买了很多次了'))

print(predict('收到产品已经半个多月了，一开始用着不太好用，慢慢的就好使了，可能是习惯问题吧，主要是屏的分辨率真的不错。'))
# In[ ]:


# predict('收到产品已经半个多月了，一开始用着不太好用，慢慢的就好使了，可能是习惯问题吧，主要是屏的分辨率真的不错。')
#
# # In[ ]:
#
#
# predict('可能中外文化理解差异，可能与小孩子太有代沟，不觉得怎么样，还是比较喜欢水墨画一点风格的漫画书，但愿我女儿长大一点能喜欢（22个月中）')
#
# # In[ ]:
#
#
# predict('假的，不好用，头皮痒的要命。。。')
#
# # In[ ]:
#
#
# predict('这是第三次点评，纯粹是为了拿积分。没什么可以多说了，服务不错。')
#
# # In[ ]:
#
#
# # 自定义review
# predict('房间挺大，就是价格贵了点')
#
# # In[ ]:
#
#
# # 自定义review
# predict('酸的要死，下次不买了')
#
# # In[ ]:
#
#
# # 自定义review
# predict('酸的要死，下次不买了')
#
# # In[ ]:
#
#
# # 自定义review
# predict('用了几天就好卡，上当了，退款')
#
# # In[ ]:
#
#
# # 自定义review
# predict('用了几次发现头好痒，感觉过敏了')
#
# # In[ ]:
#
#
# predict('手写识别还可以')
#
# # In[ ]:
#
#
# predict('T恤很好看，就是短了点')
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
