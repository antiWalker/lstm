#!/usr/bin/env python
# coding: utf-8

# 预处理

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import jieba as jb
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

# # 数据预处理
#
#  接下来我们要将cat类转换成id，这样便于以后的分类模型的训练。

# In[70]:


df['cat_id'] = df['cat'].factorize()[0]
cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)
df.sample(10)

# In[71]:


# cat_id_df.write.mode("overwrite").options(header="true").csv("./cat_id_df")

cat_id_df.to_csv('./cat_id_df.csv', sep=',', header=True, index=True)


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
df['clean_review'] = df['review'].apply(remove_punctuation)
df.sample(10)

# 我们过滤掉了review中的标点符号和一些特殊符号，并生成了一个新的字段 clean_review。接下来我们要在clean_review的基础上进行分词,把每个评论内容分成由空格隔开的一个一个单独的词语。

# In[ ]:


# 分词，并过滤停用词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df.head()
df.to_csv('./df.csv', sep=',', header=True, index=True)
print("预处理完成！")
