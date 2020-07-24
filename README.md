# NLP-Learning
Record of NLP learning process.

NLP学习笔记
===========  
------------
# > 2020/07/21 task1
----------
### 1. 学前准备  
* 登录学习 __Markdown__ 的基础知识    参考[菜鸟教程](https://www.runoob.com)
* 了解 __GitHub__ 上学习建立 __repositery__ 和相关基础操作  参考GitHub新手引导和[GitHub.com 帮助文档](https://docs.github.com/cn)
-----
### 2. task1赛事详情与资料查找
* 对 __零基础入门NLP之新闻文本分类__ 赛题目标和安排有了大致了解  
* 寻找了与 __NLP__ 相关的学习资料，比如[开源组织ApacheCN编著的AiLearning相关学习路线和资料](https://github.com/ToNextOne2018/AiLearning)以及[廖雪峰的官方网站的Python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)  
-------
### 3. 总结
* 路漫漫其修远兮，要想学有所成，还需不断努力、不断思考。
* 艰难困苦，玉汝于成。  
------------------------------  
# > 2020/07/22 task2
```python
# shift+Tab  显示工具提示

# Tab+shift+M  合并代码块
```

# 初始读取数据


```python
import pandas as pd   # 读取数据
train_df = pd.read_csv('C:/Users/14279/Desktop/NLP_news/train_set.csv',sep='\t')
train_df.head(10)  #查看前几行数据是否读入正确，默认为5行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>2967 6758 339 2021 1854 3731 4109 3792 4149 15...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11</td>
      <td>4464 486 6352 5619 2465 4802 1452 3137 5778 54...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>7346 4068 5074 3747 5681 6093 1777 2226 7354 6...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>7159 948 4866 2109 5520 2490 211 3956 5520 549...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td>3646 3055 3055 2490 4659 6065 3370 5814 2465 5...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>9</td>
      <td>3819 4525 1129 6725 6485 2109 3800 5264 1006 4...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3</td>
      <td>307 4780 6811 1580 7539 5886 5486 3433 6644 58...</td>
    </tr>
    <tr>
      <td>7</td>
      <td>10</td>
      <td>26 4270 1866 5977 3523 3764 4464 3659 4853 517...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>12</td>
      <td>2708 2218 5915 4559 886 1241 4819 314 4261 166...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3</td>
      <td>3654 531 1348 29 4553 6722 1474 5099 7541 307 ...</td>
    </tr>
  </tbody>
</table>
</div>



__分析：__ 从上图显示的头10行数据，可以看出数据读入成功，可以进行下一步数据分析操作

# 新闻类型对应标签  
> {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5,'教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}



## 1. 句子长度分析


```python
%pylab inline
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))  # 分析text对应的句子的长度情况
print(train_df['text_len'].describe())
```

    Populating the interactive namespace from numpy and matplotlib
    count    200000.000000
    mean        907.207110
    std         996.029036
    min           2.000000
    25%         374.000000
    50%         676.000000
    75%        1131.000000
    max       57921.000000
    Name: text_len, dtype: float64
    

__分析：__  
* 总共有200000条新闻
* 句子平均新闻数为907个
* 最少新闻类别只有2条新闻，最多新闻类别有57921条新闻

__可视化：__ 分布直方图  
[matplotlib.pyplot.hist(x,bins=None,range=None, density=None, bottom=None, histtype='bar', align='mid', log=False, color=None, label=None, stacked=False, normed=None)参数介绍](https://blog.csdn.net/ToYuki_/article/details/104114925)  



```python
_ = plt.hist(train_df['text_len'], bins=100000)   # 200000行数据分成100000个柱子
plt.xlabel('Text char count')
plt.title("Histogram of char count")
```




    Text(0.5, 1.0, 'Histogram of char count')




![png](https://github.com/ToNextOne2018/NLP-Learning-Notes/blob/master/jupyterTest1/output_9_1.png)



```python
_a = plt.hist(train_df['text_len'][:100000], bins=200000)   # 截取前100000行数据进行观察
plt.xlabel('Text char count')
plt.title("Graph1 Histogram of char count")
```




    Text(0.5, 1.0, 'Graph1 Histogram of char count')




![png](https://github.com/ToNextOne2018/NLP-Learning-Notes/blob/master/jupyterTest1/output_10_1.png)



```python
_b = plt.hist(train_df['text_len'][100000:], bins=200000)   # 截取后100000行数据进行观察
plt.xlabel('Text char count')
plt.title("Graph2 Histogram of char count")
```




    Text(0.5, 1.0, 'Graph2 Histogram of char count')




![png](https://github.com/ToNextOne2018/NLP-Learning-Notes/blob/master/jupyterTest1/output_11_1.png)


__分析：__  
* 由于数据量比较大，运算较慢，故分为前后各100000进行运算，可以看到大部分句子长度都集中在10000以内，且大部分句子的长度都小于5000

## 2. 新闻类别分布  
> {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5,'教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


```python
train_df['label'].value_counts().plot(kind='bar')    # 计算类别label的统计数据
plt.title('News class count')
plt.xlabel("category")
```




    Text(0.5, 0, 'category')




![png](https://github.com/ToNextOne2018/NLP-Learning-Notes/blob/master/jupyterTest1/output_14_1.png)


__分析：__   
* 数据分布大致呈阶梯状，其中属于 ___科技类___ 的新闻数量最多，超过了35000，而关于 ___星座类___ 的新闻数量最少，不足5000  
-------------  
## 3. 字符分布统计  
[ sorted(L, key=lambda x:x[1]) ](https://www.cnblogs.com/zle1992/p/6271105.html)


```python
from collections import Counter
all_lines = ' '.join(list(train_df['text']))   # 将text的中的字符重新组合成一个序列（由元组构成的列表）
word_count = Counter(all_lines.split(" "))    # 通过空格对字符进行划分，进而统计序列中总的字符数
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)   # 按照元组中第二个元组进行排序

print(len(word_count))

print(word_count[0])

print(word_count[-1])
```

    6869
    ('3750', 7482224)
    ('3133', 1)
    

__分析:__  
* 从上面可以看出,该数据集共有6869个字符（不重复）  
* '3750'这个字符出现次数最多，共出现7482224次  
* '3133'这个字符只出现了1次


```python
from collections import Counter
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])

print(word_count[1])

print(word_count[2])
```

    ('3750', 197997)
    ('900', 197653)
    ('648', 191975)
    

__分析：__  
* 从上面结果可以看出，在200000个新闻中，含有'3750'这个字符的新闻占了197997个  
* 含有'900'这个字符的新闻占了197653个  
* 含有'648'这个字符的新闻占了191975个  
* '3750'、'900'、'648'这三个字符很大概率是标点符号
---------------------------  
## 4. 习题
__* 题目一：__

* 假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成？

{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5,'教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


```python
import pandas as pd  
train_df = pd.read_csv('C:/Users/14279/Desktop/NLP_news/train_set.csv',sep='\t')   # 读取全部数据
train_df.head(3)  #查看前几行数据是否读入正确，默认为5行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>2967 6758 339 2021 1854 3731 4109 3792 4149 15...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11</td>
      <td>4464 486 6352 5619 2465 4802 1452 3137 5778 54...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>7346 4068 5074 3747 5681 6093 1777 2226 7354 6...</td>
    </tr>
  </tbody>
</table>
</div>



1. 连接句子  
2. 通过三个字符划分句子
3. 新闻均句子数


```python
%pylab inline
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split('3750' or '900' or '648')))   # 通过375，900，648三个数进行句子划分
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
i = 0
j = len(train_df['label'])  # 设置j，获取数据总行数
total_number = 0
while(i < j):
    print('第{0}条新闻有{1}个句子'.format(i,train_df['text_len'][i]))
    total_number = total_number+train_df['text_len'][i]
    i= i+1
   
print('总句子数为{0},平均每条新闻有{1}个句子'.format(total_number,total_number/j))
```

    第0条新闻有65个句子
    第1条新闻有26个句子
    第2条新闻有28个句子
    第3条新闻有87个句子
    第4条新闻有12个句子
    第5条新闻有27个句子
    第6条新闻有5个句子
    第7条新闻有44个句子
    第8条新闻有2个句子
    第9条新闻有47个句子
    第10条新闻有35个句子  
    …………  
    第199987条新闻有8个句子
    第199988条新闻有53个句子
    第199989条新闻有12个句子
    第199990条新闻有5个句子
    第199991条新闻有67个句子
    第199992条新闻有15个句子
    第199993条新闻有7个句子
    第199994条新闻有137个句子
    第199995条新闻有42个句子
    第199996条新闻有68个句子
    第199997条新闻有52个句子
    第199998条新闻有8个句子
    第199999条新闻有85个句子
    总句子数为7682224,平均每条新闻有38.41112个句子
```python

```   
__分析__  
* 由上面可得总句子数为 __7682224__ 个，每条新闻平均有 __38.41112__ 个句子
* 本题难点在于如何理解与 __DataFrame__ 相关的函数调用方法和 __while__ 循环语句的使用

__* 题目二：__

{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


```python
label_list = list(train_df['label'])   # 将标签和字符分别存入两个列表中
text_list = list(train_df['text'])
i = 0
list_total = [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']

# 将同一种类型的标签所属的新闻合为一个字符串，存储在长度为13的列表中
while(i < len(label_list)):                                               
    list_total[label_list[i]] = '{0} {1}'.format(list_total[label_list[i]],text_list[label_list[i]])
    i += 1
    
    #list_total[label_list[i]].join(text_list[label_list[i]])   all_lines = ' '.join(list(train_df['text']))
       
```


```python
from collections import Counter
i = 0
word_count = [0 for _ in range(14)]   # 生成非空列表word_count
while(i < 14):                        # 将每类新闻所对应的字符串以空格为间隔生成对应的列表
    word_count[i] = Counter(list_total[i].split(" "))
    word_count[i] = sorted(word_count[i].items(), key=lambda d:d[1], reverse = True)
    print('{0}类新闻出现最多的字符为{1}'.format(i,word_count[i][0]))
    i = i+1

```

    0类新闻出现最多的字符为('3750', 2490752)
    1类新闻出现最多的字符为('3750', 923625)
    2类新闻出现最多的字符为('3750', 848475)
    3类新闻出现最多的字符为('3750', 1903438)
    4类新闻出现最多的字符为('3055', 180192)
    5类新闻出现最多的字符为('3317', 330264)
    6类新闻出现最多的字符为('2465', 109835)
    7类新闻出现最多的字符为('3750', 380163)
    8类新闻出现最多的字符为('6220', 462973)
    9类新闻出现最多的字符为('3750', 270388)
    10类新闻出现最多的字符为('3750', 167280)
    11类新闻出现最多的字符为('3370', 93930)
    12类新闻出现最多的字符为('465', 27315)
    13类新闻出现最多的字符为('1641', 10896)
    


```python

```  
__分析__  
* 从上面分析结果可以很清晰看出各个新闻类别对应的出现次数最多的字符

---------------------------------------------
# > 2020/07/23 task3    
# 基于机器学习的文本分类  
----------------------------------------------

## 1. 模型初试



```python
# Count Vectors + RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('C:/Users/14279/Desktop/NLP_news/train_set.csv', sep='\t', nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.74
```

    0.65441877581244
    


```python
# TF-IDF +  RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('C:/Users/14279/Desktop/NLP_news/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.87
```

    0.8719098297954606
    

## 2. 习题    
__* 题目一：__  
基于控制变量法，通过增大训练集，来观察最终训练模型的预测精度


```python
# TF-IDF +  RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('C:/Users/14279/Desktop/NLP_news/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:15000], train_df['label'].values[:15000])  # 增大训练的数据集到15000

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

```

    0.9283205830005475
    

__分析：__   
* 很明显，将训练的数据集由10000增大到15000，提高了模型的精度  
----------------------------------------------  
__* 题目二：__ 
