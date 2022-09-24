---
layout: post
read_time: true
show_date: true
title:  recommender system
date:   2021-09-24 13:32:20 -0600
description: Single neuron perceptron that classifies elements learning quite quickly.
img: assets/img/posts/20210125/Perceptron.jpg 
tags: [machine learning, coding, neural networks]
author: Armando Maynez
github:  amaynez/Perceptron/
mathjax: yes
---

## 撮合平台概论$\cdot$第二章$\cdot$推荐策略概论

[TOC]

### 1.推荐系统(推荐是干啥用的)

<table>
    <tr>
        <td>
            <center>
            <img src='www.png' style='width:100%;height:100%'/>
            </br>
            <font>Fig 1.鸿荒时代</font>
            <center>
        </td>
        <td>
            <center>
            <img src='souhu.png' style='width:100%;height:100%'/>
            </br>
            <font>Fig2.门户网站</font>
            <center>
        </td>
    </tr>
        <td>
            <center>
            <img src='baidu.png' style='width:100%;height:100%'/>
            </br>
            <font>Fig 3.鸿荒时代</font>
            <center>
        </td>
        <td>
            <center>
            <img src='toutiao.png' style='width:100%;height:100%'/>
            </br>
            <font>Fig4.门户网站</font>
            <center>
        </td>
    <tr>
    </tr>
</table>


各个时期是如何找信息的?:
1. 鸿荒时代: 一屏一屏浏览, 看看有没有自己要的信息; 
2. 门户网站: 把一些比较有名的网站链接汇聚在一个地方, 用户可以在这个地方(门户网站)找到自己想要的信息, 如搜狐
3. 搜索引擎: 把网站上所有的信息整合起来, 用户通过问询(query)的方式去找想要的信息, 如百度
4. 推荐引擎: 问询的内容变为用户特征, 用户连输入都不需要了, 把人陪养的越来越懒了~, 如今日头条/抖音等

### 2.撮合系统分类
<img src='RecommendClass.png' style='width:400 px;height: 200px'/>


#### 2.1 非个性化推荐
<table>
    <tr>
        <td>
            <center>
            <img src='NonPersonal.png' style='width:100%;height:100%'/>
            <center>
        </td>
        <td>
            <center>
            <img src='Distance.png' style='width:100%;height:100%'/>
            <center>
        </td>
        <td>
            <center>
            <img src='Rating.png' style='width:100%;height:100%'/>
            <center>
        </td>
    </tr>
</table>


这种推荐方式不区分个人喜好, 所有人看到的推荐都是一致的, 常见的如微博热搜或橘子堆热搜
另外在某些平台会有按价格、距离排序的推荐, 这类也属于非个性化的推荐, 例如fleet撮合平台的司机市场 按价格和距离tab
#### 2.2 半个性化推荐
<img src='Tag.png' style='width:400 px;height: 300px'/>

半个性化推荐则在个性化上稍进一步, 开始使用用户信息, 先把用户分为一个个群体, 然后在各个群体中进行非个性化推荐
即不同群体间的两个个体看到的推荐是不一致的, 而同一个群体中的两个个体看到的推荐是一致的
比如撮合平台推荐功能, 司机市场的筛选功能, 当用户设置筛选标签时, 也相应分分为不同的组, 随后根据组特性进行推荐
常用方法: 基于内容的推荐
#### 2.3 个性化推荐
<img src='iqiyi.png' style='width:400 px;height: 300px'/>

个性化推荐把用户的信息挖掘到了极致, 把圈子缩小到了个人维度, 根据每个人的特征进行推荐,
给每个人与每个人之间的推荐商品是不同的,
常用方法: 基于内容的推荐、基于协同过滤的推荐

### 3.推荐策略设计流程(如何设计一个好的推荐策略?)
<img src='RecommendProcess.png' style='width:600 px;height: 800px'/>

1. 根据手中所有的信息, 设计一个基本的推荐策略
2. 根据手中有的数据, 在线下对推荐策略进行测试
3. 对测试数据进行评估, 根据数据挖掘优化点, 优化策略
4. 在推荐策略符合预期之后, 进行A/B测试
5. 根据A/B测试数据判断是否符合预期, 不符合预期则发掘优化点, 继续优化
6. 符合预期则全量发布
7. 统计发布后的数据, 进入下一轮的策略优化

### 4.推荐策略设计方法(有哪些设计方法)
| user/feature1 | feature2 | feature3 | feature4 | feature5 | feature6 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| user1 | 1 | 1 | 0 | 0 | 0 |
| user2 | 1 | 1 | 1 | 0 | 0 |
| user3 | 1 | 0 | 0 | 0 | 0 |
| user4 | 0 | 1 | 0 | 1 | 0 |
| user5 | 0 | 0 | 0 | 0 | 1 |
#### 4.1 相似性计算
##### 4.1.2 基本相似性计算
| 相似性方法 | 计算方式 | 备注 |
| :---: | :---: | :--- |
| Jaccard | $sim_{Jaccard}(u_i,u_j) = \frac{\|f(u_i) \cap f(u_j)\|}{\|f(u_i) \cup f(u_j)\|}$ | 两个用户的相似性等于<br>两个用户特征中的相同特征数<br>除以两个用户特征的总数<br>$f(u_i)$为$u_i$的特征集合 |
| $L_p-norms$ | $sim_{L_p}(u_i,u_j) = \frac{1}{\sqrt[p]{\frac{1}{n} \Sigma^{n}_{k=1} \|f_{i,k}-f_{j,k}\|^p}+1}$ | 其中$f_i$为用户$u_i$的特征向量, $f_{i,k}$为用户$u_i$的第$j$个特征 |
| Cosine | $sim_{Cosine}(u_i,u_j) = \frac{\vec{f_i} \cdot \vec{f_j}}{\|f_i\| \cdot \|f_j\|}$ | 其中$f_i$为用户$u_i$的特征向量 |

##### 4.1.3 基于结构的相似性计算(SimRank)
<img src='SimRank2.png' style='width:400 px;height: 300px'/>

如果两个人的朋友很相似, 那么这两个人也很相似
$$
sim_{SimRank}(u_i,u_j) = \frac{C}{\|N(u_i)\|\|N(u_j)\|} \Sigma_{m=1}^{\|N(u_i)\|} \Sigma_{n=1}^{\|N(u_j)\|} sim_{SimRank}(u_m,u_n)
$$
其中,$N(u_i)$为用户$u_i$的朋友集合

同理,如果将右侧的节点换成商品,那我们可以通过商品的相似性计算用户的相似性,通过用户的相似性计算商品的相似性

#### 4.2 推荐策略
##### 4.2.1 基于内容的方法
###### 4.2.1.1 context-based modules
<table>
    <tr>
        <td>
            <center>
            <img src='ContextBasedModule.png' style='width:90%;height:90%'/>
            </br>
            <font>基于内容的方法相关模块</font>
            <center>
        </td>
        <td>
            <center>
            <img src='ContextBasedProcess.png' style='width:90%;height:90%'/>
            </br>
            <font>基于内容的方法的推荐过程</font>
            <center>
        </td>
    </tr>
</table>

如左图所示, 基于内容的方法有**三个模块**构成:
1. 内容分析模块(Content analyzer): 这部分主要通过一些特定的分析方法, 为商品(item)生成一系列的特征, 将商品映射到特征空间
2. 用户特征模块(User Profile): 通过用户的历史数据创建用户资料, 并将用户的口味特征映射到特征空间
3. 商品检索模块(Item Retriever): 计算用户口味特征与商品特征的相似性, 检索出相似性高的商品给用户
商品检索过程(推荐过程)如右图所示

基于内容的方法的**优点**:
1. 不需要其它用户的数据，商品没有冷开始(cold)问题和稀疏问题。
2. 能为具有特殊兴趣爱好的用户进行推荐。
3. 能推荐新的或不是很流行的项目，没有新项目问题。
4. 通过列出推荐项目的内容特征，可以解释为什么推荐那些项目。
5. 已有比较好的技术，技术方面比较成熟。

基于内容的方法的**缺点**:
1. 强依赖于特征工程, 特征选的好效果就比较好, 特征选的差效果就比较差
2. 用户的口味特征必须转化到商品特征空间
3. 新用户存在冷启动(cold)问题

目前fleet撮合平台使用的便是这种方法

##### 4.2.2协同过滤方法
| user/item | item1 | item2 | item3 | item4 | item5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| user1 | 1 | 1 | 0 | 0 | 0 |
| user2 | 1 | 1 | 1 | 0 | 0 |
| user3 | 1 | 0 | 0 | 0 | 0 |
| user4 | 0 | 1 | 0 | 1 | 0 |
| user5 | 0 | 0 | 0 | 0 | 1 |
###### 4.2.2.1 基于商品的协同过滤(item-based method)
<img src='ItemBased.png' style='width:400 px;height: 300px'/>

基于商品的协同过滤数据梳理流程(计算用户$i$对商品$j$的喜好):
1. 计算商品$j$与其他所有商品的相似性
2. 选出与$j$相似度比较高的商品
3. 根据用户$i$对这些相似度比较高的商品偏好, 根据下列公式计算出用户$i$对商品$j$的喜好
$$
Pred(user_i,item_j) = \frac{\Sigma_{item_k \in N(item_i)} sim(item_i,item_k) \cdot r_{user_i,item_k}}{\Sigma_{item_k \in N(item_i)} sim(item_k,item_i)}
$$
其中,$N(item_i)$为与用户$item_i$相似性较高的用户集合
###### 4.2.2.2 基于用户的协同过滤(user-based method)
<img src='UserBased.png' style='width:400 px;height: 300px'/>

与基于商品的协同过滤相似, 基于用户的协同过滤数据梳理流程(计算用户$i$对商品$j$的喜好):
1. 计算用户$i$与其他所有用户的相似性
2. 选出与$i$相似度比较高的用户
3. 根据这些相似度比较高的用户对$j$的偏好, 根据下列公式计算出用户$i$对商品$j$的喜好
$$
Pred(user_i,item_j) = \frac{\Sigma_{user_k \in N(user_i)} sim(user_i,user_k) \cdot r_{user_k,item_j}}{\Sigma_{user_k \in N(user_i)} sim(user_i,user_k)}
$$
其中,$N(user_i)$为与用户$user_i$相似性较高的用户集合
###### 4.2.2.3 矩阵分解(matrix factorization method)
<img src='MatrixFactor.png' style='width:400 px;height: 300px'/>

前面介绍了基于商品的协同过滤和基于用户的协同过滤的方法, 但这两种方法不能直接用于生产中,
这是因为商品数量很多, 用户也会很多, 因此实际的user-item矩阵会很大, 计算相似性和检索时的时间复杂度会很高
因此我们需要对商品特征和用户特征进行降维, 
我们知道一个大小为$m \times k$的矩阵与一个$k \times n$的矩阵相乘, 可以得到一个$m \times n$的矩阵
反过来, 一个$m \times n$的矩阵可以分解为,一个大小为$m \times k$的矩阵, 和一个大小为$k \times n$的矩阵(如图所示)

因此, 我们可以把user-item矩阵分解为矩阵$U$和$V$, 其中$U$矩阵可以作为用户特征矩阵参与相似性计算, $V$矩阵可以作为商品特征矩阵参与相似性计算

具体的计算方式有SVD(奇异阵分解)或通过机器学习的方式学习,这里不做详细介绍

###### 4.2.2.4 基于协同过滤的方法的优缺点:
优势:
1. 能够过滤难以进行机器自动内容分析的信息，如艺术品，音乐等。
2. 共享其他人的经验，避免了内容分析的不完全和不精确，并且能够基于一些复杂的，难以表述的概念（如信息质量、个人品味）进行过滤。
3. 有推荐新信息的能力。可以发现内容上完全不相似的信息，用户对推荐信息的内容事先是预料不到的。这也是协同过滤和基于内容的过滤一个较大的差别，基于内容的过滤推荐很多都是用户本来就熟悉的内容，而协同过滤可以发现用户潜在的但自己尚未发现的兴趣偏好。
4. 能够有效的使用其他相似用户的反馈信息，较少用户的反馈量，加快个性化学习的速度。

劣势:
1. 新用户和新商品均存在冷启动问题(cold start)和稀疏问题
2. 计算复杂性较高

### 5.策略评估方法(如何判断当前策略的好坏?)
#### 5.1 基础指标
| 数据指标 | 计算方式 | 描述 |
| :--: | :---: | :-- |
| 多样性(Diversity) | $div(u_i) = 1 - \frac{\Sigma_{m \in R(u_i)} \Sigma_{n \in R(u_i)} sim(m,n)}{\|R(u_i)\|\|R(u_i)\|}$ | 用于衡量推荐给用户的商品是否同质率过高<br>其中$R(u_i)$为推荐给$u_i$的商品 |
| 覆盖率(Coverage) | $cove = \frac{曝光的商品数}{所有商品数}$ | 用于衡量曝光的商品是否只占所有商品的很小一部分 |
| 感兴趣度 | 详见第4.2部分 | 用于衡量用户对推荐给它的商品的喜好程度 |
#### 5.2 准确率与召回率(感兴趣度)
##### 5.2.1 TP矩阵
| | Recommend | Not Recommend |
| :---: | :---: | :---: |
| Consumed | True Positive | False Negative |
| Not Consumed | False Positive | True Negative |

##### 5.2.2 衡量标准
| 数据指标 | 计算方式 | 计算维度 | 描述 |
| :--: | :---: | :---: | :-- |
| 准确率(Precision) | $precison = \frac{TP}{TP+FP}$ | 用户维度 | 用于衡量推荐的商品中, 有多少是用户感兴趣的 |
| 召回率(Recall) | $recall = \frac{TP}{TP+FN}$ | 用户维度 | 用于衡量用户感兴趣的商品, 推荐策略推荐了多少 |
| P@k(Precision@k) | $P@k(u_i) = \frac{前k个推荐u_i感兴趣数}{k}$ | 用户维度 | 用于衡量前k个推荐中,用户$u_i$的感兴趣度 |
| 平均准确率(AP) | $AP(u_i) = \frac{\Sigma_{k=i}^m P@(k)}{m}$ | 用户维度 | Average Precision<br>用于用户$u_i$衡量平均准确率|
| 平均准确率(MAP) | $MAP(u_i) = \frac{\Sigma_{u \in U} AP(u)}{\|U\|}$ | 全局维度 | Mean Average Precision<br>用于衡量整体推荐策略的情况 |
| 累计收益(CG) | $CG_p = \Sigma_{i=1}^p rel(i)$ | 用户维度 | Cumulative Gain<br>$rel(i)$指用户对<br>第$i$个推荐的商品的感兴趣程度 |
| 贴现累计收益(DCG) | $DCG_p = \Sigma_{i=1}^p \frac{2^{rel(i)} - 1}{log_2(i+1)}$ | 用户维度 |Discounted Cumulative Gain<br>按照排序对相关度进行相应的缩减 |
| 归一化累计收益(NDCG) | $NDCG_p = \frac{DCG_p}{IDCG_p}$ | 用户维度 | Normalized Discounted<br>Cumulative Gain<br>对DCG的归一化处理,方便不同策略的对比<br>IDCG(idea DCG)指理想的DCG,<br>即把商品按用户感兴趣程度排序后的DCG |
