#!/usr/bin/env python
# coding: utf-8

# # 第三周作业--数据探索性分析与数据预处理

# # 2，第一个数据的处理--Trending YouTube Video Statistics数据集
# ## 2.1 数据可视化和摘要
# ### 2.1.1数据摘要

# 选择美国的youtube数据进行分析。
# 经分析可知，
# 该数据集的channel_title，category_id，views，likes，dislikes和comment_count具有分析价值
# 可以得到视频网站热门视频的种类，观看量和喜好变化等，故选择这些数据进行分析和预处理。

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import csv
import xlrd as xl
import numpy as np
import matplotlib.pylab
get_ipython().run_line_magic('matplotlib', 'inline')

with open('E:\\1\\Databaseyoutube\\USvideos.csv','rt',encoding='UTF-8') as csvfile:
    reader = csv.DictReader(csvfile)
    xls_sheet2 = [row['channel_title'] for row in reader]

arr = np.array(xls_sheet2)  # 转换成array
key = np.unique(xls_sheet2)  # x轴，得到的也是array对象
result = {}
for k in key:
    mask = (arr == k)
    arr_new = arr[mask]
    v = arr_new.size
    result[k] = v
# result是字典
titleList = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(titleList)


# channel_title是唯一一个标称属性，
# 统计视频标题及其出现的频率可以看出热门视频背后的频道，
# 这有助于帮助分析哪些视频频道是最热门的频道，
# 也说明其具有很高的商业价值。从上面统计的数据可以看出。
# ESPN是热门视频中出现次数最多的频道，其内容，运营都值得参考和借鉴。

# In[2]:


with open('E:\\1\\Databaseyoutube\\USvideos.csv','rt',encoding='UTF-8') as csvfile:
    reader = csv.DictReader(csvfile)
    xls_sheet2 = [row['category_id'] for row in reader]

arr = np.array(xls_sheet2)  # 转换成array
key = np.unique(xls_sheet2)  # x轴，得到的也是array对象
result = {}
for k in key:
    mask = (arr == k)
    arr_new = arr[mask]
    v = arr_new.size
    result[k] = v
# result是字典
categoryList = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(categoryList)


# 由数据集附带的json文件可知，
# category_id代表的视频类型：
# "id": "1","title": "Film & Animation"，"id": "2","title": "Autos & Vehicles",
# "id": "10","title": "Music","id": "15","title": "Pets & Animals","id": "17",
# "title": "Sports","id": "18","title": "Short Movies","id": "19","title": "Travel & Events",
# "id": "20","title": "Gaming","id": "21", "title": "Videoblogging","id": "22","title": "People & Blogs",
# "id": "23", "title": "Comedy","id": "24","title": "Entertainment","id": "25","title": "News & Politics",
# "id": "26","title": "Howto & Style","id": "27","title": "Education",
# "id": "28","title": "Nonprofits & Activism","id": "30","title": "Movies",
# "id": "31","title": "Anime/Animation","id": "32","title": "Action/Adventure",
# "id": "33","title": "Classics","id": "34","title": "Comedy","id": "35","title": "Documentary",
# "id": "36","title": "Drama","id": "37","title": "Family","id": "38", "title": "Foreign",
# "id": "39","title": "Horror","id": "40","title": "Sci-Fi/Fantasy","id": "41","title": "Thriller",
# "id": "42","title": "Shorts","id": "43","title": "Shows","id": "44","title": "Trailers",

# category_id为视频的种类，
# 上述得到了视频种类及每种的出现频率。
# 根据上面的数据可以看出，最热门的三种视频的种类为24：
# Entertainment；10：Music；26：Howto & Style，是大家最喜欢的视频内容，
# 由于category_id为视频的种类，因此进行五数概况分析并没有意义，故不作分析。

# In[4]:


data01 = pd.read_csv('E:\\1\\Databaseyoutube\\USvideos.csv')#views，likes，dislikes和comment_count
viewsList = data01['views'].values
likesList = data01['likes'].values
dislikesList = data01['dislikes'].values
commentList = data01['comment_count'].values
viewsFull = data01.loc[data01['views'].notna()]#除去views空值的数据得到新的数据集
likesFull = data01.loc[data01['likes'].notna()]#除去likes空值的数据得到新的数据集
dislikesFull = data01.loc[data01['dislikes'].notna()]#除去dislikes空值的数据得到新的数据集
commentFull = data01.loc[data01['comment_count'].notna()]#除去views空值的数据得到新的数据集

def fiveNumber(nums):#自定义能返回五数概况的数据
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.percentile(nums,25)
    Median=np.median(nums)
    Q3=np.percentile(nums,75)
    IQR=Q3-Q1
    return Minimum,Q1,Median,Q3,Maximum
def countzero(nums):#用来统计每个列表元素值为0的个数
    count = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            count += 1
    return count        
print("views的五数概况如下：")
print(fiveNumber(viewsList))
print("likes的五数概况如下：")
print(fiveNumber(likesList))
print("dislikes的五数概况如下：")
print(fiveNumber(dislikesList))
print("comment_count的五数概况如下：")
print(fiveNumber(commentList))
print("views的缺失数据为：")
print(viewsFull.shape[0] - viewsList.shape[0])
print("likes的缺失数据为：")
print(likesFull.shape[0] - likesList.shape[0])
print("dislikes的缺失数据为：")
print(dislikesFull.shape[0] - dislikesList.shape[0])
print("comment的缺失数据为：")
print(commentFull.shape[0] - commentList.shape[0])
print("likes值为0的数据为：")
print(countzero(likesList))
print("dislikes值为0的数据为：")
print(countzero(dislikesList))
print("comment值为0的数据为：")
print(countzero(commentList))


# views，likes，dislikes和comment_count的五数概况和缺失数据都在上面给出，
# 可以看出，四个数据均没有缺失数据，但是likes，dislikes和comment_count三个数据存在最小值0，
# 这和实际情况不符合，因此，在预处理阶段将这三个数据属性的0值当作缺失值来处理。

# ### 2.1.2 数据可视化

# 首先对上面得到的数据进行可视化，对channel_title和category_id的数据绘制直方图得到热门视频的频道分布和种类分布

# In[25]:


x = []  # 取频道的那一列
y = []  # 取数字的那一列
for item in titleList:#考虑到图片位置有限，把频道出现频率少于100的筛选掉
    if item[1] >= 140:
        y.append(item[1])
        x.append(item[0])
plt.figure(dpi=100,figsize=(16,9))
plt.bar(x, y, 0.5, align='center')  # 画图，设置x，y轴的数据
# fig,ax = plt.subplots(figsize=(15,9),dpi=256)  #单位为英寸，1英寸约为2.54cm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
for x, y in zip(x, y):
    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom')
plt.xticks(rotation=90)  # rotation设置x轴标签的旋转度数

plt.xlabel("channel")
plt.ylabel("frequency")

plt.show()


# 对channel_title列得到的数据画柱状图如上所示，
# 横坐标为频道，纵坐标为对应频道出现的频数，为了让数据图更加清晰，将出现次数少于140的国家剔除

# In[27]:


x = []  # 取频道的那一列
y = []  # 取数字的那一列
for item in categoryList:#考虑到图片位置有限，把频道出现频率少于100的筛选掉
#     if item[1] >= 140:
    y.append(item[1])
    x.append(item[0])
plt.figure(dpi=100,figsize=(16,9))
plt.bar(x, y, 0.5, align='center')  # 画图，设置x，y轴的数据
# fig,ax = plt.subplots(figsize=(15,9),dpi=256)  #单位为英寸，1英寸约为2.54cm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
for x, y in zip(x, y):
    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom')
plt.xticks(rotation=0)  # rotation设置x轴标签的旋转度数

plt.xlabel("category")
plt.ylabel("frequency")

plt.show()


# 可以看出热门视频中种类为：
# Entertainment，Music，Howto & Style，Comedy，People & Blogs，
# News & Politics，Nonprofits & Activism，Film & Animation，Sports，Education，
# Pets & Animals，Gaming，Travel & Events，Autos & Vehicles，Nonprofits & Activism，Shows

# In[28]:



plt.boxplot(viewsList)
plt.show()


# 不难看出可疑离群点很多，为了还原原始数据的特征不进行处理，
# 后面填充缺少的数据会剥离部分离群点重新绘制盒图。

# In[29]:


plt.boxplot(likesList )
plt.show()


# 直接绘制盒图如上所示，可疑的离群点也一同绘制出来（即图中空心点）

# In[30]:


plt.boxplot(dislikesList)
plt.show()


# 直接绘制盒图如上所示，可疑的离群点也一同绘制出来（即图中空心点）

# In[31]:


plt.boxplot(commentList)
plt.show()


# 直接绘制盒图如上所示，可疑的离群点也一同绘制出来（即图中空心点）

# ## 1.2 数据缺失的处理

# 经分析可知，likes，dislikes和comment_count数据有0值，即空缺，需要处理，
# 由于dislike数据与likes数据有分析意义上的重复，故只分析likes数据。
# 下面根据四种方案来对likes和comment_count进行处理

# ### 1.2.1 将缺失部分剔除

# #### 对likes数据进行处理

# In[71]:


# likesFull
# dislikesFull 
# commentFull

x = []
for i in range(len(likesList)):
    if likesList[i] != 0:
        x.append(likesList[i])
print("剔除缺失值后的likes的五数概况如下：")
print(fiveNumber(x))
    


# 未处理的likes数据：
# likes的五数概况如下：
# (0, 5424.0, 18091.0, 55417.0, 5613827)
# 剔除0值之后的likesd的五数概况中Q1略微增大，
# 中位数减小，Q3略微增大，这说明含零值不多,这与上面显示的172个数据相吻合

# 画出未处理likes数据的盒图，过滤掉部分离群点方便进行对比

# In[96]:


x = []
for i in range(len(likesList)):
    if likesList[i] <= 80000:#剔除掉超过80000的数据
        x.append(likesList[i])
plt.boxplot(x)
plt.show()


# likes最大的数据5613827，Q3为55417.0，
# 显然likes标签的离群点很多范围也大，因此处理原始数据的时候剔除掉超过80000的数据，
# 虽然会造成数据漂移，但是后面分析沿用统一标准，就仍有参考价值。

# In[103]:


x = []
for i in range(len(likesList)):
    if likesList[i] > 0 and likesList[i] <= 80000 :#剔除掉超过80000的数据
        x.append(likesList[i])
plt.boxplot(x)
plt.show()


# 剔除了元素值为0的点后，可以看出变化不大

# #### 对comment数据进行处理

# 画出未处理commentList数据的盒图，过滤掉部分离群点方便进行对比

# In[16]:


x = []
for i in range(len(commentList)):
    if commentList[i] <= 10000:#剔除掉超过10000的数据和等于零的数据
        x.append(commentList[i])
plt.boxplot(x)
plt.show()


# In[17]:


x = []
for i in range(len(commentList)):
    if commentList[i] != 0:
        x.append(commentList[i])
print("剔除缺失值后的comment的五数概况如下：")
print(fiveNumber(x))


# comment_count未处理的五数概况如下：
# (0, 614.0, 1856.0, 5755.0, 1361580)，
# 剔除零值后Q1，中位数，Q3的个数均增大这和上面的分析一致，下面的盒图也体现了这个趋势

# In[19]:


x = []
for i in range(len(commentList)):
    if commentList[i] > 0 and commentList[i] <= 10000 :#剔除掉超过10000的数据和等于零的数据
        x.append(commentList[i])
plt.boxplot(x)
plt.show()


# ### 2.2.2 用最高频率值来填补缺失值

# #### 对likes数据进行处理

# In[20]:



likes = likesList.tolist()
most = max(set(likes),key=likes.count) 
print(most)


# likes数据中0值最多，为了填充零值，需要先剔除零值得到频率最高的数来替代

# In[22]:


x = []
for i in range(len(likesList)):
    if likesList[i] != 0:
        x.append(likesList[i])
most = max(set(x),key=x.count) 
print(most)       


# 用2来填充0值

# In[28]:


x = []
for i in range(len(likesList)):
    if likesList[i] != 0:
        x.append(likesList[i])
    else:
        x.append(2)
print("填充缺失值后的likes的五数概况如下：")
print(fiveNumber(x))        


# 未处理的likes数据： likes的五数概况如下：
# (0, 5424.0, 18091.0, 55417.0, 5613827) 填充0值之后的likesd的五数概况不变，
# 因为填充后的2在Q1前，不影响整体的数据

# In[29]:


y = []        
for i in range(len(x)):
    if x[i] <= 80000:#剔除掉超过80000的数据
        y.append(x[i])
plt.boxplot(y)
plt.show() 


# #### 对comment数据进行处理

# In[31]:


x = []
for i in range(len(commentList)):
    if commentList[i] != 0:
        x.append(commentList[i])
most = max(set(x),key=x.count) 
print(most)  


# commentList最多的数据是0，剔除0后得到最多的数据是1，再用1填充0值

# In[32]:


x = []
for i in range(len(commentList)):
    if commentList[i] != 0:
        x.append(commentList[i])
    else:
        x.append(1)
print("填充缺失值后的comment的五数概况如下：")
print(fiveNumber(x)) 


# comment_count未处理的五数概况如下： (0, 614.0, 1856.0, 5755.0, 1361580)，用1替换零值后五数概况不变

# In[33]:


y = []
for i in range(len(x)):
    if x[i] > 0 and x[i] <= 10000 :#剔除掉超过10000的数据和等于零的数据
        y.append(x[i])
plt.boxplot(y)
plt.show()


# ### 2.2.3 通过属性的相关关系来填补缺失值

# 分析认为,视频的观看量views对likes数据和comment数据影响最大，
# 采用视频观看量views数据对likes和comments数据进行拟合

# #### 对likes数据进行处理

# In[49]:


#画出散点图
import pylab
x = []
y = []
x = data01['views'].values
y = data01['likes'].values
parameter = np.polyfit(x, y, 2)
y2 = parameter[0] * x ** 2+ parameter[1] * x + parameter[2] #假设的方程及参数
plt.scatter(x, y)
plt.plot(x, y2, color='g')
plt.show()
correlation = np.corrcoef(y, y2)[0,1]  #相关系数
correlation**2
p = np.poly1d(parameter,variable='x')#求出拟合公式
print(p)


# In[51]:



for i in range(len(y)):
    if y[i] == 0:#判断likes是否为0
        y[i] = -2.752e-11 * x[i] ** 2 + 0.02919 * x[i] + 7515#根据得到的公式将缺值填充
print("填充缺失值后的likes的五数概况如下：")
print(fiveNumber(y))       


# 未处理的likes数据： likes的五数概况如下：
# (0, 5424.0, 18091.0, 55417.0, 5613827) 填充0值之后
# ,经分析2次方拟合的效果最好，填充后的数据集五数概况变化不大

# In[52]:


z = []        
for i in range(len(y)):
    if y[i] <= 80000:#剔除掉超过80000的数据
        z.append(y[i])
plt.boxplot(z)
plt.show() 


# #### 对comment数据进行处理

# In[58]:


y = []
y = data01['comment_count'].values
parameter = np.polyfit(x, y, 3)
y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]#假设的方程及参数
plt.scatter(x, y)
plt.plot(x, y2, color='g')
plt.show()
correlation = np.corrcoef(y, y2)[0,1]  #相关系数
correlation**2
p = np.poly1d(parameter,variable='x')#求出拟合公式
print(p)


# In[66]:


for i in range(len(y)):
    if y[i] == 0:#判断likes是否为0
        y[i] = -1.047e-19 * x[i] ** 3 + 2.225e-11 * x[i] ** 2 + 0.002423 * x[i] + 1945#根据得到的公式将缺值填充
print("填充缺失值后的comment的五数概况如下：")
print(fiveNumber(y))     


# comment_count未处理的五数概况如下：
# (0, 614.0, 1856.0, 5755.0, 1361580)，用属性间的相似性来填充0值之后,
# 经分析3次方拟合的效果最好，填充后的数据集五数概况变化不大

# In[60]:


z = []        
for i in range(len(y)):
    if y[i] <= 10000:#剔除掉超过80000的数据
        z.append(y[i])
plt.boxplot(z)
plt.show() 


# #### 1.2.4通过数据对象之间的相似性来填补缺失值

# 根据数据对象的视频种类，
# 频道的相似性来填充likes和comment数据，若是都不满足，
# 则根据两者与views的拟合关系来填充

# #### 对likes数据进行处理

# In[63]:


z = []
c = []
y = []
y = data01['likes'].values
z = data01['channel_title'].values
c = data01['category_id'].values
for i in range(0, len(y)):
    if y[i] == 0:  # 判断likes是否为0
        for j in range(i - 20, i):
            if z[j] == z[i]:
                y[i] = y[j]
                break
        if y[i] == 0:
            for j in range(i - 20, i):
                if c[j] == c[i]:
                    y[i] = y[j]
                    break
        if y[i] == 0:
            y[i] = -2.752e-11 * x[i] ** 2 + 0.02919 * x[i] + 7515
print("填充缺失值后的likes的五数概况如下：")
print(fiveNumber(y))  


# 未处理的likes数据： likes的五数概况如下：
# (0, 5424.0, 18091.0, 55417.0, 5613827) 根据对象间的相似性填充0值之后，
# 填充后的数据集五数概况变化不大，效果良好

# In[64]:


h = []        
for i in range(len(y)):
    if y[i] <= 80000:#剔除掉超过80000的数据
        h.append(y[i])
plt.boxplot(h)
plt.show() 


# #### 对comment数据进行处理

# In[65]:



y = []
y = data01['comment_count'].values
for i in range(0, len(y)):
    if y[i] == 0:  # 判断comment是否为0
        for j in range(i - 20, i):
            if z[j] == z[i]:
                y[i] = y[j]
                break
        if y[i] == 0:
            for j in range(i - 20, i):
                if c[j] == c[i]:
                    y[i] = y[j]
                    break
        if y[i] == 0:
            y[i] = -1.047e-19 * x[i] ** 3 + 2.225e-11 * x[i] ** 2 + 0.002423 * x[i] + 1945
print("填充缺失值后的comments的五数概况如下：")
print(fiveNumber(y)) 


# comment_count未处理的五数概况如下：
# (0, 614.0, 1856.0, 5755.0, 1361580)，
# 用属性间的相似性来填充0值之后,经分析3次方拟合的效果最好，
# 填充后的数据集五数概况变化不大

# In[68]:


h = []        
for i in range(len(y)):
    if y[i] <= 10000:#剔除掉超过10000的数据
        h.append(y[i])
plt.boxplot(h)
plt.show() 


# In[ ]:




