#!/usr/bin/env python
# coding: utf-8

# # 第三周作业--数据探索性分析与数据预处理

# # 1，第一个数据的处理--Wine Reviews数据集
# ## 1.1 数据可视化和摘要
# ### 1.1.1数据摘要

# 经过分析可知，数据集的两块内容中，country与province，region有一定的重复，因此只处理country列的数据；
# description和designation两列数据内容太杂乱，在数据可视化阶段不考虑；
# variety和winery并无一一对应关系，但是有隐含的内在关系，因此选择variety进行处理；
# 数据price和points有重大的参考价值，需要重点处理。

# 

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import csv
import xlrd as xl
import numpy as np
import matplotlib.pylab
get_ipython().run_line_magic('matplotlib', 'inline')

xls_file150 = xl.open_workbook("F:\\PYTHON\\pythonProject12\\venv\\winemag-data_first150k.xlsx")
xls_sheet1 = []  # 定义一个列表
xls_sheet1.append(xls_file150.sheets()[0].col_values(1))  # 取第一个表格，取country那一列
for i in range(1):
    del xls_sheet1[0][0]  # 删掉前1行
arr = np.array(xls_sheet1)  # 转换成array
key = np.unique(xls_sheet1)  # x轴，得到的也是array对象
result = {}
for k in key:
    mask = (arr == k)
    arr_new = arr[mask]
    v = arr_new.size
    result[k] = v
# result是类似于这样的字典：{us：15，astrilia：23.....}
countryList = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(countryList)


# countryList统计了country这个标签下所有可能的取值，
# 并将每个取值及其出现的频率设置为一个字典放在一个列表之中并输出出来，如上所示
# 

# In[3]:


with open('F:\\PYTHON\\pythonProject12\\venv\\winemag-data_first150k.csv','rt',encoding='UTF-8') as csvfile:
    reader = csv.DictReader(csvfile)
    xls_sheet2 = [row['variety'] for row in reader]
# print(xls_sheet2)
# xls_sheet2.append(xls_file150.sheets()[0].col_values(9))  # 取第一个表格，取variety那一列
# for i in range(1):
#     del xls_sheet2[0]  # 删掉前1行
arr = np.array(xls_sheet2)  # 转换成array
key = np.unique(xls_sheet2)  # x轴，得到的也是array对象
result = {}
for k in key:
    mask = (arr == k)
    arr_new = arr[mask]
    v = arr_new.size
    result[k] = v
# result是字典
varietyList = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(varietyList)


# 为了解决乱码问题，采用直接读取csv源文件的格式得到数据。
# varietyList统计了variety这个标签下所有可能的取值，
# 并将每个取值及其出现的频率设置为一个字典放在一个列表之中并输出出来，
# 如上所示，country和variety两个标签属性分析完毕

# In[4]:


data01 = pd.read_csv('F:\\PYTHON\\pythonProject12\\venv\\winemag-data_first150k.csv')
priceList = data01['price'].values
pointsList = data01['points'].values
def fiveNumber(nums):#自定义能返回五数概况的数据
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.percentile(nums,25)
    Median=np.median(nums)
    Q3=np.percentile(nums,75)
    IQR=Q3-Q1
    return Minimum,Q1,Median,Q3,Maximum
print(fiveNumber(priceList))
print(fiveNumber(pointsList))


# 不难看出，由于数据集有空值，
# 对该数据集求Q1，Q3和中位数时得到了空值，
# 因此需要进行空值过滤在求值，修改如下：

# In[20]:


data01 = pd.read_csv('F:\\PYTHON\\pythonProject12\\venv\\winemag-data_first150k.csv')
data01notnull = data01.loc[data01['price'].notna()]#除去空值的数据得到新的数据集
data012notnull = data01.loc[data01['points'].notna()]
priceFull = data01['price'].values
pointsFull = data01['points'].values
priceList = data01notnull['price'].values#将price数据取出来方便计算
pointsList = data01['points'].values#将points数据取出来方便计算
pointsList01 = data012notnull['points'].values
def fiveNumber(nums):#自定义能返回五数概况的数据
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.percentile(nums,25)
    Median=np.median(nums)
    Q3=np.percentile(nums,75)
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR #下限值
    upper_limit=Q3+1.5*IQR #上限值
    return Minimum,Q1,Median,Q3,Maximum
print("price的五数概况如下：")
print(fiveNumber(priceList))
print("price的缺失数据为：")
print(priceFull.shape[0] - priceList.shape[0])
print("points的五数概况如下：")
print(fiveNumber(pointsList))
print("points的缺失数据为：")
print(pointsFull.shape[0] - pointsList01.shape[0])


# 由此即得到了数据集的五数概况及两者缺乏的数据

# ### 1.1.2 数据可视化

# 首先对上面得到的数据进行可视化，
# 对country和variety的数据绘制直方图得到红酒的国家分布和高分红酒的种类分布

# In[21]:


x = []  # 取国家的那一列
y = []  # 取数字的那一列
for item in countryList:#考虑到图片位置有限，把国家出现频率少于100的筛选掉
    if item[1] >= 100:
        y.append(item[1])
        x.append(item[0])

plt.figure(dpi=300,figsize=(16,9))
plt.bar(x, y, 0.5, align='center')  # 画图，设置x，y轴的数据
# fig,ax = plt.subplots(figsize=(15,9),dpi=256)  #单位为英寸，1英寸约为2.54cm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
for x, y in zip(x, y):
    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom')
plt.xticks(rotation=45)  # rotation设置x轴标签的旋转度数

plt.xlabel("country")
plt.ylabel("frequency")

plt.show()


# 对country列得到的数据画柱状图如上所示，
# 横坐标为国家，纵坐标为对应国家出现的频数，
# 为了让数据图更加清晰，将出现次数少于100的国家剔除

# In[24]:


x = []  # 取国家的那一列
y = []  # 取数字的那一列
for item in varietyList:#考虑到图片位置有限，把品牌出现频率少于1000的筛选掉
    if item[1] >= 1000:
        y.append(item[1])
        x.append(item[0])

plt.figure(dpi=300,figsize=(16,9))
plt.bar(x, y, 0.5, align='center')  # 画图，设置x，y轴的数据
# fig,ax = plt.subplots(figsize=(15,9),dpi=256)  #单位为英寸，1英寸约为2.54cm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
for x, y in zip(x, y):
    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom')
plt.xticks(rotation=90)  # rotation设置x轴标签的旋转度数

plt.xlabel("variey")
plt.ylabel("frequency")

plt.show()


# 从之前的数据不难看出，
# variety列的数据更加分散，
# 为了有更好的显示效果和参考价值，
# 将出现频率少于1000的品牌剔除得到了上图

# In[32]:


plt.boxplot(data01notnull['price'])
plt.show()


# 由于缺少数据无法绘制盒图，
# 因此默认使用剔除了空值的数据绘制price列的盒图如上所示，
# 可疑的离群点也一同绘制出来（即图中空心点）

# In[30]:


plt.boxplot(data01['points'])
plt.show()


# points不缺少数据，
# 直接绘制盒图如上所示，
# 可疑的离群点也一同绘制出来（即图中空心点），
# 两个图数据差别太大，不宜绘制在一个图中。

# ## 1.2 数据缺失的处理

# 经分析可知，分析的country，variety，price和points四组数据中，只有price数据有空缺，需要处理，
# 下面根据四种方案来对price进行处理

# ### 1.2.1 将缺失部分剔除
# 

# 由于plt.boxplot函数在有空值的情况下无法绘制盒图，
# 因此上面绘制盒图默认采取了将空值剔除的办法，
# 现在将有空值的原始数据和没有空值的情况分别绘制对比如下

# In[33]:


# plt.boxplot(data01['price'])
# plt.show()


# 有空值，无法绘制盒图，如上所示

# In[34]:


plt.boxplot(data01notnull['price'])
plt.show()


# 直接剔除空值得到的盒图如上所示

# ### 1.2.2 用最高频率值来填补缺失值
# 

# In[58]:


price1 = data01['price']
price = price1.tolist()
most = max(set(price),key=price.count) 
print(most)


# price中出现频数最多的元素为20.0

# In[57]:


price2 = price1.fillna(20.0)#用20.0来替换空值
price = price2.tolist()
plt.boxplot(price)
plt.show()


# 不难看出，用20.0填充空值后可以直接做出盒图，
# 但是由于可疑离群点太多，这个图无法进行对比，
# 因此需要舍弃一些离群点后再进行分析

# In[64]:


x = []
for i in range(0,len(price)):
    if price[i] <= 60:#舍弃数值大于60的点
        x.append(price[i])

plt.boxplot(x)
plt.show()


# 将price大于65的离群点抛弃后重新绘制盒图得到了以上结果。

# In[65]:


print("空值填充后的price的五数概况如下：")
print(fiveNumber(price))


# 未填充price的五数概况如下：
# (4.0, 16.0, 24.0, 40.0, 2300.0)，不难看出中位数和Q3均减小

# In[54]:


y = []
price3 = data01notnull['price'].values
for i in range(len(price3)):
    if price3[i] <= 70:#舍弃数值大于70的点
        y.append(price3[i])

plt.boxplot(y)
plt.show()


# 从直接舍弃空值的数据集的盒图不难看出，
# 直接舍弃的做法的盒图的中值和Q3更大，这与上面的数据计算一致

# ### 1.2.3通过属性的相关关系来填补缺失值

# 根据数据集分析得到，points和price关系更强，
# 因此采用这两者的相关关系来得到拟合曲线进而根据曲线来通过完整的points填充price缺失值

# In[53]:



#画出散点图
import pylab
x = []
y = []
x= data01notnull['points'].values
y = data01notnull['price'].values
# plt.scatter(data01['points'],data01['price'])
# pylab.plot(x,y,'o')
parameter = np.polyfit(x, y, 3)
y2 = parameter[0] * x ** 3 + parameter[1] * x ** 2 + parameter[2] * x + parameter[3]#假设的方程及参数
plt.scatter(x, y)
plt.plot(x, y2, color='g')
plt.show()
correlation = np.corrcoef(y, y2)[0,1]  #相关系数
correlation**2
p = np.poly1d(parameter,variable='x')#求出拟合公式
print(p)


# 3次方的图像拟合效果比较好，采用3次方的方程来作为points和price的拟合方程来填充空白值

# In[6]:


x = []
y = []
x= data01['points'].values
y = data01['price'].values
import math
for i in range(len(y)):
    if math.isnan(y[i]):#判断price是否为None
        y[i] = 0.0794 * x[i] ** 3 - 20.24 * x[i] ** 2 + 1721 * x[i] - 4.876e+04#根据得到的公式将price缺值填充
print(y[3493])      


# In[7]:


print("根据属性相关填充后的price的五数概况如下：")
print(fiveNumber(y))
print("未填充price的五数概况如下：(4.0, 16.0, 24.0, 40.0, 2300.0)")


# 由于填充拟合的曲线增加了数据集的值，因此填充后的price的中位数和Q3增大，由于Q1前空值较少，因此Q1并未改变

# In[66]:


z = []
for i in range(len(y)):
    if y[i] <= 80:#舍弃数值大于80的点
       z.append(y[i])

plt.boxplot(z)
plt.show()


# 很明显，根据元素相关关系填充得到的数据集Q3明显增大（超过40），
# 上确界也明显变大（大于70），这种填充方式会改变数据分布，产生难以忽略的误差

# ### 1.2.4通过数据对象之间的相似性来填补缺失值

# 根据对象相似性来分析，设定points对price影响更大，
# 若是相近两个数据的points相同，则让他们的price也相同；
# 若points不相同，variety相同，则令两者的price相同；
# 最后，若前两个均不相同但是country相同，则令两者price相同。
# 每个标签比较时，只寻找前面的20条记录的相关属性来填充price，若不满足条件则换成别的标签继续比较。
# 若是均不满足则根据1.2.3中的points的值来比较

# In[44]:


z = []
z = data01['variety'].values
c = data01['country'].values
for i in range(0, len(y)):
    if math.isnan(y[i]):  # 判断price是否为None
        for j in range(i - 20, i):
            if x[j] == x[i]:
                y[i] = y[j]
                break
        if math.isnan(y[i]):
            for j in range(i - 20, i):
                if z[j] == z[i]:
                    y[i] = y[j]
                    break
        if math.isnan(y[i]):
            for j in range(i - 20, i):
                if c[j] == c[i]:
                    y[i] = y[j]
                    break
        if math.isnan(y[i]):
            y[i] = 0.0794 * x[i] ** 3 - 20.24 * x[i] ** 2 + 1721 * x[i] - 4.876e+04        


# In[67]:



print("根据属性相关填充后的price的五数概况如下：")
print(fiveNumber(y))
print("未填充price的五数概况如下：(4.0, 16.0, 24.0, 40.0, 2300.0)")


# 通过数据对象的相似性进行元素填充的得到的数据如上所示，
# 根据对象数据进行填充效果明显，
# 填充后的数据变化明显，Q1，中位数和Q3均变化明显，
# 都有所减小，因为填充的数据和附近对象的数据类似，可能与数据缺少的值分布再前半部分有关

# In[68]:


z = []
for i in range(len(y)):
    if y[i] <= 80:#舍弃数值大于80的点
       z.append(y[i])

plt.boxplot(z)
plt.show()


# In[ ]:




