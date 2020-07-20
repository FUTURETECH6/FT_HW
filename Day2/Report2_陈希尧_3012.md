<center style = "font-size: 4em">金融科技导论实验报告</center><br/><br/><br/><br/>

**姓名**：<u>陈希尧</u>

**学号**：<u>3180103012</u>

**专业**：<u>计算机科学与技术</u>

**课程名称**：<u>金融科技导论</u>

<center style = "font-size: 1.7em">Table of Contents</center>

[TOC]

# 环境配置

检查需要的包的安装情况

<img src="assets/image-20200715110824671.png" style="zoom: 25%;" />

Notebook使用VS Code原生的内嵌Jupyter环境，其中的Jupyter Notebook使用pip安装而非conda自带。

# 实验步骤

## 导入相关包

其中matplotlib和seaborn是用于数据可视化的

<img src="assets/image-20200715213909431.png" style="zoom: 15%;" />

## 数据读取

### 读取原始训练集

<img src="assets/image-20200715213937561.png" style="zoom: 15%;" />

### 读取数据集属性信息

<img src="assets/image-20200715213956603.png" style="zoom: 15%;" />

### 整合数据集的行列信息

<img src="assets/image-20200715214029065.png" style="zoom: 15%;" />

## 观察测试集

### 缺失情况

<img src="assets/image-20200715214127831.png" style="zoom: 15%;" />

<img src="assets/image-20200715214141343.png" style="zoom: 15%;" />

### 数据特征(关联性可视化)

<img src="assets/image-20200715214244182.png" style="zoom: 15%;" />

<img src="assets/image-20200715214343185.png" style="zoom: 15%;" />

效果如下：

<img src="./assets/plt.svg" style="zoom: 20%;" >

## 缺失数据处理

注意：在修改测试集时不能调用测试集本身的数据

### 默认值填充

<img src="assets/image-20200715214702233.png" style="zoom: 15%;" />

### 平均值填充

<img src="assets/image-20200715214752234.png" style="zoom: 15%;" />

### 删除不完整的行

<img src="assets/image-20200715214917522.png" style="zoom: 15%;" />

<img src="assets/image-20200715214933860.png" style="zoom: 15%;" />

## 数据变化与离散化

### 缩放

<img src="assets/image-20200715215031677.png" style="zoom: 15%;" />

<img src="assets/image-20200715215055879.png" style="zoom: 15%;" />

### 规范化

<img src="assets/image-20200715215132708.png" style="zoom: 15%;" />

### 离散化

#### 等深分箱

<img src="assets/image-20200715215206963.png" style="zoom: 15%;" />

#### 等宽分箱

<img src="assets/image-20200715215228424.png" style="zoom: 15%;" />

## 特征构造

先看原数据

<img src="assets/image-20200715215305671.png" style="zoom: 15%;" />

实现特征交叉方法并查看结果

<img src="assets/image-20200715215358387.png" style="zoom: 15%;" />

## 数据集切分

<img src="assets/image-20200715215434174.png" style="zoom: 15%;" />

# 实验心得

经过本次实验，我初步窥探得了数据预处理的基本步骤，学习了Pandas，Numpy等数据分析包的使用，并巩固了对mtaplotlib和seaborn可视化包的掌握。相信本次实验对之后的学习很有帮助。