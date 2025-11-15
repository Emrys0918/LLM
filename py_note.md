# Python 函数

## 1. `assert()`: assert函数在Python中是一个非常有用的调试工具。它的基本作用是在代码中插入调试断言，这些断言是用来检查代码的某个条件是否为真。如果条件为假，`assert`会触发一个`AssertionError`异常。
### 例如：

```python
# 如果1不等于0，程序继续执行
assert 1 != 0

# 如果1等于0，程序抛出AssertionError
assert 1 == 0

# 如果1等于0，程序抛出AssertionError并附带解释信息
assert 1 == 0, "1不等于0"

```

## 2. `join()`: join函数用于将序列中的元素以指定的字符连接生成一个新的字符串。
### 例如：

```python
words = ['Python', 'is', 'awesome']
sentence = ' '.join(words)
print(sentence) # 输出: Python is awesome
```

## 3. `with open() as f`: 读写文件是最常见的IO操作
### 例如：

```python
with open('filename.txt', 'r') as f:
   content = f.read(f)  #文件的读操作

with open('data.txt', 'w') as f:
   f.write('hello world')  #文件的写操作

r:	 # 以只读方式打开文件。文件的指针将会放在文件的开头。这是**默认模式**。
rb:  # 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
r+:  # 打开一个文件用于读写。文件指针将会放在文件的开头。
rb+: # 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。
w:	 # 打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
wb:	 # 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
w+:	 # 打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
wb+: # 以二进制格式打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
a:	 # 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab:	 # 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+:	 # 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+: # 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。

#要读取非UTF-8编码的文本文件，需要给open()函数传入encoding参数，例如，读取GBK编码的文件：
# open()函数还接收一个errors参数，表示如果遇到编码错误后如何处理。最简单的方式是直接忽略
f = open('/Users/michael/gbk.txt', 'r', encoding='gbk', errors='ignore')
```

## 4. `pickle`包的用法

### `dump()`函数是pickle模块中用于对象序列化的函数。它的基本语法如下：`pickle.dump(obj, file, protocol=None)`,其中，obj是要被序列化的对象，file是一个类似文件的对象，用于写入序列化后的数据。protocol参数指定序列化使用的协议版本，默认为最高可用的协议。

```python

import pickle

# 创建一个列表对象
my_list = [1, 2, 3, 4, 5]

# 打开一个文件用于写入
with open('my_list.pkl', 'wb') as file:
    # 使用dump()函数将列表对象序列化并写入文件
    pickle.dump(my_list, file)
```

### `load()`函数是pickle模块中用于对象反序列化的函数。它的基本语法如下：`pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict")`,其中，file是一个类似文件的对象，用于读取序列化后的数据。fix_imports、encoding和errors参数用于控制反序列化的行为。

```python

import pickle

# 打开一个文件用于读取
with open('my_list.pkl', 'rb') as file:
    # 使用load()函数从文件中读取序列化后的数据，并将其还原为原始的Python对象
    my_list = pickle.load(file)

# 打印还原后的对象
print(my_list)
```

## 5. `pandas`包用法

### （1）创建数据框

#### 使用`read_csv()`或`read_excel()`方法读取数据文件，也可以使用`DataFrame()`方法从列表或字典创建数据帧。例如，通过以下方式创建数据框：

```python
import pandas as pd

df = pd.read_csv('example.csv')
# or
df = pd.read_excel('example.xlsx')
# or
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 32, 18]})
```

### (2) 查看数据

#### 使用`head()`和`tail()`方法查看前几行或后几行数据。可以使用`describe()`方法获取数据的描述性统计信息，例如最大值、最小值、平均值和标准差等。

```python
# 查看前5行数据
print(df.head(5))

# 查看后5行数据
print(df.tail(5))

# 查看数据的描述性统计信息
print(df.describe())
```

### (3) 索引和选择数据

#### 可以使用`loc[]`和`iloc[]`方法对数据进行索引和选择。`loc[]`方法基于标签选择数据，而`iloc[]`方法基于行和列的位置选择数据，例如：

```python
# 选择行和列：
df.loc[0, 'name']
df.iloc[0, 1]

# 选择行：
df.loc[0]
df.iloc[0]

# 选择列：
df['name']
```

### (4) 操作数据

#### Pandas提供了很多数据操作方法，例如，可以使用`mean()`方法计算列的平均值，使用`corr()`方法计算列之间相关性并使用`drop()`方法删除某些列或行。

```python
# 计算列的平均值
df['age'].mean()
# 计算列之间的相关性
df.corr()
# 删除某些列或行
df.drop('age', axis=1)
df.drop(0)
```

### (5) 处理缺失值

#### Pandas提供了方法来处理缺失值，例如可以使用`isnull()`检查失值并使用`fillna()`方法填充缺失值。

```python
# 检查缺失值
df.isnull()

# 填充缺失值
df.fillna(0)
```

### (6) 分组和聚合

#### 可以使用`groupby()`方法将数据按照某些列进行分组，然后使用聚合函数计算列的值。

```python
# 分组和聚合
df.groupby('name').mean()
```

### (7) 绘制图表

#### Pandas提供了很多绘制图表的函数，例如`plot()`方法可以绘制线图、散点图和条形图等。

```python
# 绘制线图
df.plot(x='name', y='age')
# 绘制散点图
df.plot.scatter(x='name', y='age')
# 绘制条形图
df.plot.bar(x='name', y='age')
```

### (8) 排序和排名

#### 使用`sort_values()`方法对数据进行排序，可以按照某一列的值进行升序或降序排列。使用`rank()`方法进行排名，将所有的数据按照某一列的值进行排名，例如：

```python
# 按age列进行升序排列
df.sort_values('age', ascending=True)
# 按age列进行降序排列
df.sort_values('age', ascending=False)
# 对age进行排名
df['rank'] = df['age'].rank(method='dense')
```

### (9) 数据读写

#### 可以使用`to_csv()`方法数据框写入CSV文件，使用`to_excel()`方法将数据框写入Excel文件，使用`read_sql()`方法从数据库中读取数据，例如：

```python
# 将数据框写入CSV文件
df.to_csv('example.csv', index=False)
# 将数据框写入Excel文件
df.to_excel('example.xlsx', index=False)
# 从数据库中读取数据
import sqlite3
conn = sqlite3.connect('example.db')
df = pd.read_sql('select * from table1', conn)
```

## 6. `extend()` 和 `append()`和 `expand()`

### 在Python中，`append()`和`extend()`都是用于向列表中添加元素的方法，但它们的行为有所不同。

```python
# 使用append()方法
list1 = [1, 2, 3]
list1.append(4)
print(list1) # 输出: [1, 2, 3, 4]

# 使用append()方法添加一个列表
list1.append([5, 6])
print(list1) # 输出: [1, 2, 3, 4, [5, 6]]

# 使用extend()方法
list2 = [1, 2, 3]
list2.extend([4, 5, 6])
print(list2) # 输出: [1, 2, 3, 4, 5, 6]
```

### `expand()`函数是PyTorch中的一个方法，用于将张量的单个维度扩展为更大的尺寸。它返回一个新的张量视图，而不分配新内存。

```python
import torch
a = torch.Tensor([[1], [2], [3], [4]])
print('a.size: ', a.size())
print('a: ', a)
b = a.expand(4, 2)
print('b.size: ', b.size())
print('b: ', b)
```

```outputs
a.size: torch.Size([4, 1])
a:
tensor([[1.],
       [2.],
       [3.],
       [4.]])
b.size: torch.Size([4, 2])
b:
tensor([[1., 1.],
       [2., 2.],
       [3., 3.],
       [4., 4.]])
```

## 7. `transpose()`转置

### `torch.transpose` 是 PyTorch 中用于交换张量任意两个维度的函数，常用于数据格式转换或深度学习模型输入调整。

```python
import torch
# 创建一个二维张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("原始张量:")
print(x)
# 交换维度 0 和 1
y = torch.transpose(x, 0, 1)
print("转置后的张量:")
print(y)
```

```outputs
原始张量:
tensor([[1, 2, 3],
       [4, 5, 6]])
转置后的张量:
tensor([[1, 4],
       [2, 5],
       [3, 6]])
```

```python
"""
transpose(1, 2) 在 PyTorch 里表示：交换第 1 维和第 2 维（0 开始计数）。
维度索引是从 0 开始：第 0 维、第 1 维、第 2 维、第 3 维……
"""
# 3D：
x.shape = [B, M, N]
x.transpose(1, 2).shape = [B, N, M]

# 4D：
x.shape = [B, C, H, W]
x.transpose(1, 2).shape = [B, H, C, W]

"""
transpose(-2, -1) 在 PyTorch 里表示：交换张量的倒数第二个维度和最后一个维度。负索引从末尾数起：-1 是最后一维，-2 是倒数第二维。
快速心法：不管你有几维，只要写 transpose(-2,-1)，就是把“最后两轴”对调。
"""
# 2D（矩阵）：
A.shape = [M, N]
A.transpose(-2, -1).shape = [N, M]（等同于 A.t() 或 A.T）

# 3D（批量矩阵）：
X.shape = [B, M, N]
X.transpose(-2, -1).shape = [B, N, M]（等同于 X.mT）

#4D：
Y.shape = [B, C, H, W]
Y.transpose(-2, -1).shape = [B, C, W, H]（只交换最后两维）

```

## 8. `torch.matmul()`用于矩阵乘法，它会自动广播，并且会把最后两个维度当作矩阵相乘

```python
A.shape = [B, M, K]
B.shape = [B, K, N]

C = torch.matmul(A, B)
# C.shape = [B, M, N]
```

## 9. `contiguous()`

### 如果想要断开这两个变量之间的依赖（x本身是contiguous的），就要使用`contiguous()`针对x进行变化，感觉上就是我们认为的深拷贝。当调用`contiguous()`时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。

```python
x = torch.randn(3, 2)
y = torch.transpose(x, 0, 1)
print("修改前：")
print("x-", x)
print("y-", y)
 
print("\n修改后：")
y[0, 0] = 11
print("x-", x)
```

```output
修改前：
x- tensor([[-0.5670, -1.0277],
           [ 0.1981, -1.2250],
           [ 0.8494, -1.4234]])
y- tensor([[-0.5670,  0.1981,  0.8494],
           [-1.0277, -1.2250, -1.4234]])
 
修改后：
x- tensor([[11.0000, -1.0277],
           [ 0.1981, -1.2250],
           [ 0.8494, -1.4234]])
y- tensor([[11.0000,  0.1981,  0.8494],
           [-1.0277, -1.2250, -1.4234]])
```

```python
x = torch.randn(3, 2)
y = torch.transpose(x, 0, 1).contiguous()
print("修改前：")
print("x-", x)
print("y-", y)
 
print("\n修改后：")
y[0, 0] = 11
print("x-", x)
print("y-", y)
```

```output
修改前：
x- tensor([[ 0.9730,  0.8559],
           [ 1.6064,  1.4375],
           [-1.0905,  1.0690]])
y- tensor([[ 0.9730,  1.6064, -1.0905],
           [ 0.8559,  1.4375,  1.0690]])
 
修改后：
x- tensor([[ 0.9730,  0.8559],
           [ 1.6064,  1.4375],
           [-1.0905,  1.0690]])
y- tensor([[11.0000,  1.6064, -1.0905],
           [ 0.8559,  1.4375,  1.0690]])
```

## 10. `mask_fill()` 是 PyTorch 中的一个张量操作函数，用于根据布尔掩码`(mask)`将张量中指定位置的元素替换为特定值。掩码中的 `True` 值对应的张量位置会被填充为指定的值。

```python
import torch
# 创建一个 4x4 的张量
tensor = torch.arange(0, 16).view(4, 4)
print("原始张量:\n", tensor)
# 创建一个对角线为 True 的掩码
mask = torch.eye(4, dtype=torch.bool)
print("掩码:\n", mask)
# 使用 mask_fill 将对角线元素替换为 100
filled_tensor = tensor.masked_fill(mask, 100)
print("填充后的张量:\n", filled_tensor)
```

```output
原始张量:
tensor([[ 0, 1, 2, 3],
        [ 4, 5, 6, 7],
        [ 8, 9, 10, 11],
        [12, 13, 14, 15]])
掩码:
tensor([[ True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True]])
填充后的张量:
tensor([[100, 1, 2, 3],
        [ 4, 100, 6, 7],
        [ 8, 9, 100, 11],
        [ 12, 13, 14, 100]])
```

# 附录

## 1. 广播

### 广播(Broadcasting)是`NumPy`/`PyTorch`/`TensorFlow`等科学计算库中 自动对齐维度、扩展张量形状 的机制，无需手动复制数据，就能让不同形状的张量参与运算。

### 简单比喻：广播就像“自动补齐”

想象你在给一群人发糖：

|人    |想要的糖
|------|------
|A     |1 颗
|B     |1 颗
|C     |1 颗

你只有 一盒 3 颗糖，你说：“每人发 1 颗” → 系统自动把 “1 颗” 广播给每个人。在张量中，广播就是把 小的张量“复制”扩展，匹配大的张量形状。

### PyTorch 广播规则（从右到左对齐）

两个张量能广播，当且仅当：

1. 维度数相同（不够的在左边补 1）
2. 从右往左，每个维度：

       * 相等，或

       * 其中一个是 1（会被扩展），或

       * 其中一个维度不存在（视为 1）

### 举例说明

```python
attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)
# pad_mask: [B, T] → unsqueeze → [B, 1, T] → [B, 1, 1, T]
```

现在`attn_mask`是`[B, 1, 1, T]`

在`MultiHeadAttention`中：

```python
score = torch.matmul(Q, K.transpose(-2,-1))  # [B, H, T, T]
score = score.masked_fill(attn_mask == 0, float('-inf'))
```

#### 广播过程：

|张量       |形状
|----------|------
|score     |[B, H, T, T]
|attn_mask |[B, 1, 1, T]

从右往左对齐：

```txt
score:      [B, H,  T,  T]
attn_mask:  [B, 1,  1,  T]
           ↑  ↑   ↑   ↑
           =  =   1   =   → 第2维是1，自动扩展为 H
```

**PyTorch 自动把 attn_mask 扩展为 [B, H, T, T]，但不复制内存**