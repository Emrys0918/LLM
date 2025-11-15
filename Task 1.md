# 任务1:环境安装与数据预处理

## 1.1 环境安装

- 这部分比较简单，其存在的必要是统一一下torch版本。建议使用python3.10+torch2.6-cuda124版本：
```
conda create -n net python==3.10 -y # 创建一个名称为net的conda环境，并固定python版本为3.10
conda activate net
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 # 安装预编译了适配cuda12.4的torch2.6版本
```

## 1.2 认识数据格式

- 在简介中，我们初步认识了网络流量数据的形式：数据由大量Session构成，每个Session有大量packet，每个packet由两部分构成：header和payload。
- 在`./data/pretrain/pretrain.txt`中，数据也是这样组织的，但是是纯文本形式。其中，Session之间空一行，Session中每一行都是一个packet，每个packet由`\t`分割header和payload，例如：
```
Session 1 Packet 1
Session 1 Packet 2
...
Session 1 Packet N

Session 2 Packet 1
Session 2 Packet 2
...
Session 2 Packet M
```
其中，每个packet形式如下：
```
header  payload
```
### 查看真实数据
- 首先进入main文件夹：
```
cd main
```
- 查看`pretrain.txt`大小：
```
du -sh ../data/pretrain/pretrain.txt
```

- 由于`pretrain.txt`非常大，因此我们查看其前25行
```
head -n 25 ../data/pretrain/pretrain.txt
```
验证一下之前介绍的数据格式。

### 统计数据数量
- 打开`probe_data.py`，按照其中的指示行动。

## 1.3 数据处理

- 数据处理的目的是，按照1.2中阐述的数据格式，组织全部数据，即：
    - 整个数据是一个巨大的List，List中的每个元素是一个Session。
    - 每个Session也是一个List，其中的每个元素是一个Packet。
    - 每个Packet是一个Tuple，第一个元素是Header，第二个元素是Payload。
    - Header需要通过<u>Tokenizer</u>转换为<u>Token</u>。
    - Payload保持原始文本格式，目前用不到。

- 打开`preprocess_data.py`，逐步按照指示实现上述逻辑。
- 调试并运行`preprocess_data.py`！