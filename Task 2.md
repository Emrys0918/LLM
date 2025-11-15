# 任务2 & 3:BERT预训练

## 2.1 BERT训练集生成

- 这部分的目的是，使用任务1得到的网络流量语料库，生成用于BERT预训练的训练集。训练集需要匹配BERT的主要训练任务，即<u>掩码语言模型(MLM)</u>，且需要足够大。
- 打开`method/model/dataset/BertDataset.py`，逐步按照指示实现。
- 打开`main/generate_pretrain_data.py`，逐步按照指示实现。
- 运行`main/generate_pretrain_data.py`以生成数据。

## 2.2 BERT模型构建
- 标准BERT模型由1个embedding层(method/model/embedding/embedding.py)、12个transformer层构成的encoder(method/model/transformer/encoder.py)和1个输出层(`method/model/target/BertTarget.py)构成。
- 首先实现`method/model/embedding/embedding.py`
- 接下来实现`method/model/transformer/encoder.py`
- 接下来实现`method/model/target/BertTarget.py`
- 最后，在`method/model/model_combine/BertModel.py`拼成完整的BERT

## 2.3 BERT训练流程搭建
- 只差最后的拼图了，还需要实现的内容包括：
    - BertDataloader: 把数据从BertDataset中按batch读出来并组织成tensor(`method/dataloader/BertDataloader.py`)
    - BertTrainer: 用于训练BERT的类(`method/model/trainer/BertTrainer.py`)
- 随后，完成主程序`method/start_pretrain.py`与`main/pretrain.py`，并尝试运行，成功运行以后记得持续观察。

