# 网络流量大模型预训练及其微调

本项目的主要目的是基于**网络流量**数据，预训练一个大模型，随后在下游任务（网络流量应用分类）上进行微调与测试。

### 数据特点

- 网络流量数据的单位是会话。一次客户端与服务器的会话(Session)中，包含按时间顺序排列的一个或多个IP层流量包。会话一般呈现这样的形式：某一方先发送请求，另一方回复，随后循环往复，类似多轮对话，且某一方可以连续发送多个包(发送大量数据时需要分割为多个包连续发送)。
- 每个流量包有两个组成部分：包头(header)和载荷(payload)。包头包含了该流量包在IP层及更上层的协议及其字段信息，载荷则包含了数据。载荷完全可能是空，但包头一定存在。
- 包头和载荷都是二进制，以16进制字符串的形式存储在`./data/pretrain/pretrain.txt`中。包头包括哪些协议，协议有什么字段、字段的定义均未知，且载荷可能是加密也可能是不加密的，其具体内容可以认为无法逆向。因此，保留数据的**十六进制格式**并以其为基础建模，是具有合理性的。

### 模型架构

- Transformer有Encoder和Decoder两种变体。我们将分别基于Encoder构建类BERT的Encoder-Only模型和基于Decoder构建类GPT的Decoder-Only模型。

### 预训练
- 让模型吃下语料库(pretrain.txt)中的全部数据，充分理解流量的形式、学习流量呈现出的规律。预训练的目的是让模型**积累知识到它的大量参数中**，从而后续仅需少量数据微调，即可在下游任务上表现优异。

### 下游任务微调
- 我们的下游任务是流量应用分类，即根据流量的16进制字符串表示，反推该流量对应的会话**是什么应用主导的**，例如聊天、看视频、打电话、下载数据、浏览网页等。这些行为会造成不同的流量行为模式。
- 下游任务数据相比预训练语料库是非常小的，但有**明确的分类标签**，因此可以通过在下游任务数据上微调预训练出的模型用于后续的分类任务。
- 下游任务数据包括训练、验证和测试集，存储在`./data/finetune`中。

### 任务总览
- 本项目将分多个阶段走完整个流程：
    - 1. 数据处理：构建Tokenizer、将预训练语料分为session、packet、header和payload，对所有header进行Tokenize，并分片存储。
    - 2. 模型构建：手写Transformer的Embedding层、LayerNorm、Multi-head Attention(多投注意力)、Transformer Encoder并跑通
    - 3. 模型训练：构建BERT Target(预训练预测头)、BERT Model、BERT Trainer、BERT Dataset和BERT Dataloader，构建optimizer和scheduler并写预训练脚本start_pretrain.py开始预训练
    - 4. 下游任务微调：基于上述预训练完的模型，构建BERT Classifier并写微调脚本finetune.py进行下游任务微调、并测试效果
    - 5. GPT模型构建、预训练：写Transformer Decoder、GPT Target、Model、Trainer、Dataset和Dataloader，并预训练
    - 6. GPT模型微调：构建GPT Classifier并修改finetune.py进行微调
    - 后续探索方向：
        - 使用transformers库精简上述已经完成的内容
        - payload encoder与soft prompt注入
        - 扩大模型参数量、多卡分布式训练

### Tips
- 每个任务主要是以适当的代码填充指定的文件的形式展开，但要跑通并验证正确性，调试和debug也是无法避免的
- 遇到<u>有下划线的文字</u>，说明建议查询并学习的内容。