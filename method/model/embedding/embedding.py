import torch
import torch.nn as nn
from method.model.auxiliary_func.layer_norm import LayerNorm
from method.model.tokenizer.tokenizer import DoubleByteTokenizer


class WordPosEmbedding(nn.Module):
    def __init__(self, args):
        super(WordPosEmbedding, self).__init__()
        # TODO 1: WordPosEmbedding包含word_embedding与position_embedding两大组成部分以及一个layernorm层。
        # 首先，我们先完成method/model/auxiliary_func/layer_norm.py，实现LayerNorm
        # word_embedding为大小为[词表大小, args.emb_size]的nn.Embedding
        # position_embedding为大小为[args.max_seq_length, args.emb_size]的nn.Embedding
        self.tokenizer = DoubleByteTokenizer()
        self.word_embedding = nn.Embedding(len(self.tokenizer.vocab),args.emb_size)
        self.position_embedding = nn.Embedding(args.max_seq_length, args.emb_size)
        self.Layer_Norm = LayerNorm(args.emb_size)

    def forward(self, src):
        # src是二维LongTensor，形状为[batch_size, seq_length]，里面每个元素都是token，作为word_embedding的输入
        # position_embedding的输入是二维LongTensor，形状为[batch_size, seq_length]，每行是从0到seq_length - 1的position
        # 输出为word_embedding + position_embedding，然后过layernorm得到的结果
        batch_size,seq_length = src.size()
        position = torch.arange(seq_length,dtype=torch.long,device=src.device)
        position = position.unsqueeze(0).expand(batch_size,seq_length)

        word_embed = self.word_embedding(src)
        position_embed = self.position_embedding(position)
        return self.Layer_Norm(word_embed + position_embed)