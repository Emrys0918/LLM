import torch
import torch.nn as nn
from method.model.transformer.basic import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        # 由大量TransformerEncoderLayer层组成，一共args.layers_num层，TransformerEncoderLayer也由args初始化
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.layers_num)
        ])
        self.hidden_size = args.hidden_size

    def forward(self, src, embedding):
        # 输入src: [batch_size, seq_length]，原始token
        # embedding: [batch_size, emb_size]， embedding层的输出 ❌ ，应该是[batch_size, seq_length, emb_size]
        # 首先，要生成一个mask，覆盖所有不是[PAD]的位置，以防模型forward的时候看到不应该看到的信息。
        # 这个mask的形状应该是[batch_size, 1, seq_length, seq_length]，作为每一层transformer的输入的一部分
        # 然后，依次经过所有的transformer层，得到输出。输出形状为[batch_size, seq_length, hidden_dim]
        # 此时，先实现TransformerEncoderLayer，在method/model/transformer/basic中

        pad_mask = (src == 0)

        att_mask = pad_mask.unsqueeze(1).unsqueeze(2) # 会广播

        hidden = embedding
        for layer in self.layers:
            hidden = layer(hidden,att_mask)

        return hidden

