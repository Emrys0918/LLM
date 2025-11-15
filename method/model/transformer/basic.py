import torch.nn as nn
import torch
import math
from method.model.auxiliary_func.layer_norm import LayerNorm
from method.model.auxiliary_func.act_fun import gelu

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num):
        super(MultiHeadAttention, self).__init__()
        # 三个线性层用于分别计算 query、key 和 value 的变换
        # 将最终结果由最后一个线性层输出
        assert hidden_size % head_num == 0   #hidden_size必须能被head_num整除

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = self.hidden_size // self.head_num

        self.linear_Q = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_K = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_V = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_output = nn.Linear(self.hidden_size,self.hidden_size)

    def forward(self, hidden, mask): 
        # 自注意力，QKV均为hidden，mask即为mask
        # 全部手搓
        batch_size,seq_length,_ = hidden.size()

        # 线性变换
        Q = self.linear_Q(hidden)
        K = self.linear_K(hidden)
        V = self.linear_V(hidden)

        # 映射，[batch_size,seq_length,head_num,head_dim]->[batch_size,head_num,seq_length,head_dim]
        Q = Q.view(batch_size,seq_length,self.head_num,self.head_dim).transpose(1,2)
        K = K.view(batch_size,seq_length,self.head_num,self.head_dim).transpose(1,2)
        V = V.view(batch_size,seq_length,self.head_num,self.head_dim).transpose(1,2)

        # 计算注意力分数
        score = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # mask
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        
        # softmax
        attention = torch.softmax(score,dim=-1)

        Value = torch.matmul(attention,V)

        # 拼接
        output = Value.transpose(1,2).reshape(batch_size,seq_length,self.hidden_size)

        output = self.linear_output(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, feedforward_size):
        super(FeedForward, self).__init__()
        # 定义前馈网络的两个线性层
        # 第一个hidden_size -> feedforward_size
        # 第二个feedforward_size -> hidden_size
        # 激活函数使用GeLU: method/model/auxiliayr_func/act_func.py中已经实现
        self.linear_Feed_1 = nn.Linear(in_features = hidden_size,out_features = feedforward_size)
        self.linear_Feed_2 = nn.Linear(in_features = feedforward_size,out_features = hidden_size)
        self.Gelu = gelu # 函数不需要被实例化
    
    def forward(self, x):
        # 前馈网络的计算: 线性变换 -> 激活 -> 线性变换
        return self.linear_Feed_2(self.Gelu(self.linear_Feed_1(x)))
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        # TransformerEncoderLayer由一个attention和一个ffn组成，在attention和ffn后各有一个layernorm
        # attention是多头注意力，且头的数量是args.heads_num
        # 首先实现FeedForward和MultiHeadAttention，都在本文件中
        self.attention = MultiHeadAttention(args.hidden_size,args.heads_num)
        self.layernorm1 = LayerNorm(args.hidden_size)
        self.feedforward = FeedForward(args.hidden_size,args.feedward_szie)
        self.layernorm2 = LayerNorm(args.hidden_size)
        

    def forward(self, hidden, mask):
        # attention -> layernorm -> ffn -> layernorm
        att_out = self.attention(hidden,mask)
        output = self.layernorm1(att_out + hidden)

        fee_out = self.feedforward(output)
        output = self.layernorm2(output + fee_out)

        return output
    