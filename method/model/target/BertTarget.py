import torch
import torch.nn as nn
from method.model.auxiliary_func.act_fun import *
from method.model.auxiliary_func.layer_norm import LayerNorm
from method.model.tokenizer.tokenizer import DoubleByteTokenizer


class BertTarget(nn.Module):
    def __init__(self, args):
        super(BertTarget, self).__init__()
        # 两层输出头，接收transformer encoder输出的隐藏表征，分类为token
        # 线性emb_size->emb_size，激活函数使用GeLU，再emb_size->词表大小
        # 再定义损失函数，即token分类损失
       
    def mlm(self, src, hidden, tgt_pos, tgt):
        # 定义训练任务，这里我们只取MLM即可，输入即为dataloader输出的内容再加encoder输出的内容：src/hidden/tgt_pos/tgt
        # 首先，取出hidden里所有原始token不是[PAD]的位置，这个操作还会自动将选取到的hidden拍成二维，即[-1, emb_size]
        # 然后，对这个tensor过一遍输出头，得到分类概率。
        # 接着，取出这个tensor的tgt_pos位置，即是被mask的token的分类概率(可以验证一下tgt_pos与这个tensor位置的对应关系)
        # 随后，基于取出的分类概率与tgt，计算loss
        # 计算其他统计量，包括correct_num: 答对的token数和denominator: 总的tgt token数
        # 返回loss, correct_num, denominator
        pass
    
    
    def forward(self, src, hidden, tgt_pos, tgt):
        loss_mlm, correct_mlm, denominator = self.mlm(src, hidden, tgt_pos, tgt)
        return loss_mlm, correct_mlm, denominator