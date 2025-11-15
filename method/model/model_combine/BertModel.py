import torch
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self, embedding, encoder, target):
        super(BertModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src, tgt_pos, tgt):
        # embedding -> encoder -> target
        pass