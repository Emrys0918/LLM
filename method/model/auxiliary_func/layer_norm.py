import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNorm, self).__init__()
        # TODO 1: LayerNorm由epsilon,gamma,beta组成，gamma,beta都是参数，且gamma初始化为1，beta初始化为0,要保证可训练
        
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        # epsilon是很小的常数，用于数值稳定性
        self.epsilon = 1e-12

    # def forward(self, input):
    #     # 对最后一个维度计算均值和方差,特征维度
    #     # LayerNorm公式: (x - mean) / sqrt(var + epsilon) * gamma + beta
    #     return self.gamma * (input - torch.mean(input,dim=-1,keepdim=True)) / torch.sqrt(torch.var(input,dim=-1,keepdim=True,unbiased=False) + self.epsilon) + self.beta

    def forward(self,input):
        # 对最后一个维度计算均值和方差,特征维度
        mean = torch.mean(input,dim=-1,keepdim=True)
        var = torch.var(input,dim=-1,keepdim=True,unbiased=False)

        output = self.gamma * (input - mean) / torch.sqrt(var + self.epsilon)
        output += self.beta

        return output