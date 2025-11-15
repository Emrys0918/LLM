import sys
import os
sys.path.append('..')
import torch
import pickle
import pandas as pd
from method.model.tokenizer.tokenizer import DoubleByteTokenizer

print(torch.__version__)
# str =   "0000000100020003000400050006\t000700080009000a000b000cd\n0000000100020003000400050006\t000700080009000a000b000cd\n0000000100020003000400050006\t000700080009000a000b000cd\n\n0000000100020003000400050006\t000700080009000a000b000cd\n0000000100020003000400050006\t000700080009000a000b000cd\n0000000100020003000400050006\t000700080009000a000b000cd"

# sessions = str.split("\n\n")
# res = [session for session in sessions if session.strip()]
# # print(sessions)

# packets = [session.split("\n") for session in sessions if session.strip()]
# res_packets = [packet for packet in packets]
# # print(packets)

# res = [[packet.split("\t") for packet in session] for session in packets]
# # print(res)

# Tokenizer = DoubleByteTokenizer()
# print(Tokenizer.tokenize("ffff"))
# for session in res:
#     for packet in session:
#         packet[0] = Tokenizer.tokenize(packet[0])

# print(res)
# with open('/data/jiazian/xjq/llm_practice/data/pretrain_processed/chunk_0.pkl', 'rb') as f:
#     my_list = pickle.load(f)
# print(type(my_list))

# my_list = [[["A","B","C","D"],["A","B","C","D"]],[["A","B","C","D"],["A","B","C","D"]]]
# df = pd.DataFrame(res)
# df.to_csv('example.txt',index = False,header = False)
# print(df)

# list = []
# l1 = [1,2]
# # list.append(l1)
# list.append(1)
# list.extend(l1)
# print(list)
torch.manual_seed(42)
A = torch.randn(8,4)
C = torch.randn(8,2,4,4)
print(f"{A.shape},\n{A}")
B = (A > 0)
if B is not None:
    C = C.masked_fill(B == False, float('-inf'))
print(f"{B.shape},\n{B}")
print(f"{C.shape},\n{torch.softmax(C,dim=-1)}")
# B = torch.tensor([[1, -1], [1, -1]],dtype=torch.float32)
# for i in range(len(A)):
#     for j in range(len(A[0])):
#         if A[i,j] == -1:
#             B[i,j]= -1e9
#         else:
#             B[i,j]= 0
# B = (A == -1)
# print(torch.softmax((A+B),dim=-1),A+B)
# position = torch.arange(5, dtype=torch.long)
# position1 = position.unsqueeze(0)
# print(position,position1)
# print(torch.mean(A,dim=-1))