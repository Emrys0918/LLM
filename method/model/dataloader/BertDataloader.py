import torch
import pickle as pkl
import random
import copy
from method.utils.utils import convert_payload_to_embedding

class BertDataloader(object):
    def __init__(self, args, shuffle):
        self.data = pkl.load(open(args.dataset_path, 'rb'))
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.data)
        self.device = args.device

    def __iter__(self):
        start = 0
        while True:
            # TODO 1: 补全每次call dataloader时输出一个batch的逻辑
            # 假如start超过了data的长度，说明data已经被遍历完了，这时候如果现在是训练(shuffle is True)，就再shuffle一遍data然后让start从0开始
            # 否则，取出self.data[start: min(start + self.batch_size, len(self.data))]为一个batch，然后对其进行处理：
            # 1. 构造src，形状为[B, max_len]的LongTensor，其中B为这个batch的sample数量，max_len为这个batch的sequence的最大长度
            # 2. 构造tgt_pos，形状为[total_tgt_pos]的LongTensor，其中total_tgt_pos是这个batch的tgt_pos总数。
            # tgt_pos中的pos是把batch里的sequence拍平拼接后 转换过来的tgt_pos。
            # 例如batch里有两个sequence：[1,2,3]和[4,5,6,7]，它们各自的原始tgt_pos是[0,1]和[2]，那最终的tgt_pos就是[0,1,5]
            # 3. 构造tgt，形状为[total_tgt_pos]的LongTensor，其中total_tgt_pos是这个batch的tgt_pos总数。
            # tgt即为把batch里的tgt拍平拼接后 转换过来的tgt
        
            # 使用yield，输出一个tuple作为batch，yield的含义是，这个函数不会结束执行，只是执行到yield的时候，返回指定内容，但是其运行上下文保留
            pass