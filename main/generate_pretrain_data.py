import argparse
import sys
import os
sys.path.append("..")
from method.model.dataset.BertDataset import BertDataset

 
def main():
    # TODO 1: 使用argparse.ArgumentParser，从而在命令行指定输入文件夹、输出文件夹、num_workers、volume和seq_length
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--corpus_path', type=str, default = '/data/jiazian/xjq/llm_practice/data/pretrain_processed',
                        help='输入的预训练语料路径(包含chunk_*.pkl文件的文件夹)')
    parser.add_argument('--dataset_path', type=str, default = '/data/jiazian/xjq/llm_practice/data/pretrain_data', 
                        help='输出的预训练数据集路径')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='模型能接受的最大序列长度')
    parser.add_argument('--volume', type=int, required=True,
                        help='期望制造的训练样本数量')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='多进程数量')
    
    args = parser.parse_args()
    os.makedirs(args.dataset_path, exist_ok=True)
    dataset = BertDataset(args)
    dataset.build_and_save()


if __name__ == '__main__':
    main()
    