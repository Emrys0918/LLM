import argparse
import sys
import torch
sys.path.append('..')
from method.start_pretrain import train_and_validate
from method.utils.utils import setuplogging


def main():
    # 使用argparse.ArgumentParser
    # 输入dataset_path、output_model_path、total_steps、save_checkpoint_steps
    # report_steps、batch_size、seed、max_seq_length、learning_rate、warmup
    # emb_size、layers_num、hidden_size、feedforward_size、device
    args = None
    setuplogging('/data/jiazian/net/log', 'bert')
    train_and_validate(args)


if __name__ == "__main__":
    main()