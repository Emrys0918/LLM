import random
import os
import torch
import numpy as np
import logging
import time


def set_seed(seed=7):
    random.seed(seed)  # 设置 Python 内置的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 随机种子（GPU）
    torch.backends.cudnn.deterministic = True  # 保证 CuDNN 的可复现性

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path), strict=False)

def setuplogging(log_path, target):
	root = logging.getLogger()
	root.setLevel(logging.INFO)
	formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	handler = logging.FileHandler(log_path + f'/{target}-{formatted_time}-log.txt')
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
	handler.setFormatter(formatter)
	root.addHandler(handler)

def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
