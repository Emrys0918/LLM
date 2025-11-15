import sys
import os
import random
import pickle
from tqdm import tqdm
sys.path.append('..')
from method.model.tokenizer.tokenizer import DoubleByteTokenizer

data_path = '../data/pretrain/pretrain.txt'
output_path = '../data/pretrain_processed'

# TODO 1: 创建output_path这个文件夹
# 提示：使用os.makedirs(..., exist_ok=True)
os.makedirs(output_path,exist_ok = True)

print('Reading data...')

# TODO 2: 以文本格式一次性读取data_path

with open(data_path, 'r', encoding='utf-8') as f:
    content = f.read() #读取整个txt为一个字符串

print('Splitting data...')

# TODO 3: 把读进来的全部数据按照双换行分割为Session
# 提示：记得过滤掉最后一个空Session

sessions = content.split("\n\n")
# res_sessions = [session for session in sessions if session.strip()]

# TODO 4: 把每个Session按照换行分割为Packet

packets = [[packet for packet in session.split("\n") if packet.strip()]
             for session in sessions if session.strip()]
# res_packets = [packet for packet in packets]

print('Splitting header and payload...') 

# TODO 5: 把每个Packet按照\t分割为Header和Payload

# res = []
# for session in packets:
#     HPs = [packet.split("\t") for packet in session]
#     res.append(HPs)

res = [[packet.split("\t") for packet in session] for session in packets]

# 如果内存不够，试试：

# res = [[packet.split("\t") for packet in session] for session in [session.split("\n") for session in content.split("\n\n") if session.strip()]]

print('Tokenizing content...')

# JUMP: 找到../method/model/tokenizer/tokenizer.py，先按照其中指示实现DoubleByteTokenizer
# TODO 6: 开一个DoubleByteTokenizer，对每个Packet的Header进行Tokenize
# 提示: 该步骤较慢(20分钟)，建议使用tqdm包开一个进度条

Tokenizer = DoubleByteTokenizer()
for session in tqdm(res):
    for packet in session:
        packet[0] = Tokenizer.tokenize(packet[0])
print('Tokenization complete.')

# TODO 7: 分片保存到output_path，分片的原因是，假如整体一次性保存，在后续一次性读取时内存会爆炸。
# 提示: 保存前需要随机打乱所有Session的顺序，避免后续按片处理时产生bias并保证负载均衡
# 提示: 使用pickle包存储List的切片

print('Shuffling and saving data...')

random.shuffle(res)

# 总数1765206
chunk_size = 200000
total_sessions = len(res)

for i in range(0, total_sessions, chunk_size):
    chunk = res[i:i+chunk_size]
    chunk_filename = os.path.join(output_path, f'chunk_{i//chunk_size}.pkl')
    with open(chunk_filename, 'wb') as f:
        pickle.dump(chunk, f)
    print(f'Saved chunk {i//chunk_size}: {len(chunk)} sessions')

print(f'All data saved to {output_path}. Total chunks: {(total_sessions + chunk_size - 1) // chunk_size}')
