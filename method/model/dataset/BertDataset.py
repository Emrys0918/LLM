import random
import os
import pickle as pkl
import multiprocessing as mp
from method.model.tokenizer.tokenizer import DoubleByteTokenizer



class BertDataset(object):
    def __init__(self, args):
        self.corpus_path = args.corpus_path # 输入的预训练语料路径
        self.dataset_path = args.dataset_path # 输出的预训练数据集路径
        self.seq_length = args.seq_length # 模型能接受的最大序列长度
        self.volume = args.volume # 期望制造的训练样本数量
        self.num_workers = args.num_workers
        self.tokenizer = DoubleByteTokenizer()

    def build_and_save(self):
        assert(self.num_workers >= 1)
        print(f"Starting {self.num_workers} processes to generate dataset...")
        # TODO1:开num_workers个多进程处理所有的分片数据，对于每片，调用self.worker(args)来生成数据，总共生成的数据量要约等于self.volume。args里放的内容参考

        chunk_files = sorted([f for f in os.listdir(self.corpus_path) if f.endswith('.pkl')]) #找到所有的chunk文件
        num_chunks = len(chunk_files)
        print(f"找到 {num_chunks} 个chunk files")

        samples_per_chunk = self.volume // num_chunks # 计算每个chunk需要生成的样本数

        # worker参数列表
        worker_args = []
        for i, chunk_file in enumerate(chunk_files):
            read_path = os.path.join(self.corpus_path, chunk_file)
            save_path = os.path.join(self.dataset_path, f'samples_{i}.pkl')

            # 最后一个chunk处理剩余的样本
            if i == num_chunks - 1:
                target = self.volume - (samples_per_chunk * (num_chunks - 1))
            else:
                target = samples_per_chunk

            worker_args.append((target, read_path, save_path))
        
        with mp.Pool(processes=self.num_workers) as pool:
          pool.map(self.worker, worker_args)

        print("准备工作已完成，开始处理数据...")

        all_data = []
        # TODO2:接下来收集所有单个分片数据处理出的分片训练集，把它们集合到一个大文件中，并打乱顺序

        # 读取所有worker生成的文件
        for i in range(num_chunks):
            sample_file = os.path.join(self.dataset_path, f'samples_{i}.pkl')
            chunk_data = pkl.load(open(sample_file, 'rb'))
            all_data.extend(chunk_data)
            print(f"加载 samples_{i}.pkl: {len(chunk_data)} samples")

        # 打乱顺序
        random.shuffle(all_data)
        print(f"样本总数: {len(all_data)}")

        # 保存最终数据集
        pkl.dump(all_data, open(os.path.join(self.dataset_path, 'bert.pt'), 'wb'))
        print(f"数据被保存至 {os.path.join(self.dataset_path, 'bert.pt')}")

    def worker(self, args):
        target, read_path, save_path = args
        # args中应该要包含：该数据片要生成的训练样本数量、数据片读取路径、训练样本输出路径
        # TODO 1: 读取分片数据，然后生成训练样本。
        # 每个训练样本的生成逻辑是：首先随机选择一个session，从session里随机选择一个packet作为起始。
        # 从这个packet开始，逐步构造Token序列：
        # 第一个packet变成[CLS]+packet的token序列+[SEP]，其余packet变为packet的token序列+[SEP]。
        # 例如，假如packet的token序列是[8,9,10,11,12]，那作为第一个packet就会变成[1,8,9,10,11,12,2]，作为其他packet就会变成[8,9,10,11,12,2]
        # 从起始packet开始，逐步增加后续packet直到达到最后一个packet或者达到self.seq_length为止，得到一个训练样本的原始token序列。
        # 然后，对该训练样本调用self.mask_seq进行mask，得到最终的训练样本。
        # TODO 2: 实现self.mask_seq。它的输入即为list形式的token序列，输出是掩码后的token序列、掩码位置、掩码位置处的原始token，都是list。
        # self.mask_seq的逻辑入下：
        # 对于输入序列中每个不是special token的token，有30%的概率触发掩码逻辑。
        # 掩码逻辑是，有80%替换为[MASK]，10%替换为非special token的随机token，10%不替换。

        corpus = pkl.load(open(read_path, 'rb'))
        result = []
        """
        难点：每个样本长短不一，这在一开始入手的时候容易产生误导
        """
        for _ in range(target):
            session = random.choice(corpus) # 随机选取session
            start_id = random.randint(0,len(session)-1) # 随机选取packet
            tokens = [] #创建临时存储

            # 处理第一个包
            first_packet = session[start_id] 
            tokens.append(1)
            tokens.extend(first_packet[0])
            tokens.append(2)

            # 处理后续的包
            current_id = start_id + 1
            while current_id <len(session) and len(tokens) < self.seq_length:
                packet = session[current_id] 

                # 检查是否超出长度,要加上‘2’，即特殊字符[SEP]
                if len(tokens) + len(packet[0]) +1 > self.seq_length:
                    break

                tokens.extend(packet[0])
                tokens.append(2)
                current_id += 1
            
            # mask
            masked_tokens, positions, targets = self.mask_seq(tokens)

            # 保存
            result.append({
            'input': masked_tokens,
            'positions': positions,
            'targets': targets
            })

        pkl.dump(result, open(save_path, 'wb'))


    def mask_seq(self, sequence):
        targets = []
        positions = []
        sequence = sequence.copy()
        special_id = {0,1,2,3,4,5,6}
        for i in range(len(sequence)):
            if sequence[i] in special_id:
                continue
            if random.random() < 0.3:
                targets.append(sequence[i])
                positions.append(i)
                prob = random.random()
                if prob < 0.8:
                    sequence[i]=4
                elif prob < 0.9:
                    sequence[i]=random.randint(7,65543)
        return sequence, positions, targets
        

