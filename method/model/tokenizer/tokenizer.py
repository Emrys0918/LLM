class DoubleByteTokenizer(object):
    """
    类名: DoubleByteTokenizer
    功能: 将文本序列编码为双字节格式，同时支持将编码还原为原始文本序列。
    """
    def __init__(self):
        """
        初始化分词器，加载词汇表并构建双字节编码。
        """
        # 首先定义特殊标记，这些特殊标记后面用到的时候再解释，现在先写死
        self.vocab = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[ENC]': 5,
            '[GEN]': 6,
        }
        self.num_special_tokens = len(self.vocab)

        # TODO 1: 构建双字节词汇表，从 0x0000 到 0xFFFF 共 65536 个 token
        # 依次以16进制字符串(例如FFFF)到Token ID(FFFF是65535+self.num_special_tokens)的映射加入self.vocab中
        # 提示: 建议归一化到四位，例如0是0000
        for num in range(0x10000):
            self.vocab[f"{num:04x}"] = self.num_special_tokens + num

        # 反向词汇表，用于解码操作
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, sequence):
        """
        将输入文本序列转换为双字节编码的索引序列。

        Args:
            str: 输入的文本序列。

        Returns:
            List: 编码后的索引序列。
        """
        res = [] # 存储编码结果的列表
        # TODO 2: 把sequence每四个字符Tokenize为其对应的Token ID
        # 提示: 如果剩余的字符数不足四个，需要在剩余字符前面补0
        current_str = ""
        temp_count = 0
        for elem in sequence:
            if temp_count == 4:
                res.append(self.vocab[current_str])
                temp_count = 1
                current_str = elem
            else:
                temp_count += 1
                current_str += elem

        # 处理剩下的不足四位的字符
        for i in range(4-len(current_str)):
            current_str = f"0{current_str}"
        res.append(self.vocab[current_str])

        return res

    def de_tokenize(self, ids):
        """
        将双字节编码的索引序列还原为文本序列。

        Args:
            List: 输入的索引序列。

        Returns:
            str: 解码后的文本序列。
        """
        res = ''

        for id in ids:
            res += self.inverse_vocab[id]

        return res