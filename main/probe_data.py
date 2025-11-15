data_path = '../data/pretrain/pretrain.txt'

# TODO 1: 以文本格式一次性读取data_path，并输出总行数

with open(data_path,"r",encoding = "utf-8",errors = "ignore") as f:
    data = f.read()

# TODO 2: 把读进来的全部数据按照双换行(空一行)分割为Session，并输出总Session个数和第一个Session作为参考，用于后续正式处理
# 注意！数据文件结尾也有空行，需要过滤掉分割出的最后一个Session，因为它会是空的

# 下面是错误写法，该写法是一次性完整读入文件后，在利用splitlines切分行数，最后统计列表lines的大小（即行数），这会对内存造成极大的压力，以致于好几次服务器重启

# with open(data_path,"r",encoding="utf-8") as f:
#     data = f.read()
# lines = data.splitlines() # 按行切分

"""
lines = data.splitlines() 返回的是一个 Python 的 list
"""

# print(f"总行数: {len(lines)}")
# print(f"前10行: {lines[:10]}")

sessions = data.split("\n\n")

# 打印结果
print(f"Session总数为: {len(sessions)}")
print(f"第一个Session为: {sessions[0]}")
