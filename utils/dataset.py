import torch
import os
import urllib.request
# 导入项目超参数
import config

# 全局变量：编码解码函数、训练/验证数据、词汇表大小
train_data = None
val_data = None
encode = None
decode = None
vocab_size = None

# -------------------------- 1. 加载并预处理数据 --------------------------
def load_data():
    global train_data, val_data, encode, decode, vocab_size
    
    # 自动下载 Tiny Shakespeare 数据集
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data_path = 'data/input.txt'
    
    # 创建 data 文件夹（如果不存在）
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 下载数据集
    if not os.path.exists(data_path):
        print("正在下载数据集...")
        urllib.request.urlretrieve(data_url, data_path)
    
    # 读取文本数据
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 构建字符级词汇表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # 字符 ↔ 索引 映射
    stoi = {ch: i for i, ch in enumerate(chars)}  # 字符→索引
    itos = {i: ch for i, ch in enumerate(chars)}  # 索引→字符
    
    # 编码/解码函数
    def encode_func(s):
        return [stoi[c] for c in s]  # 文本→数字序列
    
    def decode_func(l):
        return ''.join([itos[i] for i in l])  # 数字序列→文本
    
    # 全局赋值
    encode = encode_func
    decode = decode_func
    
    # 转换为张量，划分训练集(90%)/验证集(10%)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"数据集加载完成！词汇表大小: {vocab_size}, 总字符数: {len(text)}")
    return train_data, val_data, encode, decode, vocab_size

# -------------------------- 2. 生成训练/验证批次 --------------------------
def get_batch(split):
    """
    生成一个批次的输入x和标签y
    split: 'train' 或 'val'
    x: [batch_size, block_size] 输入序列
    y: [batch_size, block_size] 下一个token序列（预测目标）
    """
    data = train_data if split == 'train' else val_data
    
    # 随机采样批次的起始位置
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # 构建输入和标签
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    
    # 移动到指定设备（CPU/GPU）
    x, y = x.to(config.device), y.to(config.device)
    return x, y