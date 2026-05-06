# generate.py - 独立文本生成推理脚本（训练完成后运行）
import torch
import os

# 导入项目核心模块
import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import load_data

# -------------------------- 生成配置（可自行修改） --------------------------
MAX_NEW_TOKENS = 300    # 生成的文本长度
INPUT_CONTEXT = "JULIET:"  # 输入的开头文本（可自定义）
SAVE_OUTPUT_PATH = "outputs/generated_result.txt"  # 生成文本保存路径

# -------------------------- 初始化模型与数据 --------------------------
def init_model():
    # 1. 加载数据集（获取编码/解码函数、词汇表大小）
    _, _, encode, decode, vocab_size = load_data()
    config.vocab_size = vocab_size

    # 2. 初始化Transformer模型
    model = EncoderOnlyTransformer(vocab_size).to(config.device)
    model.eval()  # 切换为推理模式（关闭Dropout等训练层）
    return model, encode, decode

# -------------------------- 文本生成主函数 --------------------------
def generate_text(model, encode, decode):
    # 将输入文本编码为数字序列
    context = torch.tensor([encode(INPUT_CONTEXT)], dtype=torch.long, device=config.device)
    
    # 推理阶段禁用梯度计算，加速运行
    with torch.no_grad():
        generated_idx = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)
    
    # 解码为可读文本
    generated_text = decode(generated_idx[0].cpu().numpy())
    return generated_text

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("="*50)
    print("Encoder-only Transformer 文本生成")
    print("="*50)
    
    # 初始化
    model, encode, decode = init_model()
    
    # 生成文本
    print(f"\n输入开头：{INPUT_CONTEXT}")
    print("正在生成文本...\n")
    result = generate_text(model, encode, decode)
    
    # 打印结果
    print("生成完成：")
    print(result)
    
    # 保存结果到文件
    os.makedirs("outputs", exist_ok=True)
    with open(SAVE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"\n✅ 文本已保存至：{SAVE_OUTPUT_PATH}")