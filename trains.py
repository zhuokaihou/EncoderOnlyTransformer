# train.py - 训练主程序
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 导入自定义模块
import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import load_data, get_batch


# -------------------------- 评估函数：计算Loss和PPL ---------------------------
@torch.no_grad()
def estimate_loss_and_ppl(model):
    out = {}
    model.eval()  # 切换为评估模式
    for split in ['train', 'val']:
        losses = []
        # 多次采样取平均，结果更稳定
        for _ in range(config.eval_interval):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        ppl = np.exp(avg_loss)  # 核心公式：PPL = exp(交叉熵损失)
        out[split] = {"loss": avg_loss, "ppl": ppl}
    model.train()  # 切回训练模式
    return out

# -------------------------- 主训练流程 --------------------------
if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    train_data, val_data, encode, decode, vocab_size = load_data()
    config.vocab_size = vocab_size  # 把数据集词汇表大小同步到配置

    # 2. 初始化模型 + 优化器
    print(f"初始化模型，设备: {config.device}")
    model = EncoderOnlyTransformer(vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 3. 记录训练指标
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    iter_steps = []

    # 4. 训练循环
    print("开始训练...")
    pbar = tqdm(range(config.max_iters), desc="Training")
    for iter in pbar:
        # 定期评估
        if iter % config.eval_interval == 0:
            metrics = estimate_loss_and_ppl(model)
            # 记录数据
            train_loss = metrics["train"]["loss"]
            val_loss = metrics["val"]["loss"]
            train_ppl = metrics["train"]["ppl"]
            val_ppl = metrics["val"]["ppl"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            iter_steps.append(iter)

            # 进度条显示
            pbar.set_postfix({
                "train_loss": f"{train_loss:.3f}",
                "val_loss": f"{val_loss:.3f}",
                "train_ppl": f"{train_ppl:.2f}",
                "val_ppl": f"{val_ppl:.2f}"
            })

        # 训练单步
        x, y = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # -------------------------- 可视化：Loss & PPL 曲线 --------------------------
    print("绘制训练曲线...")
    # 创建输出文件夹
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    plt.figure(figsize=(12, 5))
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(iter_steps, train_losses, label="Train Loss", marker="o")
    plt.plot(iter_steps, val_losses, label="Val Loss", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # PPL曲线
    plt.subplot(1, 2, 2)
    plt.plot(iter_steps, train_ppls, label="Train PPL", marker="o")
    plt.plot(iter_steps, val_ppls, label="Val PPL", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity (PPL)")
    plt.title("PPL Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/loss_ppl_curve.png", dpi=300)
    plt.show()

    # -------------------------- 文本生成测试 --------------------------
    print("\n生成文本示例：")
    context = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=config.device)
    generated_idx = model.generate(context, max_new_tokens=200)
    generated_text = decode(generated_idx[0].cpu().numpy())
    print(generated_text)

    # 保存生成文本
    with open("outputs/generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    print("\n训练完成！结果已保存至 outputs 文件夹")