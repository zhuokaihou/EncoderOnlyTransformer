import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import get_batch, load_data


@torch.no_grad()
def estimate_loss_and_ppl(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        avg_loss = float(np.mean(losses))
        out[split] = {"loss": avg_loss, "ppl": float(np.exp(avg_loss))}
    model.train()
    return out


def plot_metrics(iter_steps, train_losses, val_losses, train_ppls, val_ppls):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iter_steps, train_losses, label="Train Loss", marker="o")
    plt.plot(iter_steps, val_losses, label="Val Loss", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iter_steps, train_ppls, label="Train PPL", marker="o")
    plt.plot(iter_steps, val_ppls, label="Val PPL", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.title("PPL Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/loss_ppl_curve.png", dpi=300)
    plt.close()


def main():
    torch.manual_seed(config.seed)

    print("Loading data...")
    _, _, encode, decode, vocab_size = load_data()
    config.vocab_size = vocab_size

    print(f"Initializing model on {config.device}...")
    model = EncoderOnlyTransformer(vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    iter_steps = []

    print("Training...")
    pbar = tqdm(range(config.max_iters), desc="Training")
    for step in pbar:
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            metrics = estimate_loss_and_ppl(model)
            train_loss = metrics["train"]["loss"]
            val_loss = metrics["val"]["loss"]
            train_ppl = metrics["train"]["ppl"]
            val_ppl = metrics["val"]["ppl"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            iter_steps.append(step)

            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.3f}",
                    "val_loss": f"{val_loss:.3f}",
                    "train_ppl": f"{train_ppl:.2f}",
                    "val_ppl": f"{val_ppl:.2f}",
                }
            )

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    plot_metrics(iter_steps, train_losses, val_losses, train_ppls, val_ppls)

    context_text = "the "
    context = torch.tensor([encode(context_text)], dtype=torch.long, device=config.device)
    generated_idx = model.generate(context, max_new_tokens=200)
    generated_text = decode(generated_idx[0].cpu().numpy())

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pt")
    with open("outputs/generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

    print("\nGenerated sample:")
    print(generated_text)
    print("\nDone. Outputs saved to outputs/.")


if __name__ == "__main__":
    main()
