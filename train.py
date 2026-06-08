import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import get_batch, get_dataset, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal character Transformer.")
    parser.add_argument("--data-path", default=config.data_path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--output-dir", default=config.output_dir)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--block-size", type=int, default=config.block_size)
    parser.add_argument("--n-embd", type=int, default=config.n_embd)
    parser.add_argument("--n-head", type=int, default=config.n_head)
    parser.add_argument("--n-layer", type=int, default=config.n_layer)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=config.grad_clip)
    parser.add_argument("--max-iters", type=int, default=config.max_iters)
    parser.add_argument("--eval-interval", type=int, default=config.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=config.eval_iters)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--sample-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = str(Path(args.output_dir) / "model.pt")

    positive_ints = [
        "batch_size",
        "block_size",
        "n_embd",
        "n_head",
        "n_layer",
        "max_iters",
        "eval_interval",
        "eval_iters",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.dropout < 0 or args.dropout >= 1:
        parser.error("--dropout must be in the interval [0, 1)")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.grad_clip < 0:
        parser.error("--grad-clip must be non-negative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")

    return args


def apply_runtime_config(args):
    config.batch_size = args.batch_size
    config.block_size = args.block_size
    config.n_embd = args.n_embd
    config.n_head = args.n_head
    config.n_layer = args.n_layer
    config.dropout = args.dropout
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.grad_clip = args.grad_clip
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.eval_iters = args.eval_iters
    config.seed = args.seed
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.checkpoint_path = args.checkpoint_path


def model_config_snapshot(vocab_size):
    return {
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "vocab_size": vocab_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "seed": config.seed,
    }


@torch.no_grad()
def estimate_loss_and_ppl(model):
    out = {}
    was_training = model.training
    model.eval()
    try:
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses.append(loss.item())
            avg_loss = float(np.mean(losses))
            out[split] = {"loss": avg_loss, "ppl": float(np.exp(avg_loss))}
    finally:
        model.train(was_training)
    return out


def plot_metrics(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["step"], history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["step"], history["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["step"], history["train_ppl"], label="Train PPL", marker="o")
    plt.plot(history["step"], history["val_ppl"], label="Val PPL", marker="s")
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.title("PPL Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "loss_ppl_curve.png", dpi=300)
    plt.close()


def build_checkpoint(model, optimizer, dataset, step, history, best_val_loss):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": model_config_snapshot(dataset.vocab_size),
        "vocab": {"stoi": dataset.stoi, "itos": dataset.itos},
        "history": history,
    }


def save_checkpoint(path, model, optimizer, dataset, step, history, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(model, optimizer, dataset, step, history, best_val_loss)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    apply_runtime_config(args)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    checkpoint_path = Path(config.checkpoint_path)
    best_checkpoint_path = output_dir / "best_model.pt"
    generated_text_path = output_dir / "generated_text.txt"

    print("Loading data...")
    _, _, encode, decode, vocab_size = load_data(
        data_path=args.data_path, download_if_missing=args.download_data
    )
    dataset = get_dataset()
    config.vocab_size = vocab_size

    print(f"Initializing model on {config.device}...")
    model = EncoderOnlyTransformer(vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
    }
    best_val_loss = float("inf")

    print("Training...")
    pbar = tqdm(range(config.max_iters), desc="Training")
    for step in pbar:
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            metrics = estimate_loss_and_ppl(model)
            train_loss = metrics["train"]["loss"]
            val_loss = metrics["val"]["loss"]
            train_ppl = metrics["train"]["ppl"]
            val_ppl = metrics["val"]["ppl"]

            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_ppl"].append(train_ppl)
            history["val_ppl"].append(val_ppl)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    dataset,
                    step,
                    history,
                    best_val_loss,
                )

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
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    plot_metrics(history, output_dir)

    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=config.device)
    generated_idx = model.generate(
        context,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    generated_text = decode(generated_idx[0].cpu().numpy())

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        dataset,
        config.max_iters - 1,
        history,
        best_val_loss,
    )
    generated_text_path.parent.mkdir(parents=True, exist_ok=True)
    generated_text_path.write_text(generated_text, encoding="utf-8")

    print("\nGenerated sample:")
    print(generated_text)
    print(f"\nDone. Outputs saved to {output_dir}/.")


if __name__ == "__main__":
    main()
