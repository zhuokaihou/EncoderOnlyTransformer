import argparse
from pathlib import Path

import torch

import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument("--checkpoint-path", default=config.checkpoint_path)
    parser.add_argument("--data-path", default=config.data_path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--output-path", default=config.generated_result_path)
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")

    return args


def apply_model_config(saved_config):
    for key in ["block_size", "n_embd", "n_head", "n_layer", "dropout", "vocab_size"]:
        if key in saved_config:
            setattr(config, key, saved_config[key])


def codec_from_vocab(vocab):
    stoi = {str(ch): int(idx) for ch, idx in vocab["stoi"].items()}
    itos = {int(idx): str(ch) for idx, ch in vocab["itos"].items()}

    def encode(text):
        unknown = sorted(set(text) - set(stoi))
        if unknown:
            raise ValueError(f"Input contains unknown characters: {unknown}")
        return [stoi[c] for c in text]

    def decode(indices):
        unknown = sorted({int(i) for i in indices if int(i) not in itos})
        if unknown:
            raise ValueError(f"Token ids are outside the vocabulary: {unknown}")
        return "".join(itos[int(i)] for i in indices)

    return encode, decode, len(stoi)


def normalize_checkpoint(raw_checkpoint):
    if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
        return raw_checkpoint
    return {"model_state_dict": raw_checkpoint}


def init_model(args):
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = None

    if checkpoint_path.exists():
        raw_checkpoint = torch.load(checkpoint_path, map_location=config.device)
        checkpoint = normalize_checkpoint(raw_checkpoint)
        apply_model_config(checkpoint.get("model_config", {}))

    if checkpoint and "vocab" in checkpoint:
        encode, decode, vocab_size = codec_from_vocab(checkpoint["vocab"])
    else:
        _, _, encode, decode, vocab_size = load_data(
            data_path=args.data_path, download_if_missing=args.download_data
        )
        config.vocab_size = vocab_size

    model = EncoderOnlyTransformer(vocab_size).to(config.device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: checkpoint not found. Generating with an untrained model.")

    model.eval()
    return model, encode, decode


def generate_text(model, encode, decode, args):
    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=config.device)
    generated_idx = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    return decode(generated_idx[0].cpu().numpy())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, encode, decode = init_model(args)
    result = generate_text(model, encode, decode, args)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")

    print(result)
    print(f"\nGenerated text saved to {output_path}")


if __name__ == "__main__":
    main()
