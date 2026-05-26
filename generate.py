from pathlib import Path

import torch

import config
from models.transformer import EncoderOnlyTransformer
from utils.dataset import load_data


MODEL_PATH = Path("outputs/model.pt")
SAVE_OUTPUT_PATH = Path("outputs/generated_result.txt")


def init_model():
    _, _, encode, decode, vocab_size = load_data()
    config.vocab_size = vocab_size
    model = EncoderOnlyTransformer(vocab_size).to(config.device)

    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=config.device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: outputs/model.pt not found. Generating with an untrained model.")

    model.eval()
    return model, encode, decode


def generate_text(model, encode, decode, prompt="the ", max_new_tokens=300):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config.device)
    generated_idx = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(generated_idx[0].cpu().numpy())


def main():
    model, encode, decode = init_model()
    result = generate_text(model, encode, decode)

    SAVE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_OUTPUT_PATH.write_text(result, encoding="utf-8")

    print(result)
    print(f"\nGenerated text saved to {SAVE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
