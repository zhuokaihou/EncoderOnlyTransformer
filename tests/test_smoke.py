import importlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reload_project_modules():
    for name in [
        "config",
        "utils.dataset",
        "utils",
        "models.transformer",
        "models",
    ]:
        sys.modules.pop(name, None)

    config = importlib.import_module("config")
    config.batch_size = 2
    config.block_size = 8
    config.n_embd = 16
    config.n_head = 4
    config.n_layer = 1
    config.dropout = 0.0
    config.device = "cpu"

    dataset = importlib.import_module("utils.dataset")
    transformer = importlib.import_module("models.transformer")
    return config, dataset, transformer


def test_data_model_forward_and_generate(monkeypatch):
    _, dataset, transformer = reload_project_modules()
    monkeypatch.chdir(ROOT)

    _, _, encode, decode, vocab_size = dataset.load_data()
    x, y = dataset.get_batch("train")
    model = transformer.EncoderOnlyTransformer(vocab_size)

    logits, loss = model(x, y)
    assert logits.shape == (2, 8, vocab_size)
    assert loss.item() > 0

    context = torch.tensor([encode("the ")], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=5)
    assert generated.shape == (1, len("the ") + 5)
    assert isinstance(decode(generated[0].tolist()), str)
