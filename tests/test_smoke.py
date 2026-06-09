import importlib
import sys
from pathlib import Path

import pytest
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


def test_dataset_rejects_invalid_inputs(monkeypatch):
    _, dataset, _ = reload_project_modules()
    monkeypatch.chdir(ROOT)

    _, _, encode, _, _ = dataset.load_data()

    with pytest.raises(ValueError, match="unknown characters"):
        encode("\0")

    with pytest.raises(ValueError, match="split must be"):
        dataset.get_batch("test")


def test_dataset_loads_corpus_directory(tmp_path):
    config, dataset, _ = reload_project_modules()
    config.block_size = 8

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(4):
        (corpus_dir / f"doc_{i}.txt").write_text(
            f"the quick brown fox document {i} " * 8,
            encoding="utf-8",
        )

    train_data, val_data, encode, decode, vocab_size = dataset.load_data(
        data_path=corpus_dir
    )
    loaded = dataset.get_dataset(data_path=corpus_dir)

    assert loaded.num_documents == 4
    assert len(train_data) > config.block_size
    assert len(val_data) > config.block_size
    assert vocab_size == loaded.vocab_size
    assert decode(encode("the ")) == "the "


def test_causal_attention_does_not_use_future_tokens(monkeypatch):
    _, dataset, transformer = reload_project_modules()
    monkeypatch.chdir(ROOT)

    _, _, _, _, vocab_size = dataset.load_data()
    model = transformer.EncoderOnlyTransformer(vocab_size)
    model.eval()

    x = torch.randint(0, vocab_size, (1, 8))
    x_with_different_future = x.clone()
    x_with_different_future[0, -1] = (x_with_different_future[0, -1] + 1) % vocab_size

    with torch.no_grad():
        logits, _ = model(x)
        changed_logits, _ = model(x_with_different_future)

    torch.testing.assert_close(
        logits[:, :-1, :], changed_logits[:, :-1, :], rtol=1e-5, atol=1e-6
    )


def test_generate_validates_sampling_args(monkeypatch):
    _, dataset, transformer = reload_project_modules()
    monkeypatch.chdir(ROOT)

    _, _, encode, _, vocab_size = dataset.load_data()
    model = transformer.EncoderOnlyTransformer(vocab_size)
    context = torch.tensor([encode("the ")], dtype=torch.long)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(context, max_new_tokens=1, temperature=0)
