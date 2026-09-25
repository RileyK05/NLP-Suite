"""Build the tiny random-weight models the test suite runs ONNX code against.

A 2-layer, 32-wide BERT with a 200-word vocabulary, exported three ways —
every hidden state (token model), the last hidden state (sentence model)
and two-class logits (classifier) — into ``tests/fixtures/models/``.
Random weights make the vectors meaningless but the plumbing real: the
tests check shapes, alignment, pooling, batching, truncation and caching
through the same runtime code the app uses. Made once, on a machine with
torch; the files are committed (a few hundred KB).

    python scripts/make_tiny_onnx.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "models"

WORDS = """
the a an and or but of to in on at by for with from as is was are were be been
bank river money rate rates interest raised sat watched boats water loan loans
i love this movie hate film good bad great terrible fine wonderful awful best worst
we they he she it you our their his her its my your who what why how when where
nation people country government law court state states war peace world year years
economy jobs work workers tax taxes budget trade price prices school schools children
president congress senate vote votes bill bills plan plans policy health care energy
time day days week month today tomorrow night morning new old big small long short
not no yes all some many more most less few every each other same own only very
have has had do does did will would can could should may might must shall say said
make made go went come came see saw know knew think thought take took give gave
bass fish lake band music jazz teacher play played house home city town village
""".split()


def main() -> int:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    torch.manual_seed(7)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for word in WORDS:
        if word not in vocab:
            vocab.append(word)
    # A few continuation pieces so unknown words split into several pieces.
    vocab.extend(f"##{piece}" for piece in ("s", "ed", "ing", "er", "ly", "a", "e", "i", "o", "n", "t", "r"))
    vocab.extend(list("abcdefghijklmnopqrstuvwxyz"))
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=2,
    )
    with tempfile.TemporaryDirectory() as scratch:
        vocab_file = Path(scratch) / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    encoder = BertModel(config).eval()
    classifier = BertForSequenceClassification(config).eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, read: str) -> None:
            super().__init__()
            self.inner = inner
            self.read = read

        def forward(self, input_ids, attention_mask, token_type_ids):  # type: ignore[no-untyped-def]
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            if self.read == "hidden":
                return torch.stack(self.inner(**feeds, output_hidden_states=True).hidden_states, dim=0)
            if self.read == "logits":
                return self.inner(**feeds).logits
            return self.inner(**feeds).last_hidden_state

    sample = tokenizer(["the bank", "we sat on the river bank"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    for folder, module, output, axes in (
        ("tiny-bert", Wrapped(encoder, "hidden"), "hidden_states", {1: "batch", 2: "sequence"}),
        ("tiny-sentence", Wrapped(encoder, "last"), "last_hidden_state", {0: "batch", 1: "sequence"}),
        ("tiny-classifier", Wrapped(classifier, "logits"), "logits", {0: "batch"}),
    ):
        target = OUT / folder
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        dynamic = {name: {0: "batch", 1: "sequence"} for name in names}
        dynamic[output] = axes
        with torch.no_grad():
            torch.onnx.export(
                module,
                tuple(sample[name] for name in names),
                str(target / "model.onnx"),
                input_names=names,
                output_names=[output],
                dynamic_axes=dynamic,
                opset_version=17,
                dynamo=False,
            )
        tokenizer.backend_tokenizer.save(str(target / "tokenizer.json"))
        print(folder, sum(path.stat().st_size for path in target.iterdir()) // 1000, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
