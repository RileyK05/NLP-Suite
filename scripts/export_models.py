"""Export the registered models to ONNX and check them against the originals.

For each model in :data:`core.models.registry.MODELS`:

1. Download it from Hugging Face at the pinned revision.
2. Export an fp32 ONNX graph (token models return every hidden state, so
   Word2Vec via BERT can still read any layer).
3. Check the fp32 graph *through the suite's own runtime code*
   (:mod:`core.models.onnx_backend`) against the original read the
   documented way (``transformers`` / ``sentence-transformers``). This
   tests the exporter and the runtime's tokenizing and pooling together;
   it must agree almost exactly (cosine >= 0.999).
4. Try smaller storage, smallest first — 4-bit weight-only, 8-bit weight-only
   with a 4-bit vocabulary table, int8
   (dynamic, per-channel), 8-bit weight-only, then fp16 weights (upcast to
   fp32 when the session loads, so the arithmetic is fp32) — and keep the
   first that passes the parity bar: cosine >= 0.99 on every parity
   sentence for embeddings; for classifiers, labels agreeing on >= 98% of
   sentences and no probability moved by more than 0.05. Dynamic int8
   quantizes activations too, and every model here fails it (BERT's last
   layer falls to 0.78, Qwen to 0.75).
5. Write ``models/<id>/`` (model.onnx, tokenizer.json, LICENSE, NOTICE,
   spec.json) and record every file's size and SHA-256 in
   ``core/models/_files.py``.

This is a maintainer tool. It needs torch, transformers,
sentence-transformers, onnx and onnxruntime — kept in their own
environment, because onnx needs a newer protobuf than some Google client
libraries accept::

    python -m venv --system-site-packages .export-env
    .export-env/Scripts/python -m pip install onnx onnxscript "protobuf>=6.31.1"
    .export-env/Scripts/python scripts/export_models.py --models bert-base-uncased

Then ``scripts/publish_models.py`` uploads ``models/`` to the release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.models.registry import MODELS, ModelSpec, get_model  # noqa: E402

SENTENCES = ROOT / "scripts" / "model_parity_sentences.txt"
FILES_MODULE = ROOT / "core" / "models" / "_files.py"
APACHE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: fp32 through our runtime must match the original this closely.
EXACT = 0.999
#: A smaller precision is kept only above this (embeddings).
PARITY = 0.99
#: ... or with at least this label agreement (classifiers),
AGREEMENT = 0.98
#: ... and no probability further than this from the original's.
PROBABILITY_GAP = 0.05

#: How each stored precision reads in a model's NOTICE.
PRECISION_WORDS = {
    "q4": "4-bit weights (MatMul, blocks of 32)",
    "q8e4": "8-bit weights (MatMul, blocks of 32) with a 4-bit vocabulary table",
    "int8": "8-bit weights with dynamically quantized activations",
    "q8": "8-bit weights (MatMul, blocks of 32)",
    "fp16": "16-bit floating-point weights",
}


def sentences() -> list[str]:
    lines = SENTENCES.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------- export


def export_fp32(spec: ModelSpec, work: Path) -> Path:
    """The original model as an fp32 ONNX graph (external data if > 2 GB)."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    target = work / "fp32" / "model.onnx"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(spec.source, revision=spec.revision)
    loader = AutoModelForSequenceClassification if spec.kind == "classifier" else AutoModel
    # transformers 5 loads a model in its stored dtype (Granite and Qwen are
    # bfloat16, which ONNX Runtime cannot run on CPU): export fp32.
    model = loader.from_pretrained(
        spec.source, revision=spec.revision, attn_implementation="eager", dtype=torch.float32
    ).eval()
    names = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenizer.model_input_names:
        names.append("token_type_ids")

    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(self, *inputs: Any) -> Any:
            feeds = dict(zip(names, inputs, strict=True))
            if spec.kind == "token_embeddings":
                return torch.stack(self.inner(**feeds, output_hidden_states=True).hidden_states, dim=0)
            if spec.kind == "classifier":
                return self.inner(**feeds).logits
            return self.inner(**feeds).last_hidden_state

    sample = tokenizer(["a short one", "and a somewhat longer second input"], padding=True, return_tensors="pt")
    output = (
        "hidden_states"
        if spec.kind == "token_embeddings"
        else "logits"
        if spec.kind == "classifier"
        else "last_hidden_state"
    )
    axes: dict[str, dict[int, str]] = {name: {0: "batch", 1: "sequence"} for name in names}
    axes[output] = {1: "batch", 2: "sequence"} if spec.kind == "token_embeddings" else {0: "batch", 1: "sequence"}
    if spec.kind == "classifier":
        axes[output] = {0: "batch"}
    started = time.time()
    with torch.no_grad():
        torch.onnx.export(
            Wrapped(),
            tuple(sample[name] for name in names),
            str(target),
            input_names=names,
            output_names=[output],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    tokenizer.backend_tokenizer.save(str(target.parent / "tokenizer.json"))
    print(f"  exported fp32 in {time.time() - started:.0f}s")
    return target


def _load(path: Path) -> Any:
    import onnx

    return onnx.load(str(path), load_external_data=True)


def _save(model: Any, path: Path) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def fp16_weights(model: Any, *, minimum: int = 1024) -> Any:
    """Store large fp32 initializers as fp16, cast back to fp32 on load.

    ONNX Runtime folds the casts when the session starts, so memory and
    arithmetic stay fp32; only the file halves.
    """
    import onnx
    from onnx import helper, numpy_helper

    graph = model.graph
    casts = []
    kept = []
    for tensor in graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT or int(np.prod(tensor.dims)) < minimum:
            kept.append(tensor)
            continue
        values = numpy_helper.to_array(tensor).astype(np.float16)
        half = numpy_helper.from_array(values, name=tensor.name + "__fp16")
        kept.append(half)
        casts.append(helper.make_node("Cast", [half.name], [tensor.name], to=onnx.TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model


def make_fp16(fp32: Path, target: Path) -> Path:
    _save(fp16_weights(_load(fp32)), target)
    return target


def _weight_only(bits: int, *, vocabulary_bits: int = 0) -> Any:
    def make(fp32: Path, target: Path) -> Path:
        """Weight-only quantization: MatMul weights in *bits*-bit blocks of 32.

        Unlike dynamic int8, activations stay fp32, which is what keeps the
        last layers of an encoder (and a decoder like Qwen) accurate. With
        *vocabulary_bits* the embedding table (a Gather, a third of Qwen's
        weights) is quantized too; ONNX Runtime supports 4 bits there.
        """
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model = _load(fp32)
        passes = [(bits, ("MatMul",))]
        if vocabulary_bits:
            passes.append((vocabulary_bits, ("Gather",)))
        for pass_bits, ops in passes:
            if ops == ("Gather",):
                # 4-bit Gather needs opset 21, and the quantizer only relabels
                # the opset (ReduceMean's axes attribute then breaks Qwen's
                # RMSNorm). Convert properly -- after the MatMul pass, when the
                # model is under protobuf's 2 GB limit.
                from onnx import version_converter

                model = version_converter.convert_version(model, 21)
            quantizer = MatMulNBitsQuantizer(
                model, bits=pass_bits, block_size=32, is_symmetric=True, op_types_to_quantize=ops
            )
            quantizer.process()
            model = quantizer.model.model
        _save(fp16_weights(model), target)
        return target

    return make


def make_int8(fp32: Path, target: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    staged = target.with_name("int8_staged.onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(fp32),
        str(staged),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        use_external_data_format=fp32.stat().st_size > 1_900_000_000,
    )
    # Embedding tables are not MatMuls; store them fp16 too.
    _save(fp16_weights(_load(staged)), target)
    for leftover in target.parent.glob("int8_staged*"):
        leftover.unlink()
    return target


# ---------------------------------------------------------------- parity


@dataclass
class Parity:
    passed: bool
    detail: dict[str, float]


def _cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    return (left * right).sum(axis=1)


def _word_pairs(texts: list[str]) -> tuple[list[str], list[str]]:
    words: list[str] = []
    contexts: list[str] = []
    for text in texts:
        split = text.split()
        for index in sorted({0, len(split) // 2, len(split) - 1}):
            words.append(split[index].lower())
            contexts.append(text)
    return words, contexts


class References:
    """The original models, read the documented way; computed once per model."""

    def __init__(self, spec: ModelSpec, texts: list[str]) -> None:
        self.spec = spec
        self.texts = texts
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        if key not in self._cache:
            self._cache[key] = self._compute(key)
        return self._cache[key]

    def _compute(self, key: str) -> Any:
        spec = self.spec
        if spec.kind == "token_embeddings":
            from core.analysis.contextual import TransformerBackend

            backend = TransformerBackend(spec.id, spec.revision, spec.source)
            words, contexts = _word_pairs(self.texts)
            if key == "texts":
                return np.array(backend.embed_texts(self.texts))
            return np.array(backend.embed_layer(words, contexts, int(key)))
        if spec.kind == "sentence_embeddings":
            from sentence_transformers import SentenceTransformer

            model = self._cache.setdefault(
                "_st", SentenceTransformer(spec.source, revision=spec.revision, device="cpu")
            )
            if key == "query":
                return model.encode(self.texts, prompt=spec.query_prompt, normalize_embeddings=True)
            return model.encode(self.texts, normalize_embeddings=True)
        from transformers import pipeline

        classifier = pipeline("sentiment-analysis", model=spec.source, revision=spec.revision, device="cpu")
        return [(row["label"], float(row["score"])) for row in classifier(self.texts)]


def check(spec: ModelSpec, folder: Path, refs: References, bar: float) -> Parity:
    """Read the parity sentences through our runtime and compare."""
    from core.models import onnx_backend

    onnx_backend._sessions.clear()
    onnx_backend._tokenizers.clear()
    texts = refs.texts
    detail: dict[str, float] = {}
    if spec.kind == "token_embeddings":
        backend = onnx_backend.OnnxTokenBackend(spec, folder)
        words, contexts = _word_pairs(texts)
        for layer in ("-1", "-5"):
            ours = np.array(backend.embed_layer(words, contexts, int(layer)))
            detail[f"words layer {layer} min cosine"] = float(_cos(ours, refs.get(layer)).min())
        detail["sentences min cosine"] = float(_cos(backend.embed_texts(texts), refs.get("texts")).min())
        return Parity(all(value >= bar for value in detail.values()), detail)
    if spec.kind == "sentence_embeddings":
        backend = onnx_backend.OnnxSentenceBackend(spec, folder)
        detail["documents min cosine"] = float(_cos(backend.embed_texts(texts), refs.get("documents")).min())
        if spec.query_prompt:
            detail["queries min cosine"] = float(_cos(backend.embed_texts(texts, query=True), refs.get("query")).min())
        return Parity(all(value >= bar for value in detail.values()), detail)
    classifier = onnx_backend.OnnxClassifier(spec, folder)
    ours_labels = classifier.classify(texts)
    theirs = refs.get("labels")
    agree = sum(1 for (a, _), (b, _) in zip(ours_labels, theirs, strict=True) if a == b) / len(texts)
    gap = max(abs(a - b) for (_, a), (_, b) in zip(ours_labels, theirs, strict=True))
    detail["label agreement"] = agree
    detail["max probability gap"] = gap
    # The probability is the tool's signed Compound score, so it must hold
    # too: 4-bit DistilBERT kept every label but moved one probability 0.32.
    needed, allowed = (AGREEMENT, PROBABILITY_GAP) if bar < EXACT else (1.0, 0.001)
    return Parity(agree >= needed and gap <= allowed, detail)


# ---------------------------------------------------------------- release


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _license_text(spec: ModelSpec) -> str:
    for url in (f"https://huggingface.co/{spec.source}/resolve/{spec.revision}/LICENSE", APACHE):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https URLs
                text = response.read().decode("utf-8")
            if "Apache License" in text:
                return text
        except OSError:
            continue
    raise SystemExit(f"could not fetch the Apache-2.0 license text for {spec.id}")


def write_release(
    spec: ModelSpec, chosen: Path, precision: str, parity: dict[str, Any], out: Path
) -> dict[str, tuple[int, str]]:
    folder = out / spec.id
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(chosen, folder / "model.onnx")
    shutil.copy2(chosen.parent.parent / "fp32" / "tokenizer.json", folder / "tokenizer.json")
    (folder / "LICENSE").write_text(_license_text(spec), encoding="utf-8")
    return write_papers(spec, folder, precision, parity)


def write_papers(spec: ModelSpec, folder: Path, precision: str, parity: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """NOTICE and spec.json beside a model's files; every file's size and SHA-256."""
    (folder / "NOTICE").write_text(
        f"{spec.display_name}\n"
        f"Source: https://huggingface.co/{spec.source} (revision {spec.revision})\n"
        f"License: {spec.license} (see LICENSE)\n\n"
        f"Modifications by NLP Suite: converted to ONNX; {PRECISION_WORDS[precision]}.\n"
        "No retraining or fine-tuning. Outputs were checked against the original\n"
        "model on the sentences in scripts/model_parity_sentences.txt (see spec.json).\n",
        encoding="utf-8",
    )
    files: dict[str, tuple[int, str]] = {}
    for name in ("model.onnx", "tokenizer.json", "LICENSE", "NOTICE"):
        path = folder / name
        files[name] = (path.stat().st_size, _sha256(path))
    (folder / "spec.json").write_text(
        json.dumps(
            {
                "id": spec.id,
                "source": spec.source,
                "revision": spec.revision,
                "kind": spec.kind,
                "precision": precision,
                "parity": parity,
                "files": {name: {"size": size, "sha256": sha} for name, (size, sha) in files.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = folder / "spec.json"
    files["spec.json"] = (spec_path.stat().st_size, _sha256(spec_path))
    return files


def record(model_id: str, precision: str, files: dict[str, tuple[int, str]]) -> None:
    """Rewrite core/models/_files.py with this model's entry updated."""
    import importlib

    import core.models._files as generated

    current = importlib.reload(generated)
    namespace = {"RELEASE": current.RELEASE}
    all_files: dict[str, dict[str, tuple[int, str]]] = dict(current.FILES)
    precisions: dict[str, str] = dict(current.PRECISION)
    all_files[model_id] = dict(sorted(files.items()))
    precisions[model_id] = precision
    order = [spec.id for spec in MODELS]
    lines = [
        '"""Published model files. Generated by ``scripts/export_models.py``; do not edit.',
        "",
        "``FILES[model_id][file_name] = (size_bytes, sha256)``. A model missing here",
        "has not been exported and published yet.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'RELEASE = "{namespace["RELEASE"]}"',
        "",
        "PRECISION: dict[str, str] = {",
        *[f'    "{key}": "{precisions[key]}",' for key in order if key in precisions],
        "}",
        "",
        "FILES: dict[str, dict[str, tuple[int, str]]] = {",
    ]
    for key in order:
        if key not in all_files:
            continue
        lines.append(f'    "{key}": {{')
        for name, (size, sha) in all_files[key].items():
            lines.append(f'        "{name}": ({size}, "{sha}"),')
        lines.append("    },")
    lines.append("}")
    FILES_MODULE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(spec: ModelSpec, work: Path, out: Path) -> None:
    print(f"== {spec.id} ({spec.source}@{spec.revision[:8]})")
    work = work / spec.id
    fp32 = export_fp32(spec, work)
    refs = References(spec, sentences())
    exact = check(spec, fp32.parent, refs, EXACT)
    print(f"  fp32 through our runtime: {exact.detail}")
    if not exact.passed:
        raise SystemExit(f"{spec.id}: the fp32 export does not match the original; fix the exporter or runtime first")
    tried: dict[str, Any] = {"fp32": exact.detail}
    # Smallest first; the first to pass is kept.
    candidates = (
        ("q4", _weight_only(4)),
        ("q8e4", _weight_only(8, vocabulary_bits=4)),
        ("int8", make_int8),
        ("q8", _weight_only(8)),
        ("fp16", make_fp16),
    )
    for precision, make in candidates:
        candidate = work / precision / "model.onnx"
        if not candidate.exists():
            make(fp32, candidate)
        shutil.copy2(fp32.parent / "tokenizer.json", candidate.parent / "tokenizer.json")
        result = check(spec, candidate.parent, refs, PARITY)
        size = candidate.stat().st_size / 1e6
        print(f"  {precision}: {size:.0f} MB, {'PASS' if result.passed else 'fail'} {result.detail}")
        tried[precision] = {**result.detail, "size_mb": round(size)}
        if result.passed:
            files = write_release(spec, candidate, precision, tried, out)
            record(spec.id, precision, files)
            total = sum(size for size, _ in files.values()) / 1e6
            print(f"  kept {precision}: {total:.0f} MB in {out / spec.id}")
            return
    raise SystemExit(f"{spec.id}: no precision passed parity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=[spec.id for spec in MODELS], help="model ids (default: all)")
    parser.add_argument("--work", type=Path, default=ROOT / ".model-work", help="scratch directory (large)")
    parser.add_argument("--out", type=Path, default=ROOT / "models", help="release directory")
    parser.add_argument(
        "--notices-only",
        action="store_true",
        help="rewrite NOTICE and spec.json of models already in --out (no export, no parity run)",
    )
    args = parser.parse_args(argv)
    for name in args.models:
        spec = get_model(name)
        if spec is None:
            parser.error(f"unknown model {name!r}")
        if args.notices_only:
            folder = args.out / spec.id
            recorded = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
            record(
                spec.id, recorded["precision"], write_papers(spec, folder, recorded["precision"], recorded["parity"])
            )
            print(f"rewrote NOTICE and spec.json for {spec.id}")
            continue
        export(spec, args.work, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
