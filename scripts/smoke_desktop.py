"""Exercise the packaged engine without importing any project dependencies."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--with-parser", action="store_true")
    parser.add_argument(
        "--corpus", type=Path, help="Verify a real TXT corpus through folder import (originals stay untouched)"
    )
    parser.add_argument("--job-timeout", type=int, default=120, help="Maximum seconds per analysis")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace / "smoke.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603
            [str(args.executable.resolve()), "--desktop", "--data-dir", str(args.workspace.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            handshake: queue.Queue[str] = queue.Queue()
            threading.Thread(target=lambda: handshake.put(process.stdout.readline()), daemon=True).start()
            connection = json.loads(handshake.get(timeout=60))

            def request(path: str, data: dict | bytes | None = None) -> object:
                headers = {"Authorization": "Bearer " + connection["token"]}
                if isinstance(data, dict):
                    headers["Content-Type"] = "application/json"
                    data = json.dumps(data).encode()
                req = urllib.request.Request(connection["baseUrl"] + "/api" + path, data=data, headers=headers)  # noqa: S310 -- local engine handshake
                with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
                    body = response.read()
                    return json.loads(body) if "json" in response.headers.get("Content-Type", "") else body

            deadline = time.monotonic() + 30
            while True:
                try:
                    request("/health")
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            project = request("/projects", {"name": "Packaged runtime smoke"})
            base = f"/projects/{project['id']}"
            fixtures = (
                (
                    "speech_2020-01-01.txt",
                    "Alice visited New York. The development of education gives people a better future. We work together for peace.",
                ),
                (
                    "speech_2020-02-01.txt",
                    "Bob works at Stanford University. Scientists study oceans and forests. Research helps protect nature.",
                ),
            )
            expected_documents = 2
            if args.corpus:
                source = args.corpus.resolve(strict=True)
                source_files = list(source.rglob("*.txt"))
                if not source.is_dir() or not source_files:
                    raise ValueError("Supply a non-empty TXT corpus directory")
                expected_documents = len(source_files)
                report = request(base + "/import-paths", {"paths": [str(source)]})
                if report["errors"] or report["imported"] != expected_documents:
                    raise RuntimeError(f"Corpus import mismatch: {report}")
                duplicate_report = request(base + "/import-paths", {"paths": [str(source)]})
                if duplicate_report != {"imported": 0, "duplicates": expected_documents, "errors": []}:
                    raise RuntimeError(f"Repeat import mismatch: {duplicate_report}")
                print(
                    f"PASS corpus folder import: {expected_documents} documents; repeat import creates no duplicates",
                    flush=True,
                )
            else:
                for name, text in fixtures:
                    request(base + "/documents?name=" + name, text.encode())
            if args.with_parser:
                # Live exploration parses once and keeps the parse as parquet.
                # The frozen build once excluded pyarrow, and every live load
                # failed on the desktop while every job above still passed.
                warmed = request(base + "/live/warm", {"parser": "spacy"})
                if warmed["state"] != "ready" or warmed["error"]:
                    raise RuntimeError(f"Live warm failed: {warmed['error'] or warmed['state']}")
                if not list(args.workspace.rglob("annotations/*.parquet")):
                    raise RuntimeError("Live warm kept no parse cache: the runtime cannot write parquet")
                print(f"PASS live warm: {warmed['documents']} documents parsed and cached", flush=True)
            selected = ["readability", "lexical_diversity", "doc_similarity", "doc_duplicates"]
            if args.with_parser:
                selected.extend(
                    ["sentence_complexity", "ner", "clause_svo", "nominalization", "narrative", "sentiment_vader_anew"]
                )
                # The bundled models, read by the frozen engine's ONNX Runtime:
                # DistilBERT (sentiment) and BERT base (topics).
                selected.extend(["sentiment_neural_bert", "bert_topics"])
            if args.corpus:
                selected = ["readability", "doc_similarity"] + (["ner"] if args.with_parser else [])
            table_project = request("/projects", {"name": "Packaged CSV smoke (no corpus)"})
            table_base = f"/projects/{table_project['id']}"
            resource = request(
                "/resources?name=statistics.csv",
                b"group,vote,word,x,y,date\n"
                b"A,yes,one,1,2,2020-01-01\nA,no,two,2,4,2020-01-02\n"
                b"B,yes,three,3,5,2020-01-03\nB,no,four,4,8,2020-01-04\n"
                b"A,yes,five,5,9,2020-01-05\nB,no,six,6,11,2020-01-06\n"
                b"A,yes,seven,7,13,2020-01-07\nB,no,eight,8,15,2020-01-08\n",
            )
            # The smoke corpus has two documents; bert_topics defaults to three
            # topics and rightly refuses more topics than documents.
            corpus_params: dict[str, dict[str, object]] = {"bert_topics": {"topics": 2}}
            cases = [(tool, base, corpus_params.get(tool, {}), expected_documents) for tool in selected]
            for tool, params in (
                ("table_chi2", {"col1": "group", "col2": "vote"}),
                ("table_crosstab", {"col1": "group", "col2": "vote"}),
                ("table_keyness", {"word-col": "word", "freq1": "x", "freq2": "y"}),
                ("table_mw", {"value-col": "x", "group-col": "group"}),
                ("table_kw", {"value-col": "x", "group-col": "group"}),
                ("table_trend", {"date-col": "date", "value-col": "x"}),
                ("table_rankcorr", {"col-x": "x", "col-y": "y"}),
                ("table_charts", {"kind": "bar", "x": "word", "y": "x", "format": "html"}),
                ("table_charts", {"kind": "bar", "x": "word", "y": "x", "format": "xlsx"}),
                ("table_wordcloud_gephi", {"mode": "wordcloud", "word-col": "word", "weight-col": "x", "image": True}),
            ):
                cases.append((tool, table_base, {**params, "input": resource["path"]}, 1))
            for tool, job_base, params, expected_inputs in cases:
                job = request(job_base + "/jobs", {"tool": tool, "params": params})
                deadline = time.monotonic() + args.job_timeout
                while time.monotonic() < deadline:
                    status = next(item for item in request(job_base + "/jobs") if item["id"] == job["id"])
                    if status["state"] not in ("QUEUED", "RUNNING"):
                        break
                    time.sleep(0.2)
                if status["state"] != "DONE":
                    raise RuntimeError(f"{tool}: {status}")
                result_path = job_base + "/jobs/" + job["id"]
                result = request(result_path + "/results")
                if len(result["inputs"]) != expected_inputs:
                    raise RuntimeError("Incorrect input provenance")
                table = request(result_path + "/artifacts/0")
                if tool == "readability" and table["total"] != expected_documents:
                    raise RuntimeError("Readability must include every imported document")
                if tool == "doc_similarity" and table["total"] != expected_documents * (expected_documents - 1) // 2:
                    raise RuntimeError("Similarity must include every distinct document pair")
                if tool == "ner" and not table["rows"]:
                    raise RuntimeError("The entity fixture produced no recognized entities")
                with zipfile.ZipFile(io.BytesIO(request(result_path + "/export"))) as archive:
                    if "result.json" not in archive.namelist() or archive.testzip() is not None:
                        raise RuntimeError("Invalid export")
                    if tool in ("table_charts", "table_wordcloud_gephi"):
                        filename = "wordcloud.png" if tool == "table_wordcloud_gephi" else "chart." + params["format"]
                        exported = next((name for name in archive.namelist() if Path(name).name == filename), None)
                        if exported is None:
                            raise RuntimeError(f"Requested export is missing: {filename}")
                        content = archive.read(exported)
                        if not content:
                            raise RuntimeError(f"Requested export is empty: {filename}")
                        if filename.endswith(".png") and not content.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise RuntimeError("Invalid PNG export")
                        if filename.endswith(".xlsx"):
                            with zipfile.ZipFile(io.BytesIO(content)) as workbook:
                                if "xl/workbook.xml" not in workbook.namelist() or workbook.testzip() is not None:
                                    raise RuntimeError("Invalid Excel workbook export")
                print(f"PASS {tool}: computation, provenance, table, ZIP", flush=True)
            restored = request("/projects/restore", request(base + "/backup"))
            if restored["id"] == project["id"] or restored["documents"] != expected_documents:
                raise RuntimeError("Project backup/restore failed")
            print("PASS project backup and non-overwriting restore", flush=True)
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.returncode:
            raise RuntimeError(f"Engine exited {process.returncode}; see {args.workspace / 'smoke.log'}")
        print("PASS graceful engine shutdown", flush=True)


if __name__ == "__main__":
    main()
