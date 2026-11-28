"""Reproduce single-request prefill TTFT and the deterministic engineering workload."""

import argparse
import hashlib
import http.client
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from cases import cases, correct

PORT = 18591
BASE = f"http://127.0.0.1:{PORT}"


def timed(c, prompt):
    payload = json.dumps(
        {
            "model": "inkling",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    start = time.monotonic_ns()
    c.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
    response = c.getresponse()
    if response.status != 200:
        raise RuntimeError(response.read()[:500])
    ttft = None
    ids = []
    usage = None
    for line in response:
        if not line.startswith(b"data:"):
            continue
        chunk = line[5:].strip()
        if chunk == b"[DONE]":
            continue
        e = json.loads(chunk)
        if e.get("error"):
            raise RuntimeError(e)
        for choice in e["choices"]:
            emitted = choice.get("token_ids") or []
            if emitted:
                ids.extend(emitted)
                if ttft is None:
                    ttft = (time.monotonic_ns() - start) / 1e6
        if e.get("usage"):
            usage = e["usage"]
    assert (
        len(ids) == 1 and usage["prompt_tokens"] == len(prompt) and ttft is not None
    ), (ids, usage)
    return {
        "ttft_ms": ttft,
        "elapsed_ms": (time.monotonic_ns() - start) / 1e6,
        "token": ids[0],
    }


def evaluate(data, dest):
    rows = []
    for case in data:
        request = {
            "model": "inkling",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "reasoning_effort": "low",
            "max_tokens": 1536,
            "return_token_ids": True,
        }
        req = urllib.request.Request(
            BASE + "/v1/chat/completions",
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            reply = json.load(resp)
        ch = reply["choices"][0]
        msg = ch["message"]
        text = msg.get("content") or ""
        rows.append(
            {
                "id": case["id"],
                "correct": correct(case, text),
                "text": text,
                "ids_sha256": hashlib.sha256(
                    json.dumps(ch.get("token_ids")).encode()
                ).hexdigest()
                if ch.get("token_ids") is not None
                else None,
                "reasoning_sha256": hashlib.sha256(
                    (
                        msg.get("reasoning") or msg.get("reasoning_content") or ""
                    ).encode()
                ).hexdigest(),
                "finish": ch["finish_reason"],
                "usage": reply["usage"],
            }
        )
        dest.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    return rows


def probes():
    result = []
    for long in (False, True):
        for row in cases(long)[:3]:
            body = {
                "model": "inkling",
                "prompt": row["prompt"],
                "temperature": 0,
                "max_tokens": 24 if long else 16,
                "return_token_ids": True,
                "prompt_logprobs": 0,
            }
            req = urllib.request.Request(
                BASE + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as stream:
                c = json.load(stream)["choices"][0]
            ids = c["prompt_token_ids"]
            result.append(
                {
                    "prompt_ids": ids,
                    "prompt_logprobs": [
                        None if d is None else d[str(i)]["logprob"]
                        for i, d in zip(ids, c["prompt_logprobs"], strict=True)
                    ],
                    "generated_ids": c["token_ids"],
                }
            )
    return result


def main(out, quality, model):
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model)
    long = cases(True)
    stream = tokenizer.encode(
        long[2]["prompt"] + "\n" + long[1]["prompt"], add_special_tokens=False
    )
    q = tokenizer.encode(
        "\nRead the preceding records and return the final quota as an integer.",
        add_special_tokens=False,
    )
    lengths = (512, 1024, 2048, 3325, 4096)
    prompts = {n: stream[: n - len(q)] + q for n in lengths}
    assert all(len(x) == n for n, x in prompts.items())
    out.mkdir(exist_ok=False, parents=True)
    cmd = [
        str(Path(sys.executable).parent / "vllm"),
        "serve",
        model,
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.94",
        "--kv-cache-dtype",
        "fp8",
        "--limit-mm-per-prompt",
        '{"image":0}',
        "--served-model-name",
        "inkling",
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "inkling",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    (out / "command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "prompt_hashes": {
                    n: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    for n, ids in prompts.items()
                },
            },
            indent=2,
        )
    )
    with (out / "server.log").open("w") as f, (out / "gpu.csv").open("w") as telemetry:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=telemetry,
        )
        start = time.monotonic()
        try:
            while proc.poll() is None and time.monotonic() - start < 1200:
                try:
                    with urllib.request.urlopen(BASE + "/v1/models", timeout=3) as r:
                        if any(m["id"] == "inkling" for m in json.load(r)["data"]):
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("server failed; see server.log")
            print("ready", round(time.monotonic() - start, 1), flush=True)
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=180)
            rng = random.Random(69979)
            for _ in range(3):
                for n in lengths:
                    timed(c, prompts[n])
            records = []
            for rep in range(10):
                order = list(lengths)
                rng.shuffle(order)
                for n in order:
                    records.append(dict(tokens=n, rep=rep, **timed(c, prompts[n])))
            c.close()
            (out / "timing.json").write_text(json.dumps(records, indent=2) + "\n")
            print(
                "p50",
                {
                    n: round(
                        statistics.median(
                            r["ttft_ms"] for r in records if r["tokens"] == n
                        ),
                        2,
                    )
                    for n in lengths
                },
                flush=True,
            )
            if quality:
                (out / "numeric.json").write_text(json.dumps(probes()) + "\n")
                rows = evaluate(long, out / "quality.json")
                print(
                    "correct",
                    sum(r["correct"] for r in rows),
                    "/",
                    len(rows),
                    flush=True,
                )
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=65)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            monitor.terminate()
            monitor.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    main(args.out, args.quality, args.model)
