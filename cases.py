"""Deterministic engineering tasks; benchmark prompts and answer keys stay separate."""

import argparse
import itertools
import json
import random
import re
import sqlite3
from pathlib import Path

SCHEMA = "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, region TEXT, state TEXT, cents INTEGER);"
REGIONS = ["east", "west", "north", "south"]
SQL = [
    ("统计已付款订单数", "COUNT(*)", "paid"),
    ("求已发货订单的合计金额（分）", "SUM(cents)", "sent"),
    ("求已付款订单的最大金额（分）", "MAX(cents)", "paid"),
    ("求已取消订单的最小订单编号", "MIN(order_id)", "cancelled"),
]


def cases(long):
    rows = []
    for i in range(16):
        region = REGIONS[i % 4]
        ask, expr, state = SQL[i // 4]
        prompt = (
            f"SQLite schema: {SCHEMA}\nstate 对照：paid=已付款，sent=已发货，"
            f"cancelled=已取消。请为 {region} 区域{ask}，只返回一条 SELECT。"
        )
        sql = f"SELECT {expr} FROM orders WHERE region='{region}' AND state='{state}'"
        rows.append(("sql", i, prompt, sql))
        nums = [(i * 7 + j * 3) % 17 for j in range(12)]
        tally = {}
        for n in nums:
            k = n % 4
            tally[k] = tally.get(k, 0) + (n if n % 2 else -n)
        prompt = (
            "What integer does this Python function return? Answer with the integer alone.\n"
            f"def score():\n    counts = {{}}\n    for n in {nums}:\n"
            "        k = n % 4\n        counts[k] = counts.get(k, 0) + (n if n % 2 else -n)\n"
            "    return max(counts.values()) - min(counts.values())"
        )
        rows.append(("code", i, prompt, str(max(tally.values()) - min(tally.values()))))
        rng = random.Random(9000 + i)
        ids = [f"RQ-{1000 + 20 * i + j}" for j in range(8)]
        target = rng.choice(ids)
        ledger = []
        for revision, key in itertools.product(range(1, 4), ids):
            team = rng.choice(["infra", "billing", "runtime"])
            quota = rng.randrange(12, 129)
            ledger.append(f"{key}, rev={revision}, team={team}, quota={quota}")
            if key == target and revision == 3:
                answer = {"team": team, "quota": quota}
        rng.shuffle(ledger)
        prompt = (
            "从下列变更记录中提取指定请求的最新版本。只返回含 team 和整数 quota 的 JSON 对象。\n"
            + "\n".join(ledger)
            + f"\n请求号：{target}。"
        )
        rows.append(("document", i, prompt, answer))

    result = []
    for category, i, prompt, answer in rows:
        if long:
            # Unique operational context on each request, same question and answer as short.
            records = [
                f"Archived observation {i:02d}-{j:03d}: node {j % 13} processed "
                f"{21 + (j + i) % 170} records, then flushed partition {(2 * j + i) % 29}."
                for j in range(140)
            ]
            prompt = (
                "以下为上周的运行记录，仅作背景。末尾是本次任务。\n"
                + "\n".join(records)
                + "\n\n"
                + prompt
            )
        result.append(
            {
                "id": f"{category}-{i:02d}",
                "category": category,
                "prompt": prompt,
                "answer": answer,
                "output_tokens": 384,
            }
        )
    return result


def sql_result(db, statement):
    cur = db.execute(statement)
    return cur.fetchmany(5)


def sql_correct(statement, gold):
    for seed in (912, 508, 677):
        rng = random.Random(seed)
        with sqlite3.connect(":memory:") as db:
            db.execute(SCHEMA)
            db.executemany(
                "INSERT INTO orders VALUES(?,?,?,?)",
                [
                    (
                        j,
                        rng.choice(REGIONS),
                        rng.choice(["paid", "sent", "cancelled"]),
                        rng.randrange(100, 12000),
                    )
                    for j in range(135)
                ],
            )
            db.commit()
            expected = sql_result(db, gold)
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
            }
            aggregates = {
                "count",
                "sum",
                "max",
                "min",
                "total",
                "coalesce",
                "ifnull",
                "lower",
                "upper",
            }

            def authorize(
                action,
                arg1,
                arg2,
                _db,
                _trigger,
                allowed=allowed,
                aggregates=aggregates,
            ):
                ok = action in allowed and (
                    action != sqlite3.SQLITE_FUNCTION or arg2 in aggregates
                )
                return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY

            db.set_authorizer(authorize)
            steps = itertools.count()
            db.set_progress_handler(lambda steps=steps: int(next(steps) > 100), 500)
            try:
                if sql_result(db, statement) != expected:
                    return False
            except (sqlite3.Error, ValueError):
                return False
    return True


def correct(case, response):
    text = response.strip()
    fenced = re.fullmatch(
        r"```(?:sql|json|text)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced[1].strip()
    if case["category"] == "sql":
        return sql_correct(text, case["answer"])
    if case["category"] == "code":
        return text == case["answer"]
    try:
        obj = json.loads(text)
        return obj == case["answer"] and type(obj["quota"]) is int
    except (ValueError, TypeError, KeyError):
        return False


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=True, parents=True)
    for label in ("short", "long"):
        pack = cases(label == "long")
        write_jsonl(
            args.directory / f"{label}.jsonl",
            [{k: r[k] for k in ("id", "prompt", "output_tokens")} for r in pack],
        )
