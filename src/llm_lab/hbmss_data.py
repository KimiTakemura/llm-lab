"""HBMSS peft-dataset を SFT 用に読み込み、トークン化する。

datasets の `load_dataset("json")` を使わず自前で json.loads する理由:
`tools[].function.parameters` は tool ごとに properties のキーが違う JSON Schema で、Arrow に
載せると全 tool のキーの和集合を持つ struct に推論され、欠けたキーが null で埋まる。それを
そのまま chat template の `tool | tojson` に流すと、学習時のツール定義が元データと変わる。
ここでは Python の dict のままテンプレートに通し、トークン化済みの列だけを Dataset にする。

損失マスクは tokenizer.apply_chat_template(return_assistant_tokens_mask=True) で作る。
これには chat template に {% generation %} マーカーが要るため、templates/qwen3_sft.jinja を
tokenizer に差し込む（公式テンプレートにはマーカーが無い）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import Dataset

TRAIN_FILES = ["layer1_domain", "layer2_system_rules", "layer3_consistency", "trajectories_next_action"]

# peft-dataset/data の場所。ローカル（WSL）の既定は HBMSS リポジトリ内。クラウドでは
# HBMSS_DATA_DIR=/workspace/peft-dataset/data のように環境変数で差し替える（--data-dir でも可）
DEFAULT_DATA_DIR = os.environ.get(
    "HBMSS_DATA_DIR", "/mnt/c/Users/takemurakimi/Project/Github/HBMSS/nk-hbmss/tools/peft-dataset/data"
)
TEMPLATE_PATH = Path(__file__).parent / "templates" / "qwen3_sft.jinja"

# eval は「学習に出ないシナリオ族（S07/S14/S19）」と「手書き held-out（H01〜H06）」の 2 系統。
# 前者が eval/indomain_loss、後者が eval/heldout_loss になる。後者は 6 件しかないので
# loss の値は荒い。判断には eval_hbmss.py の生成採点を併用する。
EVAL_GROUPS = {"indomain": lambda rec: not rec["scenario"].startswith("H"), "heldout": lambda rec: rec["scenario"].startswith("H")}


def load_sft_chat_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def read_records(data_dir: Path, split: str, files: list[str] = TRAIN_FILES) -> list[dict]:
    """split は "train"（data/*.jsonl）か "eval"（data/eval/*.jsonl）。"""
    base = data_dir if split == "train" else data_dir / "eval"
    records: list[dict] = []
    for name in files:
        with (base / f"{name}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rec["file"] = name
                    records.append(rec)
    return records


def tokenize_records(records: list[dict], tokenizer, max_length: int, label: str = "") -> Dataset:
    """messages + tools を SFT テンプレートでトークン化し、input_ids / assistant_masks を持つ Dataset にする。

    max_length を超える例は切り詰めずに落とす。切り詰めると assistant 部分（= 損失を掛ける側）が
    欠けた例を学習することになり、静かに壊れるため。
    """
    rows: list[dict] = []
    dropped = 0
    n_tokens = n_assistant = 0
    for rec in records:
        enc = tokenizer.apply_chat_template(
            rec["messages"],
            tools=rec.get("tools") or None,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        ids, mask = list(enc["input_ids"]), list(enc["assistant_masks"])
        if len(ids) > max_length:
            dropped += 1
            continue
        if 1 not in mask:
            raise ValueError(f"{rec['id']}: assistant マスクが空。テンプレートの generation マーカーを確認すること")
        n_tokens += len(ids)
        n_assistant += sum(mask)
        rows.append({"input_ids": ids, "assistant_masks": mask, "id": rec["id"], "file": rec["file"], "scenario": rec["scenario"]})
    print(
        f"[data{(' ' + label) if label else ''}] {len(rows)} 件, {n_tokens:,} tok, assistant {n_assistant:,} tok "
        f"({100 * n_assistant / max(n_tokens, 1):.1f}%), max_length 超過で除外 {dropped} 件",
        file=sys.stderr,
    )
    return Dataset.from_list(rows)


def build_datasets(data_dir: Path, tokenizer, max_length: int) -> tuple[Dataset, dict[str, Dataset]]:
    train = tokenize_records(read_records(data_dir, "train"), tokenizer, max_length, "train")
    eval_records = read_records(data_dir, "eval")
    evals = {}
    for name, pred in EVAL_GROUPS.items():
        subset = [r for r in eval_records if pred(r)]
        if subset:
            evals[name] = tokenize_records(subset, tokenizer, max_length, f"eval/{name}")
    return train, evals
