"""汎用能力の回帰テスト（評価の第 3 軸）。

eval_hbmss.py は「HBMSS のタスクがどれだけ解けるか」しか測らない。SFT では
タスク性能が上がり続ける一方で指示追従が壊れることが知られているため、
学習データと無関係な能力が保たれているかを別に測る。

測るもの:

1. **IFEval**（541 問 / 25 命令型）— 「全部小文字で」「箇条書き 3 点で」のように
   機械検証できる命令の追従率。prompt 単位と命令単位、strict と loose を出す
2. **日本語プローブ**（15 問）— IFEval は英語のみ。学習データが日本語なので、
   日本語側の指示追従を別に見る。公式ベンチではないので素のモデルとの差分だけを見る
3. **format bleed** — 無関係な質問に対して findings JSON や【事実】タグを出していないか。
   2720 件の硬い JSON で SFT した後に最も起きやすい壊れ方で、IFEval だけでは名前が付かない

使い方（eval_hbmss.py と同じバックエンド・同じ生成条件）:
    python eval_general.py --backend hf --model-name Qwen/Qwen3-14B
    python eval_general.py --backend hf --model-name Qwen/Qwen3-14B --adapter outputs/<run>
    python eval_general.py --score-only outputs/eval/<run>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_lab.eval_hbmss import _adapter_label, generate_hf, generate_openai, generate_vllm, parse_output
from llm_lab.ifeval_checks import JA_CHECKS, JA_PROBES, UNSUPPORTED, check_instruction, format_bleed

IFEVAL_DATASET = "google/IFEval"


@dataclass
class GeneralItem:
    id: str
    suite: str  # "ifeval" | "ja"
    prompt: str
    instructions: list[dict]  # [{"id": str, "kwargs": dict}]


def load_items(suites: list[str], limit: int | None, seed: int) -> list[GeneralItem]:
    items: list[GeneralItem] = []
    if "ifeval" in suites:
        from datasets import load_dataset

        ds = load_dataset(IFEVAL_DATASET, split="train")
        rows = list(ds)
        if limit is not None and limit < len(rows):
            # 命令型が偏らないよう seed 固定でシャッフルしてから切る。
            # 同じ seed / limit なら素のモデルと学習後で同一の部分集合になる
            random.Random(seed).shuffle(rows)
            rows = rows[:limit]
        for r in rows:
            items.append(
                GeneralItem(
                    id=f"ifeval-{r['key']}",
                    suite="ifeval",
                    prompt=r["prompt"],
                    # kwargs は未使用キーが None で埋まっているので落としておく
                    instructions=[
                        {"id": i, "kwargs": {k: v for k, v in kw.items() if v is not None}}
                        for i, kw in zip(r["instruction_id_list"], r["kwargs"])
                    ],
                )
            )
    if "ja" in suites:
        for n, (prompt, instrs) in enumerate(JA_PROBES):
            items.append(
                GeneralItem(id=f"ja-{n:03d}", suite="ja", prompt=prompt, instructions=[{"id": i, "kwargs": kw} for i, kw in instrs])
            )
    return items


def score_item(item: GeneralItem, response: str) -> dict:
    per_instruction = []
    for ins in item.instructions:
        if item.suite == "ja":
            fn = JA_CHECKS[ins["id"]]
            try:
                r = fn(response, ins["kwargs"])
            except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                r = False
            # None は「判定できなかった」。bool() で潰すと不正解として数えられてしまう
            strict = None if r is None else bool(r)
            loose = strict
        else:
            strict = check_instruction(ins["id"], response, ins["kwargs"], loose=False)
            loose = check_instruction(ins["id"], response, ins["kwargs"], loose=True)
        per_instruction.append({"id": ins["id"], "strict": strict, "loose": loose})

    judged = [p for p in per_instruction if p["strict"] is not None]
    return {
        "per_instruction": per_instruction,
        "n_instructions": len(per_instruction),
        "n_judged": len(judged),
        # prompt 単位は「その問の命令をすべて満たしたか」。判定できない命令があれば None
        "prompt_strict": all(p["strict"] for p in judged) if len(judged) == len(per_instruction) else None,
        "prompt_loose": all(p["loose"] for p in judged) if len(judged) == len(per_instruction) else None,
        "forbidden_bleed": format_bleed(response),
        "response_chars": len(response),
    }


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_items": len(rows), "suites": {}}
    for suite in ("ifeval", "ja"):
        srows = [r for r in rows if r["item"]["suite"] == suite]
        if not srows:
            continue
        ins_strict = [p["strict"] for r in srows for p in r["score"]["per_instruction"] if p["strict"] is not None]
        ins_loose = [p["loose"] for r in srows for p in r["score"]["per_instruction"] if p["loose"] is not None]
        pr_strict = [r["score"]["prompt_strict"] for r in srows if r["score"]["prompt_strict"] is not None]
        pr_loose = [r["score"]["prompt_loose"] for r in srows if r["score"]["prompt_loose"] is not None]
        by_type: dict[str, list[bool]] = defaultdict(list)
        for r in srows:
            for p in r["score"]["per_instruction"]:
                if p["strict"] is not None:
                    by_type[p["id"]].append(p["strict"])
        out["suites"][suite] = {
            "n_prompts": len(srows),
            "n_instructions": sum(r["score"]["n_instructions"] for r in srows),
            "instruction_strict": sum(ins_strict) / len(ins_strict) if ins_strict else None,
            "instruction_loose": sum(ins_loose) / len(ins_loose) if ins_loose else None,
            "prompt_strict": sum(pr_strict) / len(pr_strict) if pr_strict else None,
            "prompt_loose": sum(pr_loose) / len(pr_loose) if pr_loose else None,
            "format_bleed_rate": sum(1 for r in srows if r["score"]["forbidden_bleed"]) / len(srows),
            "mean_response_chars": sum(r["score"]["response_chars"] for r in srows) / len(srows),
            "by_instruction_type": {k: {"n": len(v), "strict": sum(v) / len(v)} for k, v in sorted(by_type.items())},
        }
    out["unverified_instruction_types"] = sorted(UNSUPPORTED)
    bleed = [(r["item"]["id"], r["score"]["forbidden_bleed"]) for r in rows if r["score"]["forbidden_bleed"]]
    out["format_bleed_examples"] = bleed[:20]
    return out


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_report(summary: dict) -> None:
    print("\n## 汎用能力（学習データと無関係な指示追従）\n")
    print("| suite | prompts | 命令数 | 命令単位 strict | 命令単位 loose | prompt strict | prompt loose | format bleed | 平均文字数 |")
    print("|" + "---|" * 9)
    for suite, s in summary["suites"].items():
        print(
            f"| {suite} | {s['n_prompts']} | {s['n_instructions']} | {_fmt(s['instruction_strict'])} | "
            f"{_fmt(s['instruction_loose'])} | {_fmt(s['prompt_strict'])} | {_fmt(s['prompt_loose'])} | "
            f"{_fmt(s['format_bleed_rate'])} | {s['mean_response_chars']:.0f} |"
        )
    for suite, s in summary["suites"].items():
        print(f"\n### {suite} 命令型別 strict\n")
        print("| instruction type | n | strict |")
        print("|---|---|---|")
        for k, v in s["by_instruction_type"].items():
            print(f"| {k} | {v['n']} | {v['strict']:.3f} |")
    if summary["unverified_instruction_types"]:
        print("\n判定できなかった命令型:", ", ".join(summary["unverified_instruction_types"]))
    if summary["format_bleed_examples"]:
        print("\nformat bleed の例:")
        for pid, markers in summary["format_bleed_examples"][:5]:
            print(f"  {pid}: {markers}")


def _warn_if_no_langdetect() -> None:
    """langdetect が無いと言語判定の命令が丸ごと落ちる。黙って減るので目立たせる。

    生成は無駄にならない（predictions.jsonl を --score-only で採点し直せる）。
    """
    try:
        import langdetect  # noqa: F401
    except ImportError:
        print(
            "警告: langdetect が無いため言語判定の命令（IFEval の language:response_language と "
            "日本語プローブの ja:language）を判定できません。`uv pip install langdetect` のあと "
            "--score-only で採点し直してください",
            file=sys.stderr,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汎用能力の回帰テスト（IFEval + 日本語プローブ）", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("suite")
    g.add_argument("--suites", nargs="*", default=["ifeval", "ja"], choices=["ifeval", "ja"])
    g.add_argument("--limit", type=int, default=None, help="IFEval から使う問題数（seed 固定で抽出）。既定は全 541 問")
    g.add_argument("--seed", type=int, default=0, help="--limit で抽出するときの seed。比較する run 間で揃えること")

    g = p.add_argument_group("model")
    g.add_argument("--backend", default="hf", choices=["vllm", "hf", "openai"])
    g.add_argument("--model-name", default="Qwen/Qwen3-14B")
    g.add_argument("--adapter", default=None)
    g.add_argument("--load-in-4bit", action="store_true")
    g.add_argument(
        "--no-merge-adapter",
        dest="merge_adapter",
        action="store_false",
        help="アダプタをベース重みにマージせず LoRA 層のまま推論する（4bit では既定でマージしない）",
    )
    g.add_argument("--enable-thinking", action="store_true")
    g.add_argument("--max-new-tokens", type=int, default=1024, help="300 語以上を求める問があるので 1024 は必要")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--batch-size", type=int, default=16)

    g = p.add_argument_group("vllm backend")
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--max-lora-rank", type=int, default=64)
    g.add_argument("--enforce-eager", action="store_true")

    g = p.add_argument_group("openai backend")
    g.add_argument("--base-url", default=None)
    g.add_argument("--api-key", default=None)
    g.add_argument("--served-model", default=None)
    g.add_argument("--concurrency", type=int, default=8)

    g = p.add_argument_group("output")
    g.add_argument("--run-name", default=None)
    g.add_argument("--output-dir", default=None, help="既定: ./outputs/eval-general/<run-name>")
    g.add_argument("--score-only", default=None, metavar="PREDICTIONS_JSONL")
    g.add_argument("--no-resume", dest="resume", action="store_false")

    args = p.parse_args(argv)
    if args.run_name is None:
        base = args.model_name.rstrip("/").split("/")[-1]
        base += ("-" + _adapter_label(args.adapter)) if args.adapter else "-zeroshot"
        args.run_name = base
    if args.output_dir is None:
        args.output_dir = f"./outputs/eval-general/{args.run_name}"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        rows = [json.loads(l) for l in Path(args.score_only).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            item = GeneralItem(**r["item"])
            r["score"] = score_item(item, r["response"])
        summary = summarize(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_report(summary)
        return

    _warn_if_no_langdetect()
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").is_file():
        raise SystemExit(f"アダプタが見つかりません: {args.adapter}（adapter_config.json が無い）")

    items = load_items(args.suites, args.limit, args.seed)
    print(f"{len(items)} 問を評価: backend={args.backend} model={args.model_name} adapter={args.adapter}", file=sys.stderr)

    partial_path = out_dir / "predictions.partial.jsonl"
    done: dict[str, dict] = {}
    if partial_path.exists():
        if args.resume:
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
            print(f"再開: {len(done)} 問は生成済み", file=sys.stderr)
        else:
            partial_path.unlink()
    todo = [i for i, it in enumerate(items) if it.id not in done]
    fh = partial_path.open("a", encoding="utf-8")

    def record(i: int, text: str, finish: str | None, pre_parsed=None) -> None:
        it = items[i]
        row = {"id": it.id, "raw_output": text, "finish_reason": finish}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        done[it.id] = row

    t0 = time.time()
    if todo:
        if args.backend in ("vllm", "hf"):
            from transformers import AutoTokenizer

            tok_src = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer_config.json").exists() else args.model_name
            tokenizer = AutoTokenizer.from_pretrained(tok_src)
            print(f"tokenizer / chat template: {tok_src}", file=sys.stderr)
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": items[i].prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
                for i in todo
            ]
            if args.backend == "vllm":
                for j, (text, finish) in enumerate(generate_vllm(prompts, args)):
                    record(todo[j], text, finish)
            else:
                generate_hf(prompts, args, tokenizer, on_result=lambda j, text, finish: record(todo[j], text, finish))
        else:
            # openai バックエンドは EvalItem 互換の prefix を期待するので、その形に合わせる
            class _Shim:
                def __init__(self, it: GeneralItem):
                    self.id = it.id
                    self.prefix = [{"role": "user", "content": it.prompt}]
                    self.tools = None

            generate_openai(
                [_Shim(items[i]) for i in todo], args, on_result=lambda j, text, finish, tcs: record(todo[j], text, finish)
            )
    elapsed = time.time() - t0
    fh.close()

    rows = []
    for it in items:
        r = done[it.id]
        # think ブロックと制御トークンを落とす。判定対象は本文のみ
        parsed = parse_output(r["raw_output"], None, r.get("finish_reason"))
        rows.append(
            {
                "item": asdict(it),
                "raw_output": r["raw_output"],
                "finish_reason": r.get("finish_reason"),
                "response": parsed.content,
                "score": score_item(it, parsed.content),
            }
        )

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)

    summary = summarize(rows)
    summary["config"] = {**vars(args), "elapsed_sec": elapsed}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary)
    print(f"\n生成 {elapsed/60:.1f} 分。出力: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
