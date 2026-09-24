"""評価結果を横並びで表にする。

outputs/eval/*/summary.json（タスク）と outputs/eval-general/*/summary.json（汎用能力）を
読んで比較表を出す。run をまたいで見るのが目的なので、生成もモデル読み込みもしない。

    python report_eval.py                 # 全 run
    python report_eval.py checkpoint      # 名前に checkpoint を含む run だけ
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob

TASK_COLS = [
    ("json_valid", "json"),
    ("primary_severity_match", "severity"),
    ("primary_values_match", "values"),
    ("primary_subject_match", "subject"),
    ("primary_keys_match", "keys"),
    ("strict_match_no_ruleid", "strict*"),
    ("strict_match", "strict"),
    ("tag_ok", "tag"),
    ("truncated", "trunc"),
]


def _f(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _all(summary: dict, kind: str) -> dict:
    return summary.get("groups", {}).get(kind, {}).get("all", {}).get("all", {})


def _load(pattern: str, needle: str) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(glob(pattern)):
        name = os.path.basename(os.path.dirname(p))
        if needle and needle not in name:
            continue
        with open(p, encoding="utf-8") as fh:
            out.append((name, json.load(fh)))
    return out


def short(name: str) -> str:
    """run 名は長いので、判別に効く末尾（checkpoint-NNN や zeroshot）を残す。"""
    for marker in ("checkpoint-", "zeroshot"):
        i = name.find(marker)
        if i != -1:
            return name[i:]
    return name[-24:]


def main(argv: list[str]) -> None:
    needle = argv[0] if argv else ""

    task = _load("outputs/eval/*/summary.json", needle)
    if task:
        print("## タスク評価（strict* は ruleId を見ない strict）\n")
        print("| run | n | " + " | ".join(c[1] for c in TASK_COLS) + " | tool_args | L1 f1 | 禁止語 |")
        print("|" + "---|" * (len(TASK_COLS) + 4))
        for name, s in task:
            st, tt, l1 = _all(s, "structured"), _all(s, "tool_turn"), _all(s, "l1")
            cells = " | ".join(_f(st.get(k)) for k, _ in TASK_COLS)
            print(
                f"| {short(name)} | {st.get('n', '-')} | {cells} | {_f(tt.get('tool_args_match'))} | "
                f"{_f(l1.get('char_f1'))} | {st.get('forbidden_items', '-')} |"
            )

        print("\n### severity 混同行列（gold → pred）\n")
        print("| run | OK→OK | WARN→WARN | ERR→ERR | NA→NA | 出力なし |")
        print("|" + "---|" * 6)
        for name, s in task:
            c = s.get("severity_confusion", {})
            na = sum(v for k, v in c.items() if k.endswith("->n/a"))
            print(
                f"| {short(name)} | {c.get('OK->OK', 0)} | {c.get('WARN->WARN', 0)} | "
                f"{c.get('ERR->ERR', 0)} | {c.get('NA->NA', 0)} | {na} |"
            )

        print("\n### task_type 別\n")
        print("| run | task_type | n | json | severity | strict* |")
        print("|" + "---|" * 6)
        for name, s in task:
            for k, v in s.get("groups", {}).get("structured", {}).get("task_type", {}).items():
                print(
                    f"| {short(name)} | {k} | {v['n']} | {_f(v.get('json_valid'))} | "
                    f"{_f(v.get('primary_severity_match'))} | {_f(v.get('strict_match_no_ruleid'))} |"
                )

        print("\n### シナリオ別 json_valid / severity\n")
        print("| run | " + " | ".join(sorted({k for _, s in task for k in s.get("groups", {}).get("structured", {}).get("scenario", {})})) + " |")
        scen = sorted({k for _, s in task for k in s.get("groups", {}).get("structured", {}).get("scenario", {})})
        print("|" + "---|" * (len(scen) + 1))
        for name, s in task:
            g = s.get("groups", {}).get("structured", {}).get("scenario", {})
            print(f"| {short(name)} | " + " | ".join(f"{_f(g.get(k, {}).get('json_valid'))}/{_f(g.get(k, {}).get('primary_severity_match'))}" for k in scen) + " |")

    gen = _load("outputs/eval-general/*/summary.json", needle)
    if gen:
        print("\n## 汎用能力（劣化していないか）\n")
        print("| run | IFEval 命令 strict | IFEval prompt strict | JA 命令 strict | format bleed |")
        print("|" + "---|" * 5)
        for name, s in gen:
            ife = s.get("suites", {}).get("ifeval", {})
            ja = s.get("suites", {}).get("ja", {})
            print(
                f"| {short(name)} | {_f(ife.get('instruction_strict'))} | {_f(ife.get('prompt_strict'))} | "
                f"{_f(ja.get('instruction_strict'))} | {_f(ife.get('format_bleed_rate'))} |"
            )

    if not task and not gen:
        print("summary.json が見つかりません。outputs/eval/ か outputs/eval-general/ を確認してください", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
