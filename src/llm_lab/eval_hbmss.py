"""HBMSS peft-dataset の生成評価ランナー（ステップ 0）。

学習前に「素のモデルが eval をどれだけ解けるか」を測り、学習後は同じスクリプトに
`--adapter` を足して差分を見る。eval/loss は代理指標に過ぎず、このタスクの正解は
findings JSON の severity / expected / actual / diff と tool_calls の一致で決まるため、
生成ベースの採点を主指標にする。

採点の考え方:
- L2/L3（classify / trace / diagnose / abstain）と軌跡の最終回答は findings JSON。
  gold の findings[0] を主判定（primary）とし、ruleId で突き合わせて severity と値を比べる。
  message / probableCauses は自由文なので一致は求めず、【事実】【仕様】【推測】タグの有無だけ見る
- 軌跡（next_action）は teacher forcing。各 assistant ターンを gold の直前までの履歴から
  生成させ、途中ターンは tool_calls の name / arguments、最終ターンは findings で採点する。
  ツールの実シミュレーターを持たずに「次の一手」を独立に評価できる
- L1（define / narrate）は自由文。タグの有無・文字 F1・禁止語彙だけを見る（正解一致は測れない）

使い方:
    .venv/bin/python eval_hbmss.py --backend vllm --model-name Qwen/Qwen3-4B --load-in-4bit --limit 3
    .venv/bin/python eval_hbmss.py --backend hf   --model-name Qwen/Qwen3-4B --load-in-4bit
    .venv/bin/python eval_hbmss.py --backend openai --base-url http://<host>:8000/v1 --model-name Qwen/Qwen3-14B
    .venv/bin/python eval_hbmss.py --score-only outputs/eval/<run>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_lab.hbmss_data import DEFAULT_DATA_DIR

FILES = ["layer1_domain", "layer2_system_rules", "layer3_consistency", "trajectories_next_action"]
STRUCTURED_TASKS = {"classify", "trace", "diagnose", "abstain", "next_action"}
SEVERITIES = ["OK", "WARN", "ERR", "NA"]


TAG_RE = re.compile(r"【(事実|仕様|推測)】")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# 生成末尾の制御トークン。skip_special_tokens に頼らず自前で落とす（<think> や <tool_call> は
# Qwen3 では special 扱いではないため、skip_special_tokens の挙動に依存させたくない）
CONTROL_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]


# --------------------------------------------------------------------------- data


@dataclass
class EvalItem:
    """1 回の生成に対応する評価単位。L1〜L3 は 1 レコード = 1 件、軌跡は assistant ターンごとに 1 件。"""

    id: str
    file: str
    layer: str
    task_type: str
    scenario: str
    outcome: str
    hard_negative: bool
    hard_negative_direction: str | None
    turn: int  # 何番目の assistant ターンか（0 始まり）
    is_final: bool  # レコードの最終 assistant ターンか
    prefix: list[dict]  # 生成に与える履歴（system〜直前の tool まで）
    tools: list[dict] | None
    gold_content: str
    gold_tool_calls: list[dict]  # [{"name", "arguments"(dict)}]
    rule: str | None = None  # gold findings[0] の ruleId。ruleId 別の集計に使う


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """(ファイル, シナリオ群, outcome) の比率を保ったまま n 件に絞る。

    層ごとに seed 固定でシャッフルし、比率に応じた件数を取る。端数は層を大きい順に配る。
    小さい層（H 群など）が消えないよう、各層から最低 1 件は残す。
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["file"], _scenario_group(r.get("scenario", "")), r.get("outcome"))].append(r)
    total = len(records)
    rng = random.Random(seed)
    quota: dict[tuple, int] = {}
    for k, v in buckets.items():
        rng.shuffle(v)
        quota[k] = max(1, round(n * len(v) / total))
    # 丸めで n とずれるので、大きい層から増減して合わせる
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    while sum(quota.values()) > n:
        for k in order:
            if sum(quota.values()) <= n:
                break
            if quota[k] > 1:
                quota[k] -= 1
    while sum(quota.values()) < n:
        for k in order:
            if sum(quota.values()) >= n:
                break
            if quota[k] < len(buckets[k]):
                quota[k] += 1
    out = [r for k in order for r in buckets[k][: quota[k]]]
    out.sort(key=lambda r: r["id"])
    return out


def load_items(data_dir: Path, files: list[str], limit: int | None, sample: int | None = None, sample_seed: int = 0) -> list[EvalItem]:
    items: list[EvalItem] = []
    records: list[dict] = []
    for name in files:
        path = data_dir / "eval" / f"{name}.jsonl"
        n_records = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if limit is not None and n_records >= limit:
                    break
                n_records += 1
                rec = json.loads(line)
                rec["file"] = name
                records.append(rec)
    if sample is not None and sample < len(records):
        records = stratified_sample(records, sample, sample_seed)
        print(f"層別抽出: {len(records)} レコード（seed {sample_seed}）", file=sys.stderr)

    for rec in records:
        name = rec["file"]
        msgs = rec["messages"]
        assistant_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for turn, i in enumerate(assistant_idx):
            gold = msgs[i]
            items.append(
                EvalItem(
                    id=rec["id"],
                    file=name,
                    layer=rec["layer"],
                    task_type=rec["task_type"],
                    scenario=rec["scenario"],
                    outcome=rec["outcome"],
                    hard_negative=bool(rec.get("hard_negative")),
                    hard_negative_direction=rec.get("hard_negative_direction") or None,
                    turn=turn,
                    is_final=(i == assistant_idx[-1]),
                    prefix=msgs[:i],
                    tools=rec.get("tools") or None,
                    gold_content=gold.get("content") or "",
                    gold_tool_calls=[
                        {
                            "name": tc["function"]["name"],
                            "arguments": _loads_maybe(tc["function"]["arguments"]),
                        }
                        for tc in (gold.get("tool_calls") or [])
                    ],
                    rule=(rec.get("rule_ids") or [None])[0],
                )
            )
    return items


def _loads_maybe(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def load_forbidden_vocab(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    words = obj["words"] if isinstance(obj, dict) else obj
    return sorted(set(words), key=len, reverse=True)


# --------------------------------------------------------------------------- parsing


@dataclass
class Parsed:
    content: str
    tool_calls: list[dict]
    thinking: str | None
    finish_reason: str | None


def parse_output(text: str, pre_parsed_tool_calls: list[dict] | None = None, finish_reason: str | None = None) -> Parsed:
    """生成テキストを content / tool_calls / thinking に分解する。

    OpenAI 互換サーバーがサーバー側で tool_calls を構造化して返した場合は pre_parsed_tool_calls
    に入ってくるので、本文中の <tool_call> ブロックと合わせて 1 本のリストにする。
    """
    for t in CONTROL_TOKENS:
        # 最初の制御トークン以降は切り落とす（HF のバッチ生成では pad が続く）
        idx = text.find(t)
        if idx != -1:
            text = text[:idx]

    thinking = None
    m = THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip() or None
        text = text[: m.start()] + text[m.end() :]
    elif "<think>" in text:
        # 閉じずに終わった思考。全部 thinking 扱いにして本文は空
        thinking = text.split("<think>", 1)[1].strip() or None
        text = text.split("<think>", 1)[0]

    tool_calls: list[dict] = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            tool_calls.append({"name": obj.get("name"), "arguments": _loads_maybe(obj.get("arguments"))})
        except json.JSONDecodeError:
            tool_calls.append({"name": None, "arguments": None, "raw": m.group(1)})
    for tc in pre_parsed_tool_calls or []:
        tool_calls.append({"name": tc.get("name"), "arguments": _loads_maybe(tc.get("arguments"))})

    content = TOOL_CALL_RE.sub("", text).strip()
    return Parsed(content=content, tool_calls=tool_calls, thinking=thinking, finish_reason=finish_reason)


def parse_json_obj(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- scoring


def _norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _eq(a, b) -> bool:
    return json.dumps(_norm(a), sort_keys=True, ensure_ascii=False) == json.dumps(_norm(b), sort_keys=True, ensure_ascii=False)


def _string_values(obj) -> list[str]:
    """JSON の自然文値だけを集める。禁止語彙の検査は仕様どおりキーではなく値を対象にする。"""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_string_values(x))
    elif isinstance(obj, dict):
        for x in obj.values():
            out.extend(_string_values(x))
    return out


IDENT_RE = re.compile(r"[A-Za-z_]\w*")
URL_OR_API_RE = re.compile(r"https?://|/api/")


def _ident_tokens(texts: list[str]) -> set[str]:
    return {t.lower() for text in texts for t in IDENT_RE.findall(text)}


def _context_tokens(item: "EvalItem") -> set[str]:
    return _ident_tokens([m.get("content") or "" for m in item.prefix if isinstance(m.get("content"), str)])


# validate.mjs は `patient` だけを除外している。`id` は DB の列名として語彙に入っているが、
# モデルが「患者ID」と書いただけで当たるため、列名の幻覚を測る目的から外れる。ここでは除外する
VOCAB_SKIP = {"patient", "id"}


def count_forbidden(texts: list[str], vocab: list[str], context_tokens: set[str]) -> list[str]:
    """generate/validate.mjs の validateVocabulary と同じ規則。

    出力の識別子トークンに DB 物理名が現れ、かつそれがプロンプト側（system / user / tool）に
    出てこない場合だけ「未提供の陳腐化語彙」として数える。
    """
    answer_tokens = _ident_tokens(texts)
    hits = [w for w in vocab if w not in VOCAB_SKIP and w in answer_tokens and w not in context_tokens]
    if any(URL_OR_API_RE.search(t) for t in texts):
        hits.append("<url_or_api_path>")
    return hits


def char_f1(pred: str, gold: str) -> float:
    p = Counter(pred.replace(" ", "").replace("\n", ""))
    g = Counter(gold.replace(" ", "").replace("\n", ""))
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _match_findings(pf: list[dict], gf: list[dict]) -> list[tuple[dict | None, dict]]:
    """ruleId を使わずに予測と gold の findings を対応付ける。

    subject（対象患者・日付）が一致するものを優先し、無ければ残りを順に割り当てる。
    ruleId は入力から復元できない値なので、判定値の正しさを測るときは対応付けから外す。
    """
    used: set[int] = set()
    pairs: list[tuple[dict | None, dict]] = []
    for g in gf:
        pick = None
        for i, p in enumerate(pf):
            if i not in used and _eq(p.get("subject"), g.get("subject")):
                pick = i
                break
        if pick is None:
            pick = next((i for i in range(len(pf)) if i not in used), None)
        if pick is None:
            pairs.append((None, g))
        else:
            used.add(pick)
            pairs.append((pf[pick], g))
    return pairs


def _tags_ok(findings: list[dict]) -> bool:
    for f in findings:
        if not isinstance(f, dict):
            return False
        msg = f.get("message")
        if not isinstance(msg, str) or not TAG_RE.match(msg):
            return False
        for pc in f.get("probableCauses") or []:
            if not isinstance(pc, str) or not pc.startswith("【推測】"):
                return False
    return True


def score_item(item: EvalItem, parsed: Parsed, vocab: list[str]) -> dict:
    ctx = _context_tokens(item)
    r: dict = {
        "think_leak": parsed.thinking is not None,
        "truncated": parsed.finish_reason == "length",
        "pred_chars": len(parsed.content),
    }

    if item.gold_tool_calls:
        # 軌跡の途中ターン: 次に呼ぶべきツールとその引数
        gold_names = [g["name"] for g in item.gold_tool_calls]
        pred_names = [p.get("name") for p in parsed.tool_calls]
        r["tool_name_match"] = pred_names == gold_names
        r["tool_args_match"] = r["tool_name_match"] and all(
            _eq(p.get("arguments"), g["arguments"]) for p, g in zip(parsed.tool_calls, item.gold_tool_calls)
        )
        r["tool_called"] = bool(parsed.tool_calls)
        r["forbidden_hits"] = count_forbidden([parsed.content], vocab, ctx)
        return r

    if item.task_type in STRUCTURED_TASKS:
        # 最終回答は findings JSON。ツール呼び出しが混ざっていたら減点対象として記録する
        r["stray_tool_call"] = bool(parsed.tool_calls)
        obj = parse_json_obj(parsed.content)
        gold = json.loads(item.gold_content)
        gf: list[dict] = gold["findings"]
        primary = gf[0]
        r["gold_severity"] = primary["severity"]
        r["json_valid"] = isinstance(obj, dict) and isinstance(obj.get("findings"), list) and all(
            isinstance(f, dict) for f in obj["findings"]
        )
        if not r["json_valid"]:
            r.update(
                pred_severity=None,
                ruleid_set_match=False,
                primary_severity_match=False,
                primary_values_match=False,
                primary_subject_match=False,
                primary_keys_match=False,
                strict_match=False,
                strict_match_no_ruleid=False,
                tag_ok=False,
                probable_causes_consistent=False,
                intermediate_present=False,
                intermediate_operation_match=False,
                intermediate_value_match=False,
                intermediate_operands_match=False,
                intermediate_consistent=None,
                copied_observation=None,
            )
            r["forbidden_hits"] = count_forbidden([parsed.content], vocab, ctx)
            return r

        pf: list[dict] = obj["findings"]
        by_rule: dict = {}
        for f in pf:
            by_rule.setdefault(f.get("ruleId"), f)
        pp = by_rule.get(primary["ruleId"]) or (pf[0] if pf else None)

        r["ruleid_set_match"] = {f.get("ruleId") for f in pf} == {f["ruleId"] for f in gf}
        r["pred_severity"] = pp.get("severity") if pp else None
        r["primary_severity_match"] = bool(pp) and pp.get("severity") == primary["severity"]
        r["primary_values_match"] = bool(pp) and all(_eq(pp.get(k), primary.get(k)) for k in ("expected", "actual", "diff"))
        r["primary_subject_match"] = bool(pp) and _eq(pp.get("subject"), primary.get("subject"))
        r["primary_keys_match"] = bool(pp) and set(pp.get("sourceOrderKeys") or []) == set(primary.get("sourceOrderKeys") or [])

        # strict: gold の全 findings について、同じ ruleId の予測が severity / subject / 値まで一致し、余計な finding もない
        strict = r["ruleid_set_match"] and len(pf) == len(gf)
        if strict:
            for g in gf:
                p = by_rule.get(g["ruleId"])
                strict = strict and p is not None and p.get("severity") == g["severity"] and _eq(p.get("subject"), g.get("subject"))
                strict = strict and all(_eq(p.get(k), g.get(k)) for k in ("expected", "actual", "diff"))
        r["strict_match"] = bool(strict)

        # ruleId を除いた strict。ruleId は入力のどこにも現れず（train/eval とも 0 件）、
        # eval のシナリオ族が使う T-05/T-06/C-06/W-05/C-05 は学習に 1 件も無い。
        # つまり strict_match は 396 件中 332 件で到達不能なので、判定値そのものを見る指標を別に持つ
        strict_nr = len(pf) == len(gf)
        if strict_nr:
            for p, g in _match_findings(pf, gf):
                strict_nr = strict_nr and p is not None and p.get("severity") == g["severity"] and _eq(p.get("subject"), g.get("subject"))
                strict_nr = strict_nr and all(_eq(p.get(k), g.get(k)) for k in ("expected", "actual", "diff"))
        r["strict_match_no_ruleid"] = bool(strict_nr)

        # 2026-09-25 改訂で回答の先頭に intermediate（operation / operands / value）が付いた。
        # 観測値のコピーで済ませられないようにするための教師信号なので、効いているかを直接測る
        gi, pi = gold.get("intermediate"), obj.get("intermediate")
        if isinstance(gi, dict):
            r["intermediate_present"] = isinstance(pi, dict)
            r["intermediate_operation_match"] = bool(pi) and pi.get("operation") == gi.get("operation")
            r["intermediate_value_match"] = bool(pi) and _eq(pi.get("value"), gi.get("value"))
            r["intermediate_operands_match"] = bool(pi) and _eq(pi.get("operands"), gi.get("operands"))
            # gold では intermediate.value と findings[0].expected が常に一致する（2,971/2,971）。
            # 予測が自分の再計算結果を判定に使っているかを見る指標。もっともらしい intermediate を
            # 出しておきながら expected には観測値を書く、という振る舞いをここで検出する
            if pi and pp and pi.get("value") is not None:
                r["intermediate_consistent"] = _eq(pi.get("value"), pp.get("expected"))

        # 旧モデルの主要な失敗は「再計算せず観測値を expected に書き写す」ことだった。
        # gold が expected != actual の例に限って、予測が両者を同じにしていないかを見る
        if pp and not _eq(primary.get("expected"), primary.get("actual")):
            r["copied_observation"] = _eq(pp.get("expected"), pp.get("actual"))

        r["tag_ok"] = _tags_ok(pf)
        # ERR なら原因候補が要る、OK なら要らない、という契約の整合
        if pp:
            pcs = pp.get("probableCauses") or []
            sev = pp.get("severity")
            r["probable_causes_consistent"] = (sev != "ERR" or len(pcs) > 0) and (sev != "OK" or len(pcs) == 0)
        else:
            r["probable_causes_consistent"] = False
        r["forbidden_hits"] = count_forbidden(_string_values(obj), vocab, ctx)
        return r

    # L1: 自由文
    r["tag_ok"] = bool(TAG_RE.search(parsed.content))
    r["char_f1"] = char_f1(parsed.content, item.gold_content)
    r["is_json"] = parse_json_obj(parsed.content) is not None
    r["stray_tool_call"] = bool(parsed.tool_calls)
    r["forbidden_hits"] = count_forbidden([parsed.content], vocab, ctx)
    return r


# --------------------------------------------------------------------------- backends


def render_prompts(items: list[EvalItem], tokenizer, enable_thinking: bool) -> list[str]:
    """学習時と同じ chat template でプロンプトを文字列化する。

    バックエンド間（hf / vllm）で描画を揃えるため、テンプレート適用はここで一元化し、
    各バックエンドには生の文字列だけを渡す。
    """
    prompts = []
    for it in items:
        prompts.append(
            tokenizer.apply_chat_template(
                it.prefix,
                tools=it.tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return prompts


def generate_hf(prompts: list[str], args, tokenizer, on_result=None) -> list[tuple[str, str]]:
    """on_result(i, text, finish) をバッチごとに呼ぶ。数時間かかる run が途中で死んでも結果を失わないため。"""
    import torch
    from transformers import AutoModelForCausalLM

    quant = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=quant,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        # 既定でベース重みにマージする。LoRA 層のままだと forward ごとに
        # result + lora_B(lora_A(x)) * scaling の一時テンソル（出力と同サイズ）が層ごとに積まれ、
        # 素のモデルで通るバッチでも OOM する。W + BA*scaling は等価なので結果は変わらない。
        # 4bit はマージで dequant が起きるため対象外
        if args.merge_adapter and not args.load_in_4bit:
            model = model.merge_and_unload()
            print("アダプタをベース重みにマージしました", file=sys.stderr)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    eos_ids.add(tokenizer.eos_token_id)

    # 長い順に並べて OOM は最初のバッチで踏む。padding も減る
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs: list[tuple[str, str] | None] = [None] * len(prompts)
    t0 = time.time()
    for b in range(0, len(order), args.batch_size):
        batch = order[b : b + args.batch_size]
        enc = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        new = gen[:, enc["input_ids"].shape[1] :]
        for i, seq in zip(batch, new):
            ids = seq.tolist()
            finish = "stop" if any(t in eos_ids for t in ids) else "length"
            outs[i] = (tokenizer.decode(ids, skip_special_tokens=False), finish)
            if on_result:
                on_result(i, *outs[i])
        done = b + len(batch)
        el = time.time() - t0
        print(f"[hf] {done}/{len(order)}  {el/60:.1f} min elapsed, ETA {el/done*(len(order)-done)/60:.1f} min", file=sys.stderr, flush=True)
    return outs  # type: ignore[return-value]


def generate_vllm(prompts: list[str], args) -> list[tuple[str, str]]:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model_name,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        enable_prefix_caching=True,
    )
    if args.load_in_4bit:
        kwargs["quantization"] = "bitsandbytes"
    if args.adapter:
        kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    llm = LLM(**kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False,
    )
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("adapter", 1, args.adapter)
    outputs = llm.generate(prompts, sp, lora_request=lora)
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outputs]


def generate_openai(items: list[EvalItem], args, on_result=None) -> list[tuple[str, str, list[dict]]]:
    """OpenAI 互換サーバー（vLLM serve / 各社 API）。テンプレートはサーバー側が描く。"""
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY")
    served = args.served_model or args.model_name

    def call(it: EvalItem):
        kwargs = dict(
            model=served,
            messages=it.prefix,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
        )
        if it.tools:
            kwargs["tools"] = it.tools
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"[openai] retry {attempt+1} for {it.id}: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        msg = resp.choices[0].message
        tcs = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (msg.tool_calls or [])]
        # reasoning_content で思考が返る実装もある。漏れとして扱えるように本文へ戻す
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            text = f"<think>{reasoning}</think>" + text
        return text, resp.choices[0].finish_reason, tcs

    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(call, it): i for i, it in enumerate(items)}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            results[i] = fut.result()
            if on_result:
                on_result(i, *results[i])
            if n % 20 == 0 or n == len(items):
                print(f"[openai] {n}/{len(items)}", file=sys.stderr, flush=True)
    return results


# --------------------------------------------------------------------------- report


METRIC_COLUMNS = {
    "structured": [
        "json_valid",
        "primary_severity_match",
        "primary_values_match",
        "primary_subject_match",
        "primary_keys_match",
        "intermediate_value_match",
        "intermediate_operands_match",
        "intermediate_consistent",
        "copied_observation",
        "strict_match_no_ruleid",
        "strict_match",
        "ruleid_set_match",
        "tag_ok",
        "probable_causes_consistent",
        "stray_tool_call",
        "truncated",
        "think_leak",
    ],
    "tool_turn": ["tool_called", "tool_name_match", "tool_args_match", "truncated", "think_leak"],
    "l1": ["tag_ok", "char_f1", "is_json", "stray_tool_call", "truncated", "think_leak"],
}


def _kind(row: dict) -> str:
    if row["item"]["gold_tool_calls"]:
        return "tool_turn"
    if row["item"]["task_type"] in STRUCTURED_TASKS:
        return "structured"
    return "l1"


def _rate(rows: list[dict], metric: str):
    vals = [r["score"][metric] for r in rows if r["score"].get(metric) is not None]
    if not vals:
        return None
    return sum(float(v) for v in vals) / len(vals)


def _scenario_group(s: str) -> str:
    """2026-09-25 改訂で scenario は 44 種類になった。個別に並べても読めないので S / K / H に畳む。

    S = 学習に無いシナリオ族（未知ルール中心）、K = 既知ルールの対照群、H = 手書き held-out。
    """
    return s[0] if s and s[0] in "SKH" else s


def summarize(rows: list[dict]) -> dict:
    summary: dict = {"n_items": len(rows), "groups": {}}
    for kind in ("structured", "tool_turn", "l1"):
        krows = [r for r in rows if _kind(r) == kind]
        if not krows:
            continue
        groupers = {
            "all": lambda r: "all",
            "file": lambda r: r["item"]["file"],
            "task_type": lambda r: r["item"]["task_type"],
            "scenario": lambda r: _scenario_group(r["item"]["scenario"]),
            "hard_negative": lambda r: (
                f"hn:{r['item']['hard_negative_direction'] or 'yes'}" if r["item"]["hard_negative"] else "hn:no"
            ),
            "gold_outcome": lambda r: r["item"]["outcome"],
        }
        if kind == "structured":
            # 改訂後は「どのルールで解けているか」が判断材料になる
            groupers["rule"] = lambda r: r["item"].get("rule") or "unknown"
        if kind == "tool_turn":
            groupers["turn"] = lambda r: f"turn{r['item']['turn']}"
        tables = {}
        for gname, fn in groupers.items():
            buckets: dict[str, list[dict]] = defaultdict(list)
            for r in krows:
                buckets[fn(r)].append(r)
            tables[gname] = {
                key: {"n": len(b), **{m: _rate(b, m) for m in METRIC_COLUMNS[kind]},
                      "forbidden_items": sum(1 for r in b if r["score"].get("forbidden_hits"))}
                for key, b in sorted(buckets.items())
            }
        summary["groups"][kind] = tables

    # severity の混同行列（構造化タスクの最終回答のみ）
    conf: Counter = Counter()
    for r in rows:
        if _kind(r) == "structured":
            conf[(r["score"]["gold_severity"], r["score"].get("pred_severity") or "n/a")] += 1
    summary["severity_confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(conf.items())}

    # 軌跡単位: 全ターン正解か
    traj: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r["item"]["file"] != "trajectories_next_action":
            continue
        ok = r["score"].get("tool_args_match") if r["item"]["gold_tool_calls"] else r["score"].get("strict_match")
        traj[r["item"]["id"]].append(bool(ok))
    if traj:
        summary["trajectory_all_turns_correct"] = {
            "n": len(traj),
            "rate": sum(all(v) for v in traj.values()) / len(traj),
        }
    return summary


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(summary: dict) -> None:
    for kind, title in (("structured", "構造化タスク（L2/L3 + 軌跡最終回答）"), ("tool_turn", "軌跡の途中ターン（tool_calls）"), ("l1", "L1 自由文")):
        tables = summary["groups"].get(kind)
        if not tables:
            continue
        cols = METRIC_COLUMNS[kind]
        print(f"\n## {title}\n")
        print("| group | n | " + " | ".join(cols) + " | forbidden |")
        print("|" + "---|" * (len(cols) + 3))
        for gname, table in tables.items():
            for key, vals in table.items():
                label = key if gname == "all" else f"{gname}={key}"
                print(f"| {label} | {vals['n']} | " + " | ".join(_fmt(vals[c]) for c in cols) + f" | {vals['forbidden_items']} |")
    if summary.get("severity_confusion"):
        print("\n## severity 混同行列（gold -> pred）\n")
        conf = summary["severity_confusion"]
        preds = SEVERITIES + ["n/a"]
        print("| gold \\ pred | " + " | ".join(preds) + " |")
        print("|" + "---|" * (len(preds) + 1))
        for g in SEVERITIES:
            print(f"| {g} | " + " | ".join(str(conf.get(f"{g}->{p}", 0)) for p in preds) + " |")
    if summary.get("trajectory_all_turns_correct"):
        t = summary["trajectory_all_turns_correct"]
        print(f"\n軌跡の全ターン正解率: {t['rate']:.3f} (n={t['n']})")


# --------------------------------------------------------------------------- main


def _adapter_label(adapter: str) -> str:
    """出力ディレクトリ名に使うアダプタの識別子。

    checkpoint-170 のような名前は run をまたいで重複するので、親ディレクトリ（run 名）を前置する。
    そうしないと r=16 と r=64 の checkpoint-170 が同じ出力先を指し、先の結果を上書きする。
    """
    path = Path(adapter.rstrip("/"))
    if path.name.startswith("checkpoint-") and path.parent.name:
        return f"{path.parent.name}-{path.name}"
    return path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HBMSS peft-dataset の生成評価", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="peft-dataset/data のパス（eval/ を含む）。環境変数 HBMSS_DATA_DIR でも指定可")
    g.add_argument("--files", nargs="*", default=FILES, choices=FILES)
    g.add_argument("--limit", type=int, default=None, help="ファイルごとの先頭 N レコードだけ使う（smoke test 用）")
    g.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "eval 全体から N レコードを層別抽出する。評価が全体コストの大半を占めるため、"
            "反復中はこれで縮める。層は (ファイル, シナリオ群 S/K/H, gold outcome) で、"
            "--sample-seed が同じなら run 間で同一の部分集合になる"
        ),
    )
    g.add_argument("--sample-seed", type=int, default=0, help="--sample の抽出 seed。比較する run 全体で揃えること")
    g.add_argument("--forbidden-vocab", default=None, help="既定: <data-dir>/../harness/forbidden_vocabulary.json")

    g = p.add_argument_group("model")
    g.add_argument("--backend", default="vllm", choices=["vllm", "hf", "openai"])
    g.add_argument("--model-name", default="Qwen/Qwen3-4B")
    g.add_argument("--adapter", default=None, help="LoRA アダプタのディレクトリ（学習後の評価用）")
    g.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes NF4 で読む（8 GB 機で 4B 以上を動かすとき）")
    g.add_argument(
        "--no-merge-adapter",
        dest="merge_adapter",
        action="store_false",
        help="アダプタをベース重みにマージせず LoRA 層のまま推論する（4bit では既定でマージしない）",
    )
    g.add_argument("--enable-thinking", action="store_true", help="Qwen3 の thinking モードで生成する（既定は空 <think> で非思考）")
    g.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help=(
            "2026-09-25 改訂の gold 回答は最大 1,425 tok。1024 だと 10.6% が打ち切られ、"
            "モデルの良否に関係なく 0 点になる。比較する run 全体で同じ値にすること"
        ),
    )
    g.add_argument("--temperature", type=float, default=0.0, help="0 で greedy")
    g.add_argument("--top-p", type=float, default=0.95)

    g = p.add_argument_group("hf backend")
    g.add_argument("--batch-size", type=int, default=4)

    g = p.add_argument_group("vllm backend")
    g.add_argument("--max-model-len", type=int, default=6144, help="prompt 最大 ~3.8K + 生成 1K を見込む")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--max-lora-rank", type=int, default=64)
    g.add_argument("--enforce-eager", action="store_true", help="CUDA graph を切る（VRAM 節約・起動短縮）")

    g = p.add_argument_group("openai backend")
    g.add_argument("--base-url", default=None)
    g.add_argument("--api-key", default=None, help="既定: 環境変数 OPENAI_API_KEY、無ければ EMPTY")
    g.add_argument("--served-model", default=None, help="サーバー側のモデル名（既定: --model-name）")
    g.add_argument("--concurrency", type=int, default=8)

    g = p.add_argument_group("output")
    g.add_argument("--run-name", default=None, help="既定: <model>[-<adapter>]-zeroshot")
    g.add_argument("--output-dir", default=None, help="既定: ./outputs/eval/<run-name>")
    g.add_argument("--score-only", default=None, metavar="PREDICTIONS_JSONL", help="生成せず既存の予測を再採点する")
    g.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="同じ出力先に途中結果（predictions.partial.jsonl）があっても使わず、最初から生成する",
    )

    args = p.parse_args(argv)
    if args.run_name is None:
        base = args.model_name.rstrip("/").split("/")[-1]
        base += ("-" + _adapter_label(args.adapter)) if args.adapter else "-zeroshot"
        if args.enable_thinking:
            base += "-think"
        args.run_name = base
    if args.output_dir is None:
        args.output_dir = f"./outputs/eval/{args.run_name}"
    if args.forbidden_vocab is None:
        cand = Path(args.data_dir).parent / "harness" / "forbidden_vocabulary.json"
        args.forbidden_vocab = str(cand) if cand.exists() else None
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    vocab = load_forbidden_vocab(Path(args.forbidden_vocab) if args.forbidden_vocab else None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        rows = [json.loads(l) for l in Path(args.score_only).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            item = EvalItem(**r["item"])
            parsed = parse_output(r["raw_output"], r.get("pre_parsed_tool_calls"), r.get("finish_reason"))
            r["parsed"] = asdict(parsed)
            r["score"] = score_item(item, parsed, vocab)
        summary = summarize(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_report(summary)
        return

    # アダプタの存在はモデルを読む前に確かめる。14B のロードは 1 分以上かかるうえ、
    # パスを間違えたまま素のモデルで評価が走ると「学習後の結果」として記録されかねない
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").is_file():
        raise SystemExit(
            f"アダプタが見つかりません: {args.adapter}\n"
            f"  adapter_config.json がありません。シェル変数が空のまま展開されていないか確認してください"
        )

    items = load_items(Path(args.data_dir), args.files, args.limit, args.sample, args.sample_seed)
    print(f"{len(items)} 件を評価（レコード {len({i.id for i in items})} 件）: backend={args.backend} model={args.model_name} adapter={args.adapter}", file=sys.stderr)

    # gold 側の自己検査。ここで禁止語彙に当たるなら採点側のバグか語彙リストの問題
    gold_hits = 0
    for it in items:
        texts = _string_values(parse_json_obj(it.gold_content)) if it.task_type in STRUCTURED_TASKS else [it.gold_content]
        gold_hits += bool(count_forbidden(texts, vocab, _context_tokens(it)))
    if gold_hits:
        print(f"警告: gold {gold_hits} 件が禁止語彙に該当。採点条件を確認すること", file=sys.stderr)

    # 生成結果は 1 件ずつ partial に追記する。数時間の run が途中で死んでも、同じ run-name で
    # 再実行すれば残りだけ生成する（--no-resume で最初から）。key は (id, turn)
    partial_path = out_dir / "predictions.partial.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if partial_path.exists():
        if args.resume:
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["id"], r["turn"])] = r
            print(f"再開: {len(done)} 件は生成済み（{partial_path}）", file=sys.stderr)
        else:
            partial_path.unlink()
    todo = [i for i, it in enumerate(items) if (it.id, it.turn) not in done]
    partial_fh = partial_path.open("a", encoding="utf-8")

    def record(i: int, text: str, finish: str | None, pre_parsed: list[dict] | None = None) -> None:
        it = items[i]
        row = {"id": it.id, "turn": it.turn, "raw_output": text, "finish_reason": finish, "pre_parsed_tool_calls": pre_parsed}
        partial_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial_fh.flush()
        done[(it.id, it.turn)] = row

    t0 = time.time()
    tokenizer = None
    prompts: list[str] | None = None
    if args.backend in ("vllm", "hf"):
        from transformers import AutoTokenizer

        # 学習済み adapter には train_hbmss.py が SFT 用テンプレート入りの tokenizer を同梱している。
        # 軌跡の途中ターンの描画（空 <think> の有無）が学習時と揃うよう、そちらを優先する。
        # 素のモデルは公式テンプレートで評価する
        tok_src = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer_config.json").exists() else args.model_name
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        print(f"tokenizer / chat template: {tok_src}", file=sys.stderr)
        prompts = render_prompts(items, tokenizer, args.enable_thinking)
        if todo:
            todo_prompts = [prompts[i] for i in todo]
            if args.backend == "vllm":
                for j, (text, finish) in enumerate(generate_vllm(todo_prompts, args)):
                    record(todo[j], text, finish)
            else:
                generate_hf(todo_prompts, args, tokenizer, on_result=lambda j, text, finish: record(todo[j], text, finish))
    elif todo:
        generate_openai([items[i] for i in todo], args, on_result=lambda j, text, finish, tcs: record(todo[j], text, finish, tcs))
    elapsed = time.time() - t0
    partial_fh.close()

    rows = []
    for idx, it in enumerate(items):
        r = done[(it.id, it.turn)]
        parsed = parse_output(r["raw_output"], r.get("pre_parsed_tool_calls"), r.get("finish_reason"))
        rows.append(
            {
                "item": asdict(it),
                "prompt_tokens": len(tokenizer(prompts[idx])["input_ids"]) if tokenizer and prompts else None,
                "raw_output": r["raw_output"],
                "finish_reason": r.get("finish_reason"),
                "pre_parsed_tool_calls": r.get("pre_parsed_tool_calls"),
                "parsed": asdict(parsed),
                "score": score_item(it, parsed, vocab),
            }
        )

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)

    summary = summarize(rows)
    summary["config"] = {**vars(args), "elapsed_sec": elapsed, "n_forbidden_vocab": len(vocab)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary)
    print(f"\n生成 {elapsed/60:.1f} 分。出力: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
