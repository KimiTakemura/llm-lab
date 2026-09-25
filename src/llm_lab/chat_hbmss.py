"""学習したモデルに手でプロンプトを投げて出力を見るための対話シェル。

採点ではなく観察のための道具。したがって**描画と生成条件は eval_hbmss.py と完全に同じ**にしてある
（同じ chat template、enable_thinking=False、既定は貪欲デコード、max_new_tokens 1536）。
ここで見た挙動がそのまま測定値の説明になるようにするため。

    python chat_hbmss.py --adapter outputs/<run>/checkpoint-340
    python chat_hbmss.py --adapter outputs/<run>/checkpoint-340 --load-in-4bit   # 8 GB 機

入力は複数行を受け付ける。データセットの user 内容は長い JSON なので、貼り付けたあと
単独行の `.` で送信する。`:` で始まる行はコマンド（`:h` で一覧）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llm_lab.eval_hbmss import FILES, load_items
from llm_lab.hbmss_data import DEFAULT_DATA_DIR

HELP = """\
コマンド:
  .                 入力を確定して送信（単独行）
  :item <id>        eval のレコードを読み込んで送信し、gold と並べて表示する
  :item ?           使える id を 20 件表示する
  :file <path>      ファイルの中身を user メッセージとして送信する
  :sys <text>       system プロンプトを差し替える（`:sys -` でデータ既定に戻す）
  :hist             いまの会話履歴を表示する
  :reset            会話履歴を捨てる（system は残る）
  :save <path>      直前のやり取りを JSON で保存する
  :raw              直前の応答を特殊トークン込みの生テキストで表示する
  :q                終了
"""


def _pretty(text: str) -> str:
    """findings JSON なら読める形に整えて返す。JSON でなければそのまま。"""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text


def _default_system(data_dir: Path) -> str | None:
    """データ側の system プロンプトを 1 件目から取る。

    手で打った実験と学習時の条件をずらさないため。ここが違うだけで出力は変わる。
    """
    for name in FILES:
        path = data_dir / "eval" / f"{name}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    for m in json.loads(line)["messages"]:
                        if m["role"] == "system":
                            return m["content"]
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HBMSS モデルの対話シェル", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-name", default="Qwen/Qwen3-14B")
    p.add_argument("--adapter", default=None, help="LoRA アダプタのディレクトリ")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="`:item` と既定 system プロンプトの取得元")
    p.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes NF4。8 GB 機で 14B を動かすとき")
    p.add_argument("--max-new-tokens", type=int, default=1536, help="eval と同じ既定値")
    p.add_argument("--temperature", type=float, default=0.0, help="0 で貪欲デコード。eval と同じ")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--enable-thinking", action="store_true", help="think を有効にする（学習条件と変わるので通常は使わない）")
    p.add_argument("--item", default=None, help="起動直後に読み込む eval レコードの id")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.adapter and not (Path(args.adapter) / "adapter_config.json").is_file():
        raise SystemExit(
            f"アダプタが見つかりません: {args.adapter}\n"
            f"  adapter_config.json がありません。シェル変数が空のまま展開されていないか確認してください"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_dir = Path(args.data_dir)
    items = {}
    if (data_dir / "eval").is_dir():
        items = {it.id: it for it in load_items(data_dir, FILES, None) if it.turn == 0}
        print(f"eval レコード {len(items)} 件を `:item` 用に読み込みました", file=sys.stderr)

    # tokenizer はアダプタ側を優先する。学習時の chat template が同梱されているため
    tok_src = args.adapter or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    print(f"tokenizer / chat template: {tok_src}", file=sys.stderr)

    quant = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # VRAM に載り切らない場合に落ちるより遅くても動く方を選ぶ。8 GB 機で 14B を見るため
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.load_in_4bit:
            model = model.merge_and_unload()
            print("アダプタをベース重みにマージしました", file=sys.stderr)
    model.eval()

    system = _default_system(data_dir)
    history: list[dict] = []
    tools = None
    gold: str | None = None
    last_raw = ""
    last_exchange: dict = {}

    print(f"\nモデル: {args.model_name}  アダプタ: {args.adapter or 'なし'}", file=sys.stderr)
    print(f"system: {'データ既定' if system else 'なし'}  温度: {args.temperature}  max_new_tokens: {args.max_new_tokens}", file=sys.stderr)
    print(HELP, file=sys.stderr)

    def send(user: str) -> None:
        nonlocal last_raw, last_exchange
        msgs = ([{"role": "system", "content": system}] if system else []) + history + [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking
        )
        enc = tokenizer([prompt], return_tensors="pt").to(model.device)
        kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if args.temperature > 0:
            kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        new = out[0, enc["input_ids"].shape[1] :]
        el = time.time() - t0
        last_raw = tokenizer.decode(new, skip_special_tokens=False)
        text = tokenizer.decode(new, skip_special_tokens=True)

        # stderr の付帯情報より先に本文を出し切る。ログに落としたとき順番が入れ替わらないように
        print("\n" + _pretty(text.strip()), flush=True)
        finish = "stop" if out[0, -1].item() in (tokenizer.eos_token_id,) else "length"
        print(
            f"\n[入力 {enc['input_ids'].shape[1]} tok / 生成 {len(new)} tok / {el:.1f} 秒 / "
            f"{len(new) / el:.1f} tok/s / 終了 {finish}]",
            file=sys.stderr,
        )
        if finish == "length":
            print("  ※ 上限で打ち切られています。--max-new-tokens を上げてください", file=sys.stderr)
        if gold is not None:
            print("\n--- gold ---\n" + _pretty(gold), file=sys.stderr)
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": text}])
        last_exchange = {"system": system, "user": user, "assistant": text, "gold": gold, "raw": last_raw}

    buf: list[str] = []
    while True:
        try:
            line = input("... " if buf else ">>> ")
        except EOFError:
            break

        if line.startswith(":"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("h", "help"):
                print(HELP, file=sys.stderr)
            elif cmd == "reset":
                history.clear()
                gold = tools = None
                print("履歴を捨てました", file=sys.stderr)
            elif cmd == "hist":
                for m in history:
                    print(f"[{m['role']}] {m['content'][:400]}", file=sys.stderr)
            elif cmd == "sys":
                system = _default_system(data_dir) if rest == "-" else (rest or None)
                print(f"system を更新しました（{len(system) if system else 0} 文字）", file=sys.stderr)
            elif cmd == "raw":
                print(repr(last_raw), file=sys.stderr)
            elif cmd == "save":
                if not last_exchange:
                    print("まだ保存するものがありません", file=sys.stderr)
                else:
                    Path(rest).write_text(json.dumps(last_exchange, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"保存しました: {rest}", file=sys.stderr)
            elif cmd == "file":
                path = Path(rest)
                if not path.is_file():
                    print(f"ファイルがありません: {rest}", file=sys.stderr)
                else:
                    gold = None
                    send(path.read_text(encoding="utf-8"))
            elif cmd == "item":
                if rest == "?" or not rest:
                    for i, k in enumerate(items):
                        if i >= 20:
                            print(f"  ... 他 {len(items) - 20} 件", file=sys.stderr)
                            break
                        print(f"  {k}", file=sys.stderr)
                elif rest not in items:
                    print(f"その id はありません: {rest}（`:item ?` で一覧）", file=sys.stderr)
                else:
                    it = items[rest]
                    history.clear()
                    system = next((m["content"] for m in it.prefix if m["role"] == "system"), None)
                    tools, gold = it.tools, it.gold_content
                    user = next(m["content"] for m in reversed(it.prefix) if m["role"] == "user")
                    print(f"\n--- {it.id}  {it.task_type} / {it.scenario} / gold {it.outcome} ---", file=sys.stderr)
                    send(user)
            else:
                print(f"不明なコマンド: :{cmd}（`:h` で一覧）", file=sys.stderr)
            continue

        if line.strip() == ".":
            if buf:
                gold = None
                send("\n".join(buf))
                buf = []
            continue
        buf.append(line)

    print("\n終了しました", file=sys.stderr)


if __name__ == "__main__":
    main()
