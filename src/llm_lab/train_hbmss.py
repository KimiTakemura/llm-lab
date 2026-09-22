"""HBMSS peft-dataset を Qwen3 に LoRA-SFT する実験ランナー。

train.py（OLMo-2 1B + wikitext の継続事前学習ランナー）とは学習の体裁が違うので別ファイルにした。
比較実験の作法は train.py から引き継ぐ:

- eval を 2 系統持つ（indomain = 学習に出ないシナリオ族 S07/S14/S19、heldout = 手書き H01〜H06）
- seed / data_seed を固定してデータ順を全条件で揃える
- warmup + cosine
- LoRA ハイパラを W&B config に明示的に載せる

train.py と違う点:

- 会話形式（messages + tools）を Qwen3 の chat template で描画し、assistant 部分だけに損失を掛ける。
  user 側の JSON が全体の 78% を占めるので、全トークンに掛けると入力の暗記に学習が支配される。
  マスクは templates/qwen3_sft.jinja の {% generation %} マーカーから作る（hbmss_data.py 参照）
- packing はしない（TRL の packing は自前トークン化と両立しない）。長さは最大 3.8K tok なので --max-length 4096 で全件入る
- --load-in-4bit で QLoRA（NF4 + paged AdamW 8bit）。14B は 24 GB GPU、4B は 8 GB GPU で回る想定
- --liger（既定 on）で fused linear cross entropy を使い、152K 語彙 × 4K tok の logits を実体化しない。
  8 GB 機ではこれが無いと 4B でも OOM する。要 `pip install liger-kernel`
- 学習後の判定は eval/loss ではなく eval_hbmss.py の生成採点で行う（--adapter に output_dir を渡す）

使い方:
    # ローカル smoke test（Qwen3-4B, 8 GB）
    .venv/bin/python train_hbmss.py --model-name Qwen/Qwen3-4B --load-in-4bit --max-steps 20 --no-wandb
    # 本番（Qwen3-14B, 24 GB 以上）
    .venv/bin/python train_hbmss.py --model-name Qwen/Qwen3-14B --load-in-4bit --epochs 3 --lr 1e-4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from llm_lab.hbmss_data import DEFAULT_DATA_DIR, build_datasets, load_sft_chat_template
from llm_lab.train import TARGET_MODULE_PRESETS  # Qwen3 の線形層名は OLMo-2 と同じ



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HBMSS peft-dataset の Qwen3 LoRA-SFT ランナー",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model / data")
    g.add_argument("--model-name", default="Qwen/Qwen3-14B")
    g.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="peft-dataset/data のパス（eval/ を含む）。環境変数 HBMSS_DATA_DIR でも指定可")
    g.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes NF4 で読む（QLoRA）")
    g.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "kernels-community/flash-attn", "flash_attention_2", "eager"],
        help="packing しないので sdpa で十分。FA2 は速度・メモリの改善用",
    )
    g.add_argument("--max-length", type=int, default=4096, help="これを超える例は切り詰めず除外する")
    g.add_argument(
        "--eval-size",
        type=int,
        default=160,
        help="indomain eval（480 件）から使う件数。毎回全件だと eval が学習より長くなる。heldout は常に全件",
    )

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=None, help="既定: 2*r")
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--target-modules", default="attn+mlp", choices=sorted(TARGET_MODULE_PRESETS))

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=1e-4, help="QLoRA 14B の相場は 1e-4〜2e-4。train.py の 1e-3 は持ち込まない")
    g.add_argument("--scheduler", default="cosine")
    g.add_argument("--warmup-ratio", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--epochs", type=float, default=2.0, help="SFT なので epoch 単位。--max-steps > 0 ならそちらが優先")
    g.add_argument("--max-steps", type=int, default=-1, help="smoke test / step 固定比較用")
    g.add_argument("--batch-size", type=int, default=1, help="per_device_train_batch_size。長さ 4K なので 1 が基本")
    g.add_argument("--grad-accum", type=int, default=16)
    g.add_argument("--optim", default=None, help="既定: 4bit なら paged_adamw_8bit、それ以外 adamw_torch")
    g.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    g.add_argument("--no-liger", dest="liger", action="store_false", help="liger-kernel を使わない")
    g.add_argument("--no-bf16", dest="bf16", action="store_false", help="fp32 で回す（CPU での配線確認用）")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--split-seed", type=int, default=None, help="eval サブセット抽出の seed（既定: --seed）")

    g = p.add_argument_group("evaluation / logging")
    g.add_argument("--eval-steps", type=int, default=25)
    g.add_argument("--logging-steps", type=int, default=5)
    g.add_argument("--no-eval-on-start", dest="eval_on_start", action="store_false")
    g.add_argument("--no-save-adapter", dest="save_adapter", action="store_false")
    g.add_argument("--output-dir", default=None, help="既定: ./outputs/<run-name>")

    g = p.add_argument_group("W&B")
    g.add_argument("--wandb-project", default="llm-lab-test")
    g.add_argument("--wandb-group", default="hbmss-sft-v1")
    g.add_argument("--wandb-tags", nargs="*", default=[])
    g.add_argument("--run-name", default=None)
    g.add_argument("--no-wandb", dest="use_wandb", action="store_false")

    args = p.parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.optim is None:
        args.optim = "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch"
    if args.run_name is None:
        short = args.model_name.rstrip("/").split("/")[-1]
        length = f"s{args.max_steps}" if args.max_steps > 0 else f"ep{args.epochs:g}"
        args.run_name = (
            f"hbmss-{short}-r{args.lora_r}-a{args.lora_alpha}-lr{args.lr:.0e}"
            f"-{args.target_modules}-wd{args.weight_decay:g}-{length}" + ("-4bit" if args.load_in_4bit else "")
        )
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.run_name}"
    return args


def build_wandb_config(args: argparse.Namespace, target_modules: list[str], total_steps: int) -> dict:
    effective_batch_size = args.batch_size * args.grad_accum
    return {
        "task": "hbmss-sft",
        "model_name": args.model_name,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules_preset": args.target_modules,
        "target_modules": target_modules,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "total_steps": total_steps,
        "optim": args.optim,
        "attn_impl": args.attn_impl,
        "liger": args.liger,
        "effective_batch_size": effective_batch_size,
        "max_length": args.max_length,
        "packing": False,
        "eval_size": args.eval_size,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "save_adapter": args.save_adapter,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_modules = TARGET_MODULE_PRESETS[args.target_modules]
    load_dotenv(".env")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 公式テンプレートには generation マーカーが無いので SFT 用に差し替える。
    # trainer.save_model がこの tokenizer ごと保存するため、推論側は adapter ディレクトリの
    # tokenizer を読めば学習時と同じ描画になる
    tokenizer.chat_template = load_sft_chat_template()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_sets = build_datasets(Path(args.data_dir), tokenizer, args.max_length)
    if args.eval_size and args.eval_size < len(eval_sets["indomain"]):
        # split_seed に紐付けて、学習 seed を振っても評価サブセットが動かないようにする
        eval_sets["indomain"] = eval_sets["indomain"].shuffle(seed=args.split_seed).select(range(args.eval_size))
    meta_cols = ["id", "file", "scenario"]
    train_ds = train_ds.remove_columns(meta_cols)
    eval_sets = {k: v.remove_columns(meta_cols) for k, v in eval_sets.items()}

    # スケジュールは step 数で持つ（transformers v5 で warmup_ratio が非推奨）。
    # epoch 指定のときは総 step 数をここで確定させ、warmup と W&B config に使う
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    args.warmup_steps = round(total_steps * args.warmup_ratio)
    print(f"[train] {len(train_ds)} 件, {steps_per_epoch} step/epoch, 総 {total_steps} step, warmup {args.warmup_steps} step", file=sys.stderr)

    if args.use_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            tags=args.wandb_tags,
            config=build_wandb_config(args, target_modules, total_steps),
        )

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
        attn_implementation=args.attn_impl,
        quantization_config=quant,
        device_map={"": 0} if args.load_in_4bit else None,
    )
    # gradient checkpointing と KV cache は両立しない（警告が出るだけだが明示しておく）
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        init_lora_weights=True,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        # 4bit のときは TRL 側が prepare_model_for_kbit_training で checkpointing を有効化する
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.liger,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=args.eval_on_start,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=args.bf16,
        use_cpu=not torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
        # データは hbmss_data.py でトークン化・マスク済み。TRL 側の整形は通さない。
        # 損失マスクは collator が assistant_masks 列を見て labels に -100 を入れる
        # （assistant_only_loss=True はトークン化済みデータでは受け付けられないので指定しない）
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
        packing=False,  # トークン化済み + skip_prepare_dataset では TRL の packing は通らない
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_sets,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()

    trainer.train()

    if args.save_adapter:
        trainer.save_model(args.output_dir)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
