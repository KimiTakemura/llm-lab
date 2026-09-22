"""OLMo-2 1B の LoRA-SFT 実験ランナー。

条件を変えた複数 run を W&B 上で公平に比較するため、以下を既定の挙動にしている:

- eval split を 2 本切る（in-domain と OOD）。in-domain だけだと「過学習」と
  「汎用能力が壊れた」を区別できないため、判断軸を最初から 2 系統持つ
- seed / data_seed を固定してデータ順を全条件で揃える（差 = 条件の差 を保証）
- warmup + cosine で、序盤の grad_norm スパイクと「LR が枯れただけの平坦」を排除
- LoRA ハイパラを W&B config に明示的に載せ、Runs テーブルで並べ替え可能にする

使い方:
    .venv/bin/python main.py --lr 5e-4 --lora-r 64 --target-modules attn+mlp

環境要件:
    FlashAttention 2 は `kernels` 経由で Hub のプリビルドを使う（flash-attn のビルド不要）。
    バージョンは transformers 5.5 の要求どおり `kernels>=0.12.0,<0.13` に固定すること。
    0.16 系を入れると LayerRepository の API 変更で transformers の import 自体が壊れる。
"""

from __future__ import annotations

import argparse
import os

import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# OLMo-2 の線形層名。Attention のみか MLP まで含めるかは効果が大きい比較軸なので
# プリセットとして切り替えられるようにしている。
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULE_PRESETS = {
    "attn": ATTN_MODULES,
    "mlp": MLP_MODULES,
    "attn+mlp": ATTN_MODULES + MLP_MODULES,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OLMo-2 1B の LoRA-SFT 実験ランナー",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model / data")
    g.add_argument("--model-name", default="allenai/OLMo-2-0425-1B")
    g.add_argument("--dataset", default="Salesforce/wikitext")
    g.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    g.add_argument("--dataset-split", default="train")
    g.add_argument("--text-field", default="text", help="学習に使うテキストカラム名")
    g.add_argument(
        "--attn-impl",
        default="kernels-community/flash-attn",
        choices=["kernels-community/flash-attn", "flash_attention_2", "sdpa", "eager"],
        help=(
            "packing 時に文書境界を越えた汚染を断てるのは FlashAttention 系のみ。"
            "既定は Hub kernels 経由の FA2（flash-attn のビルド不要）"
        ),
    )
    g.add_argument("--eval-size", type=int, default=500, help="学習データから切り出す評価サンプル数")

    g = p.add_argument_group("OOD evaluation")
    g.add_argument(
        "--ood-dataset",
        default="stanfordnlp/imdb",
        help=(
            "学習分布の外側を測るための第 2 評価セット。既定は映画レビュー（口語）で、"
            "学習側の Wikipedia とドメインが十分に離れている"
        ),
    )
    g.add_argument("--ood-config", default=None)
    g.add_argument("--ood-split", default="test")
    g.add_argument("--ood-text-field", default=None, help="既定: --text-field と同じ")
    g.add_argument("--ood-size", type=int, default=500)
    g.add_argument(
        "--no-ood-eval",
        dest="ood_eval",
        action="store_false",
        help="OOD 評価を行わない。metric 名が eval/loss に戻る",
    )

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16, help="LoRA ランク。大きいほど表現力が上がる")
    g.add_argument("--lora-alpha", type=int, default=None, help="スケーリング係数（既定: 2*r）")
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument(
        "--target-modules",
        default="attn+mlp",
        choices=sorted(TARGET_MODULE_PRESETS),
        help="LoRA を挿す線形層。MLP を含めると知識側にも効く",
    )

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=2e-4, help="学習率。最も効く比較軸")
    g.add_argument("--scheduler", default="cosine", help="lr_scheduler_type")
    g.add_argument("--warmup-ratio", type=float, default=0.03, help="序盤の grad_norm スパイク対策")
    g.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW の decoupled weight decay。既定 0.0 は HF の既定値と同じ",
    )
    g.add_argument("--max-steps", type=int, default=300, help="全条件で揃えること")
    g.add_argument("--batch-size", type=int, default=2, help="per_device_train_batch_size")
    g.add_argument("--grad-accum", type=int, default=8, help="gradient_accumulation_steps")
    g.add_argument("--max-length", type=int, default=512, help="1 シーケンスあたりの最大トークン長")
    g.add_argument("--no-packing", dest="packing", action="store_false", help="packing を無効化")
    g.add_argument("--seed", type=int, default=42, help="seed と data_seed の両方に使う")
    g.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "train/eval split の seed（既定: --seed と同じ）。"
            "seed 間のばらつきを測るとき、評価セットを固定したまま学習 seed だけ振るのに使う"
        ),
    )

    g = p.add_argument_group("evaluation / logging")
    g.add_argument("--eval-steps", type=int, default=25)
    g.add_argument("--logging-steps", type=int, default=5)
    g.add_argument(
        "--no-eval-on-start",
        dest="eval_on_start",
        action="store_false",
        help="step 0（= ベースモデル素）の eval/loss を取らない",
    )
    # 既定 on。長いランを保存せずに捨てると、あとから再評価する手段が完全に失われる。
    g.add_argument("--save-adapter", action="store_true", help="学習後に LoRA アダプタを保存する（既定）")
    g.add_argument(
        "--no-save-adapter",
        dest="save_adapter",
        action="store_false",
        help="アダプタを保存しない（使い捨ての smoke test 用）",
    )
    p.set_defaults(save_adapter=True)
    g.add_argument("--output-dir", default=None, help="既定: ./outputs/<run-name>")

    g = p.add_argument_group("W&B")
    g.add_argument("--wandb-project", default="llm-lab-test")
    g.add_argument("--wandb-group", default="lora-sweep-v1", help="比較したい run 群でこれを揃える")
    g.add_argument("--wandb-tags", nargs="*", default=[])
    g.add_argument("--run-name", default=None, help="既定: ハイパラから自動生成")
    g.add_argument("--no-wandb", dest="use_wandb", action="store_false")

    args = p.parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    # 既定では従来どおり seed と一致させる。明示したときだけ split を固定して学習 seed を振れる。
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.ood_text_field is None:
        args.ood_text_field = args.text_field
    if args.run_name is None:
        args.run_name = (
            f"r{args.lora_r}-a{args.lora_alpha}-lr{args.lr:.0e}"
            f"-{args.target_modules}-wd{args.weight_decay:g}-s{args.max_steps}"
        )
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.run_name}"
    # transformers v5 で warmup_ratio が非推奨になったため、ここで step 数に落とす。
    # 比率で持っておくと max_steps を変えてもスケジュールの形が保たれる。
    args.warmup_steps = round(args.max_steps * args.warmup_ratio)
    return args


def build_wandb_config(args: argparse.Namespace, target_modules: list[str]) -> dict:
    """W&B の Runs テーブル / parallel coordinates で軸として使う値をまとめる。

    SFTConfig は HF の WandbCallback が自動でログするが、LoRA 側（r / alpha /
    target_modules）は入らない。条件を定義する値はここで明示的に載せる。
    """
    effective_batch_size = args.batch_size * args.grad_accum
    return {
        "model_name": args.model_name,
        "dataset": f"{args.dataset}:{args.dataset_config}",
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
        "attn_impl": args.attn_impl,
        "max_steps": args.max_steps,
        "effective_batch_size": effective_batch_size,
        "max_length": args.max_length,
        "packing": args.packing,
        # step 数ではなく計算量で条件を横並びにするための派生値
        "tokens_per_step": effective_batch_size * args.max_length,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "ood_dataset": args.ood_dataset if args.ood_eval else None,
        "save_adapter": args.save_adapter,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_modules = TARGET_MODULE_PRESETS[args.target_modules]

    load_dotenv(".env")

    if args.use_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        # Trainer より先に init しておくと HF の WandbCallback はこの run を再利用する。
        # こうしないと LoRA 側の config を載せられない。
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            tags=args.wandb_tags,
            config=build_wandb_config(args, target_modules),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 既定で Hub kernels のリポジトリ ID を直に渡している。"flash_attention_2" を渡すと
    # transformers 5.5 が内部で "kernels-community/flash-attn2" に書き換えるが、TRL 0.24 の
    # 許可リストは "kernels-community/flash-attn"（末尾 2 なし）なので、実際は FA2 が
    # 効いているのに packing 汚染の警告が誤発生する。ID を直指定して名前を一致させる。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )

    # ベースモデル（chat_template未設定）なのでプレーンテキストで動作確認する
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    dataset = dataset.filter(lambda ex: len(ex[args.text_field].strip()) > 0)
    # split を固定することで、学習 seed を振っても全 run が同一の評価セットを見る。
    # ここを args.seed にすると「学習のばらつき」と「評価セットのばらつき」が混ざる。
    splits = dataset.train_test_split(test_size=args.eval_size, seed=args.split_seed)

    # 判断軸を 2 本にする。in-domain だけでは「学習分布に過学習した」のか
    # 「モデルとして壊れた」のかが区別できない。dict を渡すと HF が
    # eval_<key>_loss を個別に出すので、W&B 上では eval/indomain_loss と
    # eval/ood_loss の 2 系列になる（= 旧来の eval/loss は消える）。
    eval_datasets = {"indomain": splits["test"]}
    if args.ood_eval:
        ood = load_dataset(args.ood_dataset, args.ood_config, split=args.ood_split)
        ood = ood.filter(lambda ex: len(ex[args.ood_text_field].strip()) > 0)
        # split_seed に紐付けて、学習 seed を振っても OOD 側が動かないようにする
        ood = ood.shuffle(seed=args.split_seed)
        ood = ood.select(range(min(args.ood_size, len(ood))))
        if args.ood_text_field != args.text_field:
            ood = ood.rename_column(args.ood_text_field, args.text_field)
        # label 等の余計なカラムを落としておかないと SFTTrainer の整形で邪魔になる
        eval_datasets["ood"] = ood.select_columns([args.text_field])

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
        max_steps=args.max_steps,
        learning_rate=args.lr,
        # 線形減衰 + warmup なしだと「収束」と「LR が枯れた」が区別できないため cosine + warmup
        lr_scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        # 条件比較の判断軸。train/loss は下げようと思えばいくらでも下がる
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # step 0 = ベースモデル素の性能。全条件共通のアンカーになる
        eval_on_start=args.eval_on_start,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        # データ順まで揃えて「差 = 条件の差」にする
        seed=args.seed,
        data_seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
        dataset_text_field=args.text_field,
        max_length=args.max_length,
        packing=args.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        # OOD を切ったときは素の dataset を渡す。dict のままだと 1 本でも
        # eval_indomain_loss になり、旧 run の eval/loss と名前が揃わない。
        eval_dataset=eval_datasets if args.ood_eval else splits["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    if args.save_adapter:
        trainer.save_model(args.output_dir)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
