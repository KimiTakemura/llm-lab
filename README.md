# llm-lab

LoRA-SFT の実験場。いまの主題は **HBMSS peft-dataset を Qwen3-14B に LoRA で学習させる**こと。

| ファイル | 役割 |
|---|---|
| `train_hbmss.py` → `src/llm_lab/train_hbmss.py` | HBMSS の SFT ランナー（assistant のみ損失、QLoRA / bf16 LoRA、W&B） |
| `eval_hbmss.py` → `src/llm_lab/eval_hbmss.py` | タスク評価。findings JSON と tool_calls を gold と完全一致で比べる |
| `eval_general.py` → `src/llm_lab/eval_general.py` | 汎用能力の回帰テスト。IFEval + 日本語プローブ + 出力契約の漏れ出し |
| `src/llm_lab/hbmss_data.py` | JSONL 読み込みとトークン化。`templates/qwen3_sft.jinja` で損失マスクを作る |
| `main.py` → `src/llm_lab/train.py`, `sweep.yaml` | 旧: OLMo-2 1B + wikitext の LR / epoch 探索 |

## セットアップ

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA に合わせる
uv pip install -r requirements.txt
cp .env.example .env   # WANDB_API_KEY などを記入
```

データは HBMSS リポジトリの `tools/peft-dataset/data`（train + eval）と `harness/forbidden_vocabulary.json`。
場所は `--data-dir` か環境変数 `HBMSS_DATA_DIR` で指定する。

## 使い方

```bash
# 素のモデルのベースライン（生成採点）
python eval_hbmss.py --backend hf --model-name Qwen/Qwen3-14B --batch-size 16

# 学習（48 GB 以上なら --load-in-4bit を外して bf16 LoRA）
python train_hbmss.py --model-name Qwen/Qwen3-14B --epochs 3 --lr 1e-4

# 学習後の採点（adapter 同梱の tokenizer / テンプレートが使われる）
python eval_hbmss.py --backend hf --model-name Qwen/Qwen3-14B --adapter outputs/<run>
```

判定は `eval/indomain_loss` ではなく `eval_hbmss.py` の採点結果（`outputs/eval/<run>/summary.json`）で行う。
タスク性能だけで選ぶと指示追従が壊れていても気づけないので、`eval_general.py` を必ず併せて見る:

```bash
python eval_general.py --backend hf --model-name Qwen/Qwen3-14B --limit 200            # 学習前
python eval_general.py --backend hf --model-name Qwen/Qwen3-14B --adapter outputs/<run> --limit 200
```

`--limit` と `--seed` を揃えれば同じ部分集合で比較できる。`format bleed`（無関係な質問に findings JSON を
返す率）が上がっていたら、タスク score が良くても採用しない。
生成は 1 件ずつ `predictions.partial.jsonl` に追記され、同じ run-name で再実行すれば途中から再開する。
