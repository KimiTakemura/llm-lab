"""HBMSS 評価ランナーの起動口。実体は src/llm_lab/eval_hbmss.py（main.py と同じ流儀）。

    .venv/bin/python eval_hbmss.py --help
    .venv/bin/python eval_hbmss.py --backend vllm --model-name Qwen/Qwen3-4B --load-in-4bit --limit 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_lab.eval_hbmss import main  # noqa: E402

if __name__ == "__main__":
    main()
