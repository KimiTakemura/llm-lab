"""HBMSS SFT ランナーの起動口。実体は src/llm_lab/train_hbmss.py（main.py と同じ流儀）。

    .venv/bin/python train_hbmss.py --help
    .venv/bin/python train_hbmss.py --model-name Qwen/Qwen3-4B --load-in-4bit --max-steps 20 --no-wandb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_lab.train_hbmss import main  # noqa: E402

if __name__ == "__main__":
    main()
